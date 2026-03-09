#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/planner.hpp"
#include "core/memory.hpp"
#include "core/kernels.hpp"
#include "core/debug.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstring>
#include <stdexcept>

class Executor
{
private:
    CompiledGraph compiled;
    MemoryManager &memManager;
    const Graph &graph;

public:
    Executor(CompiledGraph cg, MemoryManager &mm, const Graph &g)
        : compiled(std::move(cg)), memManager(mm), graph(g) {}

    void run(const std::unordered_map<uint32_t, const void *> &inputs,
             const DirtyBucket &bucket)
    {
        std::cout << "running..." << std::endl;
        std::unordered_set<uint32_t> neededNodes;
        if (!compiled.instructions.empty())
        {
            neededNodes.insert(compiled.instructions.back().nodeId);
        }

        // Pass 1: Backward Liveness (Prune unneeded nodes when cached)
        for (auto it = compiled.instructions.rbegin(); it != compiled.instructions.rend(); ++it)
        {
            uint32_t nodeId = it->nodeId;
            const TensorNode &node = compiled.nodesMap.at(nodeId);
            auto regionIt = bucket.regions.find(nodeId);
            bool isDirty = (regionIt != bucket.regions.end() && !regionIt->second.empty());
            bool inCache = memManager.has(it->backend, nodeId);
            if (neededNodes.count(nodeId))
            {
                if (isDirty || !inCache)
                {
                    for (uint32_t pId : it->inputNodeIds)
                    {
                        neededNodes.insert(pId);
                        const TensorNode &pNode = compiled.nodesMap.at(nodeId);
                        regionIt = bucket.regions.find(pId);
                        isDirty = (regionIt != bucket.regions.end() && !regionIt->second.empty());
                        inCache = memManager.has(pNode.backend, pId);
                        if (!isDirty && inCache)
                        {
                            // Lock clean nodes that are in cache
                            auto &outBuf = memManager.buffers.at(pNode.backend);
                            auto blockIt = outBuf.allocationMap.at(pId);
                            blockIt->refCount = compiled.refCounts[nodeId];
                            blockIt->isLocked = true;
                        }
                    }
                }
            }
        }

        // Pass 2: Execute Compiled Instructions (bucket-aware)
        uint32_t instIdx = 0;
        uint32_t nPartial = 0;
        for (const OpInstruction &inst : compiled.instructions)
        {
            instIdx++;
            const uint32_t nodeId = inst.nodeId;
            const TensorNode &node = compiled.nodesMap.at(nodeId);
            auto &outBuf = memManager.buffers.at(node.backend);

            auto regionIt = bucket.regions.find(inst.nodeId);
            bool isDirty = (regionIt != bucket.regions.end() && !regionIt->second.empty());
            bool inCache = memManager.has(inst.backend, nodeId);
            bool isNeeded = neededNodes.count(nodeId);

            // Pruned node (e.g. its child was fully cached and clean)
            // TODO: release this memory before execution loop
            if (!isNeeded)
            {
                for (uint32_t inId : inst.inputNodeIds)
                {
                    memManager.release(compiled.nodesMap.at(inId).backend, inId);
                }
                continue;
            }

            // 1. Skip if clean and already in memory
            if (!isDirty && inCache)
            {
                auto blockIt = outBuf.allocationMap.at(nodeId);
                blockIt->refCount = compiled.refCounts[nodeId];
                blockIt->isLocked = true;
                for (uint32_t inId : inst.inputNodeIds)
                {
                    memManager.release(compiled.nodesMap.at(inId).backend, inId);
                }
                continue;
            }

            // 2. RESOLVE INPUTS FIRST (Before metadata changes/allocations)
            // We fetch pointers and views now while parent IDs are still in the allocationMap
            struct ResolvedInput
            {
                const void *ptr;
                TensorView view;
            };
            std::vector<ResolvedInput> resolvedInputs;
            for (uint32_t inId : inst.inputNodeIds)
            {
                const TensorNode &inNode = compiled.nodesMap.at(inId);
                resolvedInputs.push_back({memManager.buffers.at(inNode.backend).arena_ptr + memManager.buffers.at(inNode.backend).getOffset(inId),
                                          memManager.getView(inNode)});
            }

            // 3. ALLOCATE / TRANSFER OUTPUT
            if (!inCache)
            {
                if (inst.inplaceInputIndex >= 0)
                {
                    uint32_t srcId = inst.inputNodeIds[inst.inplaceInputIndex];
                    memManager.transferOwnership(inst.backend, srcId, inst.nodeId);
                }
                else
                {
                    uint64_t sizeBytes = getSizeBytes(node.shape, node.dtype);
                    memManager.allocate(inst.backend, inst.nodeId, sizeBytes, StorageType::TRANSIENT, compiled.refCounts[inst.nodeId], 0.0f); // TODO: pass in actual cost
                }
            }

            // Ensure output is locked
            auto outBlockIt = outBuf.allocationMap.at(inst.nodeId);
            outBlockIt->refCount = compiled.refCounts[inst.nodeId];
            outBlockIt->isLocked = true;

            // 4. EXECUTE KERNEL

            // Determine compute regions and their targeted kernels
            std::vector<Region> computeRegions;
            std::vector<uint64_t> computeKernels;

            if (isDirty && inCache)
            {
                // Partial compute: only the dirty regions
                auto regionIt = bucket.regions.find(inst.nodeId);
                computeRegions = regionIt->second;
                nPartial++;

                auto kIt = bucket.kernelIds.find(inst.nodeId);
                if (kIt != bucket.kernelIds.end())
                {
                    computeKernels = kIt->second;
                }
                else
                {
                    computeKernels.assign(computeRegions.size(), inst.kernelId);
                }
            }
            else
            {
                // Full compute: build a full-extent region
                Region full;
                for (uint32_t dim : node.shape)
                {
                    full.region.push_back({0, dim});
                }
                computeRegions = {full};
                computeKernels = {inst.kernelId};
            }

            auto slicesIt = bucket.inputSlices.find(inst.nodeId);

            // Execute kernel for each compute region
            for (size_t rIdx = 0; rIdx < computeRegions.size(); ++rIdx)
            {
                const Region &outRegion = computeRegions[rIdx];
                const KernelEntry &kernel = KernelRegistry::get().getKernel(computeKernels[rIdx]);

                bool isFullRegion = true;
                for (size_t d = 0; d < outRegion.region.size(); ++d)
                {
                    if (outRegion.region[d].start != 0 || outRegion.region[d].stop != node.shape[d])
                    {
                        isFullRegion = false;
                        break;
                    }
                }

                // Build input pointers and views
                std::vector<const void *> kernelInputs;
                std::vector<TensorView> kernelInViews;

                for (size_t pIdx = 0; pIdx < inst.inputNodeIds.size(); ++pIdx)
                {
                    uint32_t inId = inst.inputNodeIds[pIdx];
                    const TensorNode &inNode = compiled.nodesMap.at(inId);
                    auto &inBuf = memManager.buffers.at(inNode.backend);

                    TensorView inView = resolvedInputs[pIdx].view;

                    // Apply input slicing if partial and slices are available
                    if (!isFullRegion && slicesIt != bucket.inputSlices.end() && rIdx < slicesIt->second.size() && pIdx < slicesIt->second[rIdx].size() && !slicesIt->second[rIdx][pIdx].empty())
                    {
                        const Region &inputSlice = slicesIt->second[rIdx][pIdx][0];

                        // Build a sliced view: adjust baseOffset and shape
                        TensorView slicedView = inView;
                        uint64_t elementSize = getDTypeSize(inNode.dtype);

                        // Compute new baseOffset from slice starts using strides
                        uint64_t extraOffset = 0;
                        for (size_t d = 0; d < inputSlice.region.size() && d < slicedView.strides.size(); ++d)
                        {
                            extraOffset += inputSlice.region[d].start * static_cast<uint64_t>(slicedView.strides[d]);
                        }
                        slicedView.baseOffset += extraOffset * elementSize;

                        // Update shape to the slice extent
                        for (size_t d = 0; d < inputSlice.region.size() && d < slicedView.shape.size(); ++d)
                        {
                            slicedView.shape[d] = inputSlice.region[d].stop - inputSlice.region[d].start;
                        }

                        // Recalculate strides for the sliced shape
                        slicedView.strides = TensorView::calcContiguousStrides(slicedView.shape);

                        kernelInputs.push_back(inBuf.arena_ptr + slicedView.baseOffset);
                        kernelInViews.push_back(slicedView);
                    }
                    else
                    {
                        kernelInputs.push_back(inBuf.arena_ptr + inView.baseOffset);
                        kernelInViews.push_back(inView);
                    }
                }

                // Build output pointer and view
                std::vector<void *> kernelOutputs;
                std::vector<TensorView> kernelOutViews;

                TensorView outView = memManager.getView(node);

                if (!isFullRegion)
                {
                    uint64_t elementSize = getDTypeSize(node.dtype);
                    uint64_t extraOffset = 0;
                    for (size_t d = 0; d < outRegion.region.size() && d < outView.strides.size(); ++d)
                    {
                        extraOffset += outRegion.region[d].start * static_cast<uint64_t>(outView.strides[d]);
                    }
                    outView.baseOffset += extraOffset * elementSize;

                    for (size_t d = 0; d < outRegion.region.size() && d < outView.shape.size(); ++d)
                    {
                        outView.shape[d] = outRegion.region[d].stop - outRegion.region[d].start;
                    }

                    outView.strides = TensorView::calcContiguousStrides(outView.shape);
                }

                kernelOutputs.push_back(outBuf.arena_ptr + outView.baseOffset);
                kernelOutViews.push_back(outView);

                // Run the kernel
                for (uint32_t inId : inst.inputNodeIds)
                {
                    Debug::checkNan(compiled.nodesMap.at(inId), memManager, "Kernel Input: " + std::to_string(inId));
                }
                kernel.run(kernelInputs, kernelOutputs, kernelInViews, kernelOutViews);
                Debug::checkNan(compiled.nodesMap.at(inst.nodeId), memManager, "Kernel Output: " + std::to_string(inst.nodeId));
            }

            // 5. Release Consumed Parents
            for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
            {
                if (static_cast<int>(i) == inst.inplaceInputIndex && !inCache)
                    continue;

                uint32_t inId = inst.inputNodeIds[i];
                memManager.release(compiled.nodesMap.at(inId).backend, inId);
            }

            std::cout << instIdx << "/" << compiled.instructions.size() << ", #part: " << nPartial << "\r";
        }
    }
};