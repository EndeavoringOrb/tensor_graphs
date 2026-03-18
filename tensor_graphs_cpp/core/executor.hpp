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
    MemoryManager &memManager;
    const Graph &graph;
    std::unordered_map<uint32_t, float> nodeCosts;

public:
    Executor(MemoryManager &mm, const Graph &g)
        : memManager(mm), graph(g) {}

    void run(const std::unordered_map<uint32_t, const void *> &inputs,
             const CompiledGraph &compiled, const DirtyBucket &bucket)
    {
        std::cout << "running..." << std::endl;

        auto getLogicalId = [&](uint32_t physId)
        {
            auto it = compiled.logicalNodeMap.find(physId);
            return it != compiled.logicalNodeMap.end() ? it->second : physId;
        };

        std::unordered_map<uint32_t, std::vector<uint32_t>> parentMap;
        for (const auto &pair : compiled.nodesMap)
        {
            parentMap[pair.first] = pair.second.parentIds;
        }

        std::unordered_set<uint32_t> neededNodes;
        if (!compiled.instructions.empty())
        {
            neededNodes.insert(compiled.instructions.back().nodeId);
        }

        for (auto it = compiled.instructions.rbegin(); it != compiled.instructions.rend(); ++it)
        {
            uint32_t nodeId = it->nodeId;
            uint32_t logicalId = getLogicalId(nodeId);
            const TensorNode &node = compiled.nodesMap.at(nodeId);
            auto regionIt = bucket.regions.find(logicalId);
            bool isDirty = (regionIt != bucket.regions.end() && !regionIt->second.empty());
            bool inCache = memManager.has(it->backend, nodeId);
            if (neededNodes.count(nodeId))
            {
                if (isDirty || !inCache)
                {
                    for (uint32_t pId : it->inputNodeIds)
                    {
                        neededNodes.insert(pId);
                        const TensorNode &pNode = compiled.nodesMap.at(pId);
                        uint32_t pLogicalId = getLogicalId(pId);
                        regionIt = bucket.regions.find(pLogicalId);
                        isDirty = (regionIt != bucket.regions.end() && !regionIt->second.empty());
                        inCache = memManager.has(pNode.backend, pId);
                        if (!isDirty && inCache)
                        {
                            auto &outBuf = memManager.buffers.at(pNode.backend);
                            auto blockIt = outBuf.allocationMap.at(pId);
                            blockIt->refCount = compiled.refCounts.at(pId);
                            blockIt->isLocked = true;
                        }
                    }
                }
            }
        }

        uint32_t instIdx = 0;
        uint32_t nPartial = 0;
        for (const OpInstruction &inst : compiled.instructions)
        {
            instIdx++;
            const uint32_t nodeId = inst.nodeId;
            uint32_t logicalId = getLogicalId(nodeId);
            const TensorNode &node = compiled.nodesMap.at(nodeId);
            auto &outBuf = memManager.buffers.at(node.backend);

            auto regionIt = bucket.regions.find(logicalId);
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

            if (!isDirty && inCache)
            {
                auto blockIt = outBuf.allocationMap.at(nodeId);
                blockIt->refCount = compiled.refCounts.at(nodeId);
                blockIt->isLocked = true;
                for (uint32_t inId : inst.inputNodeIds)
                {
                    memManager.release(compiled.nodesMap.at(inId).backend, inId);
                }
                continue;
            }

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

            if (inst.inplaceInputIndex >= 0)
            {
                uint32_t srcId = inst.inputNodeIds[inst.inplaceInputIndex];
                memManager.transferOwnership(inst.backend, srcId, inst.nodeId);
            }
            else if (!inCache)
            {
                uint64_t sizeBytes = getSizeBytes(node.shape, node.dtype);
                float cost = compiled.nodeCosts.at(inst.nodeId);
                memManager.allocate(inst.backend, inst.nodeId, sizeBytes, StorageType::TRANSIENT, compiled.refCounts.at(inst.nodeId), cost, &parentMap, &compiled.nodeCosts);
            }

            auto outBlockIt = outBuf.allocationMap.at(inst.nodeId);
            outBlockIt->refCount = compiled.refCounts.at(inst.nodeId);
            outBlockIt->isLocked = true;

            std::vector<Region> computeRegions;
            std::vector<uint64_t> computeKernels;

            if (isDirty && inCache)
            {
                auto regionIt2 = bucket.regions.find(logicalId);
                computeRegions = regionIt2->second;
                nPartial++;

                if (inst.cachedKernelIds.size() == computeRegions.size())
                {
                    computeKernels = inst.cachedKernelIds;
                }
                else
                {
                    computeKernels.assign(computeRegions.size(), inst.fullKernelId);
                }
            }
            else
            {
                Region full;
                for (uint32_t dim : node.shape)
                {
                    full.region.push_back({0, dim});
                }
                computeRegions = {full};
                computeKernels = {inst.fullKernelId};
            }

            auto slicesIt = bucket.inputSlices.find(logicalId);

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

                std::vector<const void *> kernelInputs;
                std::vector<TensorView> kernelInViews;

                for (size_t pIdx = 0; pIdx < inst.inputNodeIds.size(); ++pIdx)
                {
                    uint32_t inId = inst.inputNodeIds[pIdx];
                    const TensorNode &inNode = compiled.nodesMap.at(inId);
                    auto &inBuf = memManager.buffers.at(inNode.backend);

                    TensorView inView = resolvedInputs[pIdx].view;

                    if (!isFullRegion && slicesIt != bucket.inputSlices.end() && pIdx < slicesIt->second.size() && rIdx < slicesIt->second[pIdx].size() && !slicesIt->second[pIdx][rIdx].empty())
                    {
                        const Region &inputSlice = slicesIt->second[pIdx][rIdx];

                        TensorView slicedView = inView;
                        uint64_t elementSize = getDTypeSize(inNode.dtype);

                        uint64_t extraOffset = 0;
                        for (size_t d = 0; d < inputSlice.region.size() && d < slicedView.strides.size(); ++d)
                        {
                            extraOffset += inputSlice.region[d].start * static_cast<uint64_t>(slicedView.strides[d]);
                        }
                        slicedView.baseOffset += extraOffset * elementSize;

                        for (size_t d = 0; d < inputSlice.region.size() && d < slicedView.shape.size(); ++d)
                        {
                            slicedView.shape[d] = inputSlice.region[d].stop - inputSlice.region[d].start;
                        }
                        kernelInputs.push_back(inBuf.arena_ptr + slicedView.baseOffset);
                        kernelInViews.push_back(slicedView);
                    }
                    else
                    {
                        kernelInputs.push_back(inBuf.arena_ptr + inView.baseOffset);
                        kernelInViews.push_back(inView);
                    }
                }

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
                }

                kernelOutputs.push_back(outBuf.arena_ptr + outView.baseOffset);
                kernelOutViews.push_back(outView);

                for (uint32_t inId : inst.inputNodeIds)
                {
                    Debug::checkNan(compiled.nodesMap.at(inId), memManager, "Kernel Input: " + std::to_string(inId));
                }
                kernel.run(kernelInputs, kernelOutputs, kernelInViews, kernelOutViews);
                Debug::checkNan(compiled.nodesMap.at(inst.nodeId), memManager, "Kernel Output: " + std::to_string(inst.nodeId));
            }

            for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
            {
                if (static_cast<int>(i) == inst.inplaceInputIndex)
                    continue;

                uint32_t inId = inst.inputNodeIds[i];
                memManager.release(compiled.nodesMap.at(inId).backend, inId);
            }

            std::cout << instIdx << "/" << compiled.instructions.size() << ", #part: " << nPartial << "\r";
        }
    }
};
