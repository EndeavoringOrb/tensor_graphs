#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/planner.hpp"
#include "core/memory.hpp"
#include "core/kernels.hpp"
#include <unordered_map>
#include <vector>
#include <cstring>
#include <stdexcept>

class Executor
{
private:
    CompiledGraph compiled;
    MemoryManager &memManager;
    const Graph &graph;

    TensorView getEffectiveView(const TensorNode &node) const
    {
        if (!node.view.shape.empty())
        {
            return node.view;
        }
        TensorView v;
        v.baseOffset = 0;
        v.shape = node.shape;
        v.strides = TensorView::calcContiguousStrides(node.shape);
        return v;
    }

public:
    Executor(CompiledGraph cg, MemoryManager &mm, const Graph &g)
        : compiled(std::move(cg)), memManager(mm), graph(g) {}

    void run(const std::unordered_map<uint32_t, const void *> &inputs,
             const DirtyBucket &bucket)
    {
        // 1. Setup Runtime Inputs (Dynamic Transients)
        for (const auto &pair : inputs)
        {
            uint32_t nodeId = pair.first;
            const void *dataPtr = pair.second;

            const TensorNode &node = graph.nodes[nodeId];
            if (node.opType != OpType::INPUT)
                continue;

            uint64_t sizeBytes = getSizeBytes(node.shape, node.dtype);
            uint32_t refs = compiled.refCounts[nodeId];

            uint64_t offset = memManager.allocate(node.backend, nodeId, sizeBytes, StorageType::TRANSIENT, refs, 0.0f);

            auto &buf = memManager.buffers.at(node.backend);
            std::memcpy(buf.arena.data() + offset, dataPtr, sizeBytes);
        }

        // 2. Execute Compiled Instructions (bucket-aware)
        for (const OpInstruction &inst : compiled.instructions)
        {
            const TensorNode &node = graph.nodes[inst.nodeId];

            // Determine dirty status from bucket
            auto regionIt = bucket.regions.find(inst.nodeId);
            bool isDirty = (regionIt != bucket.regions.end() && !regionIt->second.empty());

            // Check if output is already cached in memory
            auto &outBuf = memManager.buffers.at(node.backend);
            bool inCache = (outBuf.allocationMap.find(node.id) != outBuf.allocationMap.end());

            // Decision: skip if clean and cached
            if (!isDirty && inCache)
            {
                // Re-lock the existing allocation so it survives this iteration
                auto blockIt = outBuf.allocationMap.at(node.id);
                blockIt->refCount = compiled.refCounts[inst.nodeId];
                blockIt->isLocked = true;

                // Still release parent refs
                for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
                {
                    if (static_cast<int>(i) == inst.inplaceInputIndex)
                        continue;
                    uint32_t inId = inst.inputNodeIds[i];
                    memManager.release(graph.nodes[inId].backend, inId);
                }
                continue;
            }

            const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.kernelId);

            // Determine compute regions
            std::vector<Region> computeRegions;
            if (isDirty && inCache)
            {
                // Partial compute: only the dirty regions
                computeRegions = regionIt->second;
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
            }

            // 2.b Allocate / Transfer memory for the Output
            if (!inCache)
            {
                if (inst.inplaceInputIndex >= 0)
                {
                    uint32_t srcId = inst.inputNodeIds[inst.inplaceInputIndex];
                    memManager.transferOwnership(inst.backend, srcId, inst.nodeId);

                    auto blockIt = outBuf.allocationMap.at(inst.nodeId);
                    blockIt->refCount = compiled.refCounts[inst.nodeId];
                    blockIt->isLocked = true;
                }
                else
                {
                    uint64_t sizeBytes = getSizeBytes(node.shape, node.dtype);
                    uint32_t refs = compiled.refCounts[inst.nodeId];
                    memManager.allocate(inst.backend, inst.nodeId, sizeBytes, StorageType::TRANSIENT, refs, 0.0f);
                }
            }
            else
            {
                // Already cached but dirty: re-lock
                auto blockIt = outBuf.allocationMap.at(node.id);
                blockIt->refCount = compiled.refCounts[inst.nodeId];
                blockIt->isLocked = true;
            }

            // Get the bucket's inputSlices for this node (if available)
            auto slicesIt = bucket.inputSlices.find(inst.nodeId);

            // 2.c Execute kernel for each compute region
            for (size_t rIdx = 0; rIdx < computeRegions.size(); ++rIdx)
            {
                const Region &outRegion = computeRegions[rIdx];

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
                    const TensorNode &inNode = graph.nodes[inId];
                    auto &inBuf = memManager.buffers.at(inNode.backend);
                    auto it = inBuf.allocationMap.find(inId);

                    if (it == inBuf.allocationMap.end())
                    {
                        throw std::runtime_error("Input node not found in memory allocation map");
                    }

                    uint64_t inOffset = it->second->offset;
                    TensorView inView = getEffectiveView(inNode);

                    // Apply input slicing if partial and slices are available
                    if (!isFullRegion && slicesIt != bucket.inputSlices.end()
                        && rIdx < slicesIt->second.size()
                        && pIdx < slicesIt->second[rIdx].size()
                        && !slicesIt->second[rIdx][pIdx].empty())
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

                        kernelInputs.push_back(inBuf.arena.data() + inOffset + slicedView.baseOffset);
                        kernelInViews.push_back(slicedView);
                    }
                    else
                    {
                        kernelInputs.push_back(inBuf.arena.data() + inOffset + inView.baseOffset);
                        kernelInViews.push_back(inView);
                    }
                }

                // Build output pointer and view
                std::vector<void *> kernelOutputs;
                std::vector<TensorView> kernelOutViews;

                uint64_t outOffset = outBuf.allocationMap.at(node.id)->offset;
                TensorView outView = getEffectiveView(node);

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

                kernelOutputs.push_back(outBuf.arena.data() + outOffset + outView.baseOffset);
                kernelOutViews.push_back(outView);

                // Run the kernel
                kernel.run(kernelInputs, kernelOutputs, kernelInViews, kernelOutViews);
            }

            // 2.e Release Consumed Parents
            for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
            {
                if (static_cast<int>(i) == inst.inplaceInputIndex)
                    continue;

                uint32_t inId = inst.inputNodeIds[i];
                memManager.release(graph.nodes[inId].backend, inId);
            }
        }
    }
};