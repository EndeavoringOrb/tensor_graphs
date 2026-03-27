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
    std::unordered_map<uint32_t, float> nodeCosts;

public:
    Executor(MemoryManager &mm)
        : memManager(mm) {}

    void run(const std::unordered_map<uint32_t, const void *> &inputs,
             const CompiledGraph &compiled, const DirtyBucket &bucket)
    {
        std::cout << "running..." << std::endl;

        std::unordered_map<uint32_t, std::vector<uint32_t>> parentMap;
        for (const auto &pair : compiled.nodesMap)
        {
            parentMap[pair.first] = pair.second.parentIds;
        }

        uint32_t instIdx = 0;
        uint32_t nPartial = 0;
        for (size_t idx = 0; idx < compiled.instructions.size(); ++idx)
        {
            const OpInstruction &inst = compiled.instructions[idx];

            // Check for interrupt signal at each instruction boundary
            if (InterruptManager::isInterrupted())
            {
                std::cerr << "\n[Executor] Interrupt detected, aborting execution..." << std::endl;
                InterruptManager::cleanup();
                std::exit(SIGINT);
            }

            instIdx++;
            const uint32_t nodeId = inst.nodeId;
            uint32_t logicalId = compiled.getLogicalId(nodeId);
            const TensorNode &node = compiled.nodesMap.at(nodeId);
            auto &outBuf = memManager.buffers.at(node.backend);

            struct ResolvedInput
            {
                const void *ptr;
                TensorView view;
                uint32_t allocationId;
            };
            std::vector<ResolvedInput> resolvedInputs;
            for (uint32_t inId : inst.inputNodeIds)
            {
                const TensorNode &inNode = compiled.nodesMap.at(inId);
                uint32_t activeInId = compiled.getLogicalId(inId);

                TensorView safeView = inNode.view;
                if (safeView.shape.empty() && !inNode.shape.empty())
                {
                    safeView.shape = inNode.shape;
                    safeView.strides = TensorView::calcContiguousStrides(inNode.shape);
                    safeView.dtype = inNode.dtype;
                    safeView.baseOffset = 0;
                }
                safeView.baseOffset += memManager.buffers.at(inNode.backend).getOffset(activeInId);

                resolvedInputs.push_back({memManager.buffers.at(inNode.backend).arena_ptr + safeView.baseOffset, safeView, activeInId});
            }

            const bool isEndOfLogicalChain = (idx + 1 == compiled.instructions.size()) ||
                                             (compiled.instructions[idx + 1].logicalNodeId != logicalId);
            const uint32_t outputMemId = (logicalId != UINT32_MAX && (logicalId == nodeId || isEndOfLogicalChain))
                                             ? logicalId
                                             : nodeId;

            uint32_t memId = outputMemId;
            if (inst.inplaceInputIndex >= 0)
            {
                uint32_t srcId = inst.inputNodeIds[inst.inplaceInputIndex];
                uint32_t activeSrcId = srcId;
                uint32_t srcLogicalId = compiled.getLogicalId(srcId);
                if (!memManager.has(inst.backend, srcId) && memManager.has(inst.backend, srcLogicalId))
                {
                    activeSrcId = srcLogicalId;
                }
                memManager.transferOwnership(inst.backend, activeSrcId, outputMemId);
            }
            else
            {
                uint64_t sizeBytes = getSizeBytes(node.shape, node.dtype);
                float cost = compiled.nodeCosts.at(inst.nodeId);
                memManager.allocate(inst.backend, outputMemId, sizeBytes, inst.outputStorageType, compiled.refCounts.at(inst.nodeId), cost, &parentMap, &compiled.nodeCosts);
            }

            auto outBlockIt = outBuf.allocationMap.at(memId);
            outBlockIt->refCount = compiled.refCounts.at(inst.nodeId);
            outBlockIt->isLocked = true;

            std::vector<Region> computeRegions;
            std::vector<uint64_t> computeKernels;

            if (!inst.outputRegions.empty())
            {
                computeRegions = inst.outputRegions;
                computeKernels = inst.cachedKernelIds;
                nPartial++;
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

            const std::vector<std::vector<Region>> *instructionSlices = !inst.inputSlices.empty() ? &inst.inputSlices : nullptr;

            for (size_t rIdx = 0; rIdx < computeRegions.size(); ++rIdx)
            {
                const Region &outRegion = computeRegions[rIdx];
                if (computeKernels[rIdx] == 0)
                {
                    std::cout << "\n[Executor Error] Found kernel UID 0 for node " << inst.nodeId
                              << " at region index " << rIdx << "\n"
                              << toString(node) << std::endl;
                    std::cout << "Cached kernel IDs size: " << inst.cachedKernelIds.size() << std::endl;
                    for (size_t i = 0; i < inst.cachedKernelIds.size(); ++i)
                    {
                        std::cout << "  [" << i << "]: " << inst.cachedKernelIds[i] << std::endl;
                    }
                }
                const KernelEntry &kernel = KernelRegistry::get().getKernel(computeKernels[rIdx]);

                const bool fullRegion = isFullRegion(outRegion, node.shape);

                std::vector<const void *> kernelInputs;
                std::vector<TensorView> kernelInViews;

                for (size_t pIdx = 0; pIdx < inst.inputNodeIds.size(); ++pIdx)
                {
                    uint32_t inId = inst.inputNodeIds[pIdx];
                    const TensorNode &inNode = compiled.nodesMap.at(inId);
                    auto &inBuf = memManager.buffers.at(inNode.backend);

                    TensorView inView = resolvedInputs[pIdx].view;

                    if (!fullRegion && instructionSlices && pIdx < instructionSlices->size() && rIdx < (*instructionSlices)[pIdx].size() && !(*instructionSlices)[pIdx][rIdx].empty())
                    {
                        const Region &inputSlice = (*instructionSlices)[pIdx][rIdx];

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
                        kernelInputs.push_back(resolvedInputs[pIdx].ptr);
                        kernelInViews.push_back(inView);
                    }
                }

                std::vector<void *> kernelOutputs;
                std::vector<TensorView> kernelOutViews;

                TensorView outView;
                {
                    uint64_t arenaOffset = outBuf.getOffset(outputMemId);
                    if (node.view.shape.empty())
                    {
                        outView.baseOffset = arenaOffset;
                        outView.shape = node.shape;
                        outView.strides = TensorView::calcContiguousStrides(node.shape);
                        outView.dtype = node.dtype;
                    }
                    else
                    {
                        outView = node.view;
                        outView.baseOffset += arenaOffset;
                    }
                }

                if (!fullRegion)
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

                for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
                {
                    const uint32_t inId = inst.inputNodeIds[i];
                    TensorNode debugInput = compiled.nodesMap.at(inId);
                    debugInput.id = resolvedInputs[i].allocationId;
                    Debug::checkNan(debugInput, memManager, "Kernel Input: " + std::to_string(inId));
                }
                kernel.run(kernelInputs, kernelOutputs, kernelInViews, kernelOutViews);
                TensorNode debugOutput = compiled.nodesMap.at(inst.nodeId);
                debugOutput.id = outputMemId;
                Debug::checkNan(debugOutput, memManager, "Kernel Output: " + std::to_string(inst.nodeId));
            }

            for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
            {
                if (static_cast<int>(i) == inst.inplaceInputIndex)
                    continue;

                uint32_t inId = inst.inputNodeIds[i];
                uint32_t activeInId = inId;
                uint32_t inLogicalId = compiled.getLogicalId(inId);
                if (!memManager.has(compiled.nodesMap.at(inId).backend, inId) && memManager.has(compiled.nodesMap.at(inId).backend, inLogicalId))
                {
                    activeInId = inLogicalId;
                }
                memManager.release(compiled.nodesMap.at(inId).backend, activeInId);
            }

            if (outputMemId == inst.nodeId && inst.logicalNodeId != UINT32_MAX && inst.logicalNodeId != inst.nodeId)
            {
                if (isEndOfLogicalChain && memManager.has(inst.backend, inst.nodeId))
                {
                    memManager.transferOwnership(inst.backend, inst.nodeId, inst.logicalNodeId);
                }
            }

            std::cout << instIdx << "/" << compiled.instructions.size() << ", #part: " << nPartial << "\r" << std::flush;
        }
    }
};
