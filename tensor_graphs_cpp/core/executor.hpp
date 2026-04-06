// tensor_graphs_cpp/core/executor.hpp
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

            const uint32_t nodeId = inst.nodeId;
            uint32_t logicalId = compiled.getLogicalId(nodeId);
            const TensorNode &node = compiled.nodesMap.at(nodeId);
            auto &outBuf = memManager.buffers.at(node.backend);

            const bool isEndOfLogicalChain = (idx + 1 == compiled.instructions.size()) ||
                                             (compiled.instructions[idx + 1].logicalNodeId != logicalId);
            const uint32_t outputMemId = (logicalId != UINT32_MAX && (logicalId == nodeId || isEndOfLogicalChain))
                                             ? logicalId
                                             : nodeId;

            if (inst.inplaceInputIndex < 0 && inst.viewInputIndex < 0)
            {
                uint64_t sizeBytes = getSizeBytes(node.getShape(), node.dtype);
                float cost = compiled.nodeCosts.at(inst.nodeId);
                memManager.allocate(inst.backend, outputMemId, sizeBytes, inst.outputStorageType, compiled.refCounts.at(inst.nodeId), cost, &parentMap, &compiled.nodeCosts);
            }

            std::vector<const void *> kernelInputs;
            std::vector<TensorView> kernelInViews;
            for (uint32_t inId : inst.inputNodeIds)
            {
                const TensorNode &inNode = compiled.nodesMap.at(inId);
                TensorView view = memManager.getView(inNode, compiled);
                if (memManager.has(inNode.backend, inId))
                {
                    view = memManager.getView(inNode);
                }

                kernelInViews.push_back(view);
                kernelInputs.push_back(memManager.buffers.at(inNode.backend).arena_ptr + view.baseOffset);
            }

            for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
            {
                const uint32_t inId = inst.inputNodeIds[i];
                TensorNode debugInput = compiled.nodesMap.at(inId);
                debugInput.id = compiled.getLogicalId(inId);
                Debug::checkNan(debugInput, memManager, "Kernel Input: " + std::to_string(inId), compiled);
            }

            if (inst.inplaceInputIndex >= 0)
            {
                uint32_t srcId = compiled.getLogicalId(inst.inputNodeIds[inst.inplaceInputIndex]);
                memManager.transferOwnership(inst.backend, srcId, outputMemId);
            }
            else if (inst.viewInputIndex >= 0)
            {
                uint32_t srcId = compiled.getLogicalId(inst.inputNodeIds[inst.viewInputIndex]);
                memManager.addAlias(inst.backend, srcId, outputMemId, compiled.refCounts.at(inst.nodeId), inst.outputStorageType);
            }

            auto it = memManager.buffers.find(node.backend);
            if (it == memManager.buffers.end())
            {
                Error::throw_err("[Executor.run] Backend buffer not initialized for " + toString(node.backend));
            }
            uint32_t targetId = memManager.resolveAlias(outputMemId);
            uint64_t arenaOffset = it->second.getOffset(targetId);
            TensorView outView = TensorView(node, arenaOffset + node.viewOffset * getDTypeSize(node.dtype));
            std::vector<TensorView> kernelOutViews = {outView};
            std::vector<void *> kernelOutputs = {outBuf.arena_ptr + outView.baseOffset};

            if (inst.viewInputIndex < 0)
            {
                MemBlock &outBlock = memManager.getBlock(node.backend, outputMemId, compiled);
                outBlock.refCount = compiled.refCounts.at(inst.nodeId);
                outBlock.storageType = inst.outputStorageType;
                outBlock.isLocked = true;
            }

            const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.fullKernelId);

            if (!kernel.isView)
            {
                kernel.run(kernelInputs, kernelOutputs, kernelInViews, kernelOutViews);
            }

            TensorNode debugOutput = compiled.nodesMap.at(inst.nodeId);
            debugOutput.id = outputMemId;
            Debug::checkNan(debugOutput, memManager, "Kernel Output: " + std::to_string(inst.nodeId), compiled);

            for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
            {
                if (static_cast<int>(i) == inst.inplaceInputIndex)
                    continue;
                if (static_cast<int>(i) == inst.viewInputIndex)
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

            instIdx++;
            std::cout << instIdx << "/" << compiled.instructions.size() << "\r" << std::flush;
        }
    }
};