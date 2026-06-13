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
#include <functional>

class Executor
{
private:
    MemoryManager &memManager;
    std::unordered_map<uint32_t, float> nodeCosts;

public:
    using DebugCallback = std::function<void(uint32_t logicalId, const TensorView &view, const void *data)>;

    Executor(MemoryManager &mm)
        : memManager(mm) {}

    void run(const CompiledGraph &compiled,
             const DebugCallback &debugCallback = nullptr)
    {
        double totalKernelTime = 0.0f;

        uint32_t instIdx = 0;
        uint32_t nInst = compiled.instructions.size();
        bool disableTimer = true;
#ifdef DEBUG
        disableTimer = false;
#endif
        ProgressTimer timer(nInst, "running ", disableTimer);
        for (size_t idx = 0; idx < nInst; ++idx)
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
            if (node.backend == Backend::STORAGE)
            {
                Error::throw_err("[Executor.run] should not be executing anything on Backend::STORAGE");
            }

            const bool isEndOfLogicalChain = (idx + 1 == nInst) ||
                                             (compiled.instructions[idx + 1].logicalNodeId != logicalId);
            const uint32_t outputMemId = (logicalId != UINT32_MAX && (logicalId == nodeId || isEndOfLogicalChain))
                                             ? logicalId
                                             : nodeId;

            if (inst.inplaceInputIndex < 0 && inst.viewInputIndex < 0)
            {
                uint64_t sizeBytes = getSizeBytes(node.getShape(), node.dtype);
                float cost = compiled.nodeCosts.at(inst.nodeId);
                memManager.allocate(inst.backend, outputMemId, sizeBytes, inst.outputStorageType, compiled.refCounts.at(inst.nodeId), cost);
            }

            KernelContext ctx;
            for (uint32_t inId : inst.inputNodeIds)
            {
                const TensorNode &inNode = compiled.nodesMap.at(inId);

                if (inNode.backend == Backend::STORAGE)
                {
                    uint32_t logicalInId = compiled.getLogicalId(inId);
                    TensorMetadata meta = FileRegistry::get().getNodeMeta(logicalInId);
                    TensorView view;
                    view.baseOffset = meta.dataOffsetStart;
                    view.dtype = meta.dtype;
                    view.setShape(meta.shape);
                    ctx.inViews.push_back(view);
                    ctx.inputs.push_back(nullptr);
                    ctx.fd.push_back(FileRegistry::get().getNodeFd(logicalInId));
                }
                else
                {
                    uint32_t activeInId = inId;
                    uint32_t inLogicalId = compiled.getLogicalId(inId);
                    if (!memManager.has(inNode.backend, inId) && memManager.has(inNode.backend, inLogicalId))
                    {
                        activeInId = inLogicalId;
                    }

                    TensorView view = memManager.getView(inNode, activeInId);
                    ctx.inViews.push_back(view);
                    ctx.inputs.push_back(memManager.buffers.at(inNode.backend).arena_ptr + view.baseOffset);
                    ctx.fd.push_back(-1);
                }
            }

#ifdef DEBUG_CHECKNAN
            for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
            {
                const uint32_t inId = inst.inputNodeIds[i];
                const TensorNode &inNode = compiled.nodesMap.at(inId);
                if (inNode.backend != Backend::STORAGE)
                {
                    uint32_t activeInId = inId;
                    uint32_t inLogicalId = compiled.getLogicalId(inId);
                    if (!memManager.has(inNode.backend, inId) && memManager.has(inNode.backend, inLogicalId))
                    {
                        activeInId = inLogicalId;
                    }
                    TensorNode debugInput = inNode;
                    debugInput.id = activeInId;
                    Debug::checkNan(debugInput, memManager, "Kernel Input: " + std::to_string(inId));
                }
            }
#endif

            if (inst.inplaceInputIndex >= 0)
            {
                uint32_t inId = inst.inputNodeIds[inst.inplaceInputIndex];
                const TensorNode &inNode = compiled.nodesMap.at(inId);
                uint32_t srcId = inId;
                uint32_t inLogicalId = compiled.getLogicalId(inId);
                if (!memManager.has(inNode.backend, inId) && memManager.has(inNode.backend, inLogicalId))
                {
                    srcId = inLogicalId;
                }
                memManager.transferOwnership(inNode.backend, srcId, outputMemId);
            }
            else if (inst.viewInputIndex >= 0)
            {
                uint32_t inId = inst.inputNodeIds[inst.viewInputIndex];
                const TensorNode &inNode = compiled.nodesMap.at(inId);
                uint32_t srcId = inId;
                uint32_t inLogicalId = compiled.getLogicalId(inId);
                if (!memManager.has(inNode.backend, inId) && memManager.has(inNode.backend, inLogicalId))
                {
                    srcId = inLogicalId;
                }
                memManager.addAlias(inNode.backend, srcId, outputMemId, compiled.refCounts.at(inst.nodeId), inst.outputStorageType);
            }

            auto it = memManager.buffers.find(node.backend);
            if (it == memManager.buffers.end())
            {
                Error::throw_err("[Executor.run] Backend buffer not initialized for " + toString(node.backend));
            }
            uint32_t targetId = memManager.resolveAlias(outputMemId);
            Backend actualBackend = node.backend;
            if (memManager.buffers.count(actualBackend) == 0 ||
                memManager.buffers.at(actualBackend).allocationMap.count(targetId) == 0)
            {
                bool found = false;
                for (auto const &pair : memManager.buffers)
                {
                    if (pair.second.allocationMap.count(targetId))
                    {
                        actualBackend = pair.first;
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    Error::throw_err("[Executor.run] Physical allocation " + std::to_string(targetId) + " not found in any backend buffer.");
                }
            }

            auto &actualBuf = memManager.buffers.at(actualBackend);
            uint64_t arenaOffset = actualBuf.getOffset(targetId);

            // Create views and pointers relative to the actual physical buffer
            ctx.outViews = {TensorView(node, arenaOffset + node.viewOffset * getDTypeSize(node.dtype))};
            ctx.outputs = {actualBuf.arena_ptr + ctx.outViews[0].baseOffset};

            if (inst.viewInputIndex < 0)
            {
                if (memManager.aliasMap.find(outputMemId) != memManager.aliasMap.end())
                {
                    memManager.aliasRefCounts[outputMemId] = compiled.refCounts.at(inst.nodeId);
                    memManager.aliasStorageTypes[outputMemId] = inst.outputStorageType;
                }
                else
                {
                    MemBlock &outBlock = memManager.getBlock(node.backend, outputMemId);
                    outBlock.refCount = compiled.refCounts.at(inst.nodeId);
                    outBlock.storageType = inst.outputStorageType;
                    outBlock.isLocked = true;
                }
            }

            const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.fullKernelId);

            if (!kernel.isView)
            {
                // ProgressTimer kernelTimer(0, "", true);
                kernel.run(ctx);
                // totalKernelTime += kernelTimer.getElapsed();
            }

            if (debugCallback && logicalId != UINT32_MAX && isEndOfLogicalChain)
            {
                const uint8_t *basePtr = actualBuf.arena_ptr + ctx.outViews[0].baseOffset;
                uint64_t maxOffset = 0;
                for (size_t d = 0; d < ctx.outViews[0].getShape().size(); ++d)
                {
                    if (ctx.outViews[0].getShape()[d] > 0)
                    {
                        maxOffset += (ctx.outViews[0].getShape()[d] - 1) * ctx.outViews[0].strides[d];
                    }
                }
                uint64_t bytesToCopy = (ctx.outViews[0].getShape().empty() ? 1 : (maxOffset + 1)) * getDTypeSize(ctx.outViews[0].dtype);

#ifdef USE_CUDA
                if (actualBackend == Backend::CUDA)
                {
                    cudaDeviceSynchronize();
                    std::vector<uint8_t> hostData(bytesToCopy);
                    cudaMemcpy(hostData.data(), basePtr, bytesToCopy, cudaMemcpyDeviceToHost);
                    debugCallback(logicalId, ctx.outViews[0], hostData.data());
                }
                else
#endif
                {
                    debugCallback(logicalId, ctx.outViews[0], basePtr);
                }
            }

            TensorNode debugOutput = compiled.nodesMap.at(inst.nodeId);
            debugOutput.id = outputMemId;
            Debug::checkNan(debugOutput, memManager, "Kernel Output: " + std::to_string(inst.nodeId));

            for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
            {
                if (static_cast<int>(i) == inst.inplaceInputIndex)
                    continue;
                if (static_cast<int>(i) == inst.viewInputIndex)
                    continue;

                uint32_t inId = inst.inputNodeIds[i];
                const TensorNode &inNode = compiled.nodesMap.at(inId);
                if (inNode.backend == Backend::STORAGE)
                    continue;
                uint32_t activeInId = inId;
                uint32_t inLogicalId = compiled.getLogicalId(inId);
                if (!memManager.has(inNode.backend, inId) && memManager.has(inNode.backend, inLogicalId))
                {
                    activeInId = inLogicalId;
                }
                memManager.release(inNode.backend, activeInId);
            }

            if (outputMemId == inst.nodeId && inst.logicalNodeId != UINT32_MAX && inst.logicalNodeId != inst.nodeId)
            {
                if (isEndOfLogicalChain && memManager.has(inst.backend, inst.nodeId))
                {
                    memManager.transferOwnership(inst.backend, inst.nodeId, inst.logicalNodeId);
                }
            }

            instIdx++;
            timer.tick();
        }
        // std::cout << "\nTotal Kernel Time: " << std::to_string(totalKernelTime * 1000) << "ms" << std::endl;
    }
};