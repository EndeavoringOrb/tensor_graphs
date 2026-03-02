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

    void run(const std::unordered_map<uint32_t, const void *> &inputs)
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

            // Re-allocate / ensure locked state for this execution iteration
            uint64_t offset = memManager.allocate(node.backend, nodeId, sizeBytes, StorageType::TRANSIENT, refs, 0.0f);

            auto &buf = memManager.buffers.at(node.backend);
            std::memcpy(buf.arena.data() + offset, dataPtr, sizeBytes);
        }

        // 2. Execute Compiled Instructions
        for (const OpInstruction &inst : compiled.instructions)
        {
            const TensorNode &node = graph.nodes[inst.nodeId];
            const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.kernelId);

            // 2.a Fetch Inputs
            std::vector<const void *> kernelInputs;
            std::vector<TensorView> kernelInViews;
            for (uint32_t inId : inst.inputNodeIds)
            {
                const TensorNode &inNode = graph.nodes[inId];
                auto &inBuf = memManager.buffers.at(inNode.backend);
                auto it = inBuf.allocationMap.find(inId);

                if (it == inBuf.allocationMap.end())
                {
                    throw std::runtime_error("Input node not found in memory allocation map");
                }

                uint64_t inOffset = it->second->offset;
                TensorView inView = getEffectiveView(inNode);

                kernelInputs.push_back(inBuf.arena.data() + inOffset + inView.baseOffset);
                kernelInViews.push_back(inView);
            }

            // 2.b Allocate / Transfer memory for the Output
            if (inst.inplaceInputIndex >= 0)
            {
                uint32_t srcId = inst.inputNodeIds[inst.inplaceInputIndex];
                memManager.transferOwnership(inst.backend, srcId, inst.nodeId);

                auto &buf = memManager.buffers.at(inst.backend);
                auto blockIt = buf.allocationMap.at(inst.nodeId);
                blockIt->refCount = compiled.refCounts[inst.nodeId];
                blockIt->isLocked = true;
            }
            else
            {
                uint64_t sizeBytes = getSizeBytes(node.shape, node.dtype);
                uint32_t refs = compiled.refCounts[inst.nodeId];
                memManager.allocate(inst.backend, inst.nodeId, sizeBytes, StorageType::TRANSIENT, refs, 0.0f);
            }

            // 2.c Fetch Output Memory Pointers
            std::vector<void *> kernelOutputs;
            std::vector<TensorView> kernelOutViews;

            auto &outBuf = memManager.buffers.at(node.backend);
            uint64_t outOffset = outBuf.allocationMap.at(node.id)->offset;
            TensorView outView = getEffectiveView(node);

            kernelOutputs.push_back(outBuf.arena.data() + outOffset + outView.baseOffset);
            kernelOutViews.push_back(outView);

            // 2.d Run the Operator Kernel
            kernel.run(kernelInputs, kernelOutputs, kernelInViews, kernelOutViews);

            // 2.e Release Consumed Parents
            for (size_t i = 0; i < inst.inputNodeIds.size(); ++i)
            {
                // Inplace parents were already logically consumed by ownership transfer
                if (static_cast<int>(i) == inst.inplaceInputIndex)
                    continue;

                uint32_t inId = inst.inputNodeIds[i];
                memManager.release(graph.nodes[inId].backend, inId);
            }
        }
    }
};