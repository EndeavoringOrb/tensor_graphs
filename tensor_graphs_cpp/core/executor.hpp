#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/plan/planner.hpp"
#include "core/memory.hpp"
#include "core/kernels.hpp"
#include "core/debug.hpp"

class Executor
{
private:
    MemoryManager &memManager;

public:
    Executor(MemoryManager &mm) : memManager(mm) {}

    void run(const CompiledGraph &compiled, const Debug::Callback &debugCallback = nullptr)
    {
        uint32_t nInst = compiled.instructions.size();
        bool disableTimer = true;
#ifdef DEBUG
        disableTimer = false;
#endif
        ProgressTimer timer(nInst, "running ", disableTimer);
        for (size_t idx = 0; idx < nInst; ++idx)
        {
            const OpInstruction &inst = compiled.instructions[idx];

            if (InterruptManager::isInterrupted()) {
                std::cerr << "\n[Executor] Interrupt detected, aborting execution..." << std::endl;
                InterruptManager::cleanup();
                std::exit(SIGINT);
            }

            KernelContext ctx;

            for (size_t i = 0; i < inst.inputNodeIds.size(); ++i) {
                const TensorNode &inNode = compiled.nodesMap.at(inst.inputNodeIds[i]);
                const ParallelBuffer &inBuf = inst.inBuffers[i];
                DeviceBuffer* inBufObj = memManager.getBuffer(inBuf.mem_space);
                if (!inBufObj) Error::throw_err("Input DeviceBuffer not found");
                
                TensorView view(inNode, 0); 
                view.baseOffset = inNode.viewOffset * getDTypeSize(inNode.dtype); 
                inBufObj->setupInput(ctx, inBuf.offset, view, compiled.getLogicalId(inst.inputNodeIds[i]));
            }

            const TensorNode &outNode = compiled.nodesMap.at(inst.nodeId);
            DeviceBuffer* outBufObj = memManager.getBuffer(inst.outBuffer.mem_space);
            if (!outBufObj) Error::throw_err("Output DeviceBuffer not found");

            TensorView outView(outNode, 0);
            outView.baseOffset = outNode.viewOffset * getDTypeSize(outNode.dtype);
            outBufObj->setupOutput(ctx, inst.outBuffer.offset, outView, compiled.getLogicalId(inst.nodeId));

            const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.fullKernelId);

            if (!kernel.is_view && kernel.run) {
                kernel.run(ctx);
            }

            if (debugCallback) {
                if (outBufObj->mem_space.type == HandleType::OPENCL) {
                    clFinish(OpenCLState::get().queue);
                }
#ifdef USE_CUDA
                else if (outBufObj->mem_space.type == HandleType::CUDA) {
                    cudaDeviceSynchronize();
                }
#endif
                debugCallback(compiled.getLogicalId(inst.nodeId), outNode, ctx, ctx.outputs[0]);
            }

            // Cleanup Context
            for (const ParallelBuffer &inBuf : inst.inBuffers) {
                memManager.getBuffer(inBuf.mem_space)->cleanupContext(ctx);
            }
            outBufObj->cleanupContext(ctx);

            timer.tick();
        }
    }
};