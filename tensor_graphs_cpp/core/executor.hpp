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
        for (uint64_t idx = 0; idx < nInst; ++idx)
        {
            const OpInstruction &inst = compiled.instructions[idx];

            if (InterruptManager::isInterrupted())
            {
                std::cerr << "\n[Executor] Interrupt detected, aborting execution..." << std::endl;
                InterruptManager::cleanup();
                std::exit(SIGINT);
            }

            KernelContext ctx;

            for (uint64_t i = 0; i < inst.inputNodeIds.size(); ++i)
            {
                const TensorView &inView = compiled.nodeViews.at(inst.inputNodeIds[i]);
                const ParallelBuffer &inBuf = inst.inBuffers[i];
                DeviceBuffer *inBufObj = memManager.getBuffer(inBuf.mem_space);
                if (!inBufObj)
                    Error::throw_err("Input DeviceBuffer not found");

                inBufObj->setupInput(ctx, inView, compiled.getLogicalId(inst.inputNodeIds[i]));
            }

            const TensorView &outView = compiled.nodeViews.at(inst.nodeId);
            DeviceBuffer *outBufObj = memManager.getBuffer(inst.outBuffer.mem_space);
            if (!outBufObj)
                Error::throw_err("Output DeviceBuffer not found");

            outBufObj->setupOutput(ctx, outView, compiled.getLogicalId(inst.nodeId));

            const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.fullKernelId);

            if (!kernel.is_view && kernel.run)
            {
                kernel.run(ctx);
            }

            if (debugCallback)
            {
                if (outBufObj->mem_space.type == HandleType::OPENCL)
                {
                    clFinish(OpenCLState::get().queue);
                }
#ifdef USE_CUDA
                else if (outBufObj->mem_space.type == HandleType::CUDA)
                {
                    cudaDeviceSynchronize();
                }
#endif
                debugCallback(compiled.getLogicalId(inst.nodeId), ctx, ctx.outputs[0]);
            }

            // Cleanup Context
            for (const ParallelBuffer &inBuf : inst.inBuffers)
            {
                memManager.getBuffer(inBuf.mem_space)->cleanupContext(ctx);
            }
            outBufObj->cleanupContext(ctx);

            timer.tick();
        }
    }
};