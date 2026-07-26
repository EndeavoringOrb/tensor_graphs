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

            for (uint64_t i = 0; i < inst.children.size(); ++i)
            {
                const TensorView &inView = compiled.nodeViews.at(inst.children[i]);
                const ParallelBuffer &inBuf = inst.inBuffers[i];
                DeviceBuffer *inBufObj = memManager.getBuffer(inBuf.mem_space);
                if (!inBufObj)
                    Error::throw_err("Input DeviceBuffer not found");

                LogicalId logical_id;
                if (compiled.has_logical_id(inst.children[i]))
                {
                    logical_id = compiled.get_logical_id(inst.children[i]);
                }
                inBufObj->setupInput(ctx, inView, logical_id);
            }

            const TensorView &outView = compiled.nodeViews.at(inst.eclass_id);
            DeviceBuffer *outBufObj = memManager.getBuffer(inst.outBuffer.mem_space);
            if (!outBufObj)
                Error::throw_err("Output DeviceBuffer not found");

            LogicalId logical_id;
            if (compiled.has_logical_id(inst.eclass_id))
            {
                logical_id = compiled.get_logical_id(inst.eclass_id);
            }
            outBufObj->setupOutput(ctx, outView, logical_id);

            const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.kernel_id);
            std::string opName = kernel.opName.empty() ? toString(kernel.opType) : kernel.opName;

            if (OpenCLState::get().initialized)
            {
                clFinish(OpenCLState::get().queue);
            }
#ifdef USE_CUDA
            cudaDeviceSynchronize();
#endif
            Debug::checkValues(ctx.inputs, ctx.inViews, "(inputs) inst # " + std::to_string(idx) + toString(inst) + "\n" + toString(kernel));

            if (!kernel.is_view && kernel.run)
            {
                kernel.run(ctx);
            }

            if (OpenCLState::get().initialized)
            {
                clFinish(OpenCLState::get().queue);
            }
#ifdef USE_CUDA
            cudaDeviceSynchronize();
#endif
            Debug::checkValues(ctx.outputs, ctx.outViews, "(output) inst # " + std::to_string(idx) + toString(inst) + "\n" + toString(kernel));

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
                debugCallback(logical_id, ctx, ctx.outputs[0]);
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