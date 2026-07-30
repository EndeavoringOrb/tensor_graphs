// File: tensor_graphs_cpp/core/executor.hpp
#pragma once
#include "core/debug.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/memory.hpp"
#include "core/plan/planner.hpp"
#include "core/types.hpp"

class Executor
{
private:
    MemoryManager &memManager;

public:
    Executor(MemoryManager &mm) : memManager(mm)
    {
    }

    void run(const CompiledGraph &compiled, const Debug::Callback &debugCallback = nullptr)
    {
        uint32_t nInst = compiled.instructions.size();
        bool disableTimer = true;
#ifdef DEBUG
        disableTimer = false;
#endif
        ProgressTimer timer(nInst, "running ", disableTimer);

        bool cuda_busy = false;
        bool opencl_busy = false;
        std::unordered_map<uint32_t, EngineType> buffer_last_writer;

        for (uint64_t idx = 0; idx < nInst; ++idx)
        {
            const OpInstruction &inst = compiled.instructions[idx];

            if (InterruptManager::isInterrupted())
            {
                std::cerr << "\n[Executor] Interrupt detected, aborting execution..." << std::endl;
                InterruptManager::cleanup();
                std::exit(SIGINT);
            }

            const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.kernel_id);
            std::string kernel_name = kernel.opName.empty() ? toString(kernel.opType) : kernel.opName;
            EngineType current_engine = kernel.engines.empty() ? EngineType::CPU : kernel.engines[0].type;

            bool sync_cuda = false;
            bool sync_opencl = false;

            // Check if any input was last written by a different asynchronous engine
            for (const ParallelBuffer &inBuf : inst.inBuffers)
            {
                auto it = buffer_last_writer.find(inBuf.id.value);
                if (it != buffer_last_writer.end())
                {
                    EngineType writer_engine = it->second;
                    if (writer_engine != current_engine)
                    {
                        if (writer_engine == EngineType::CUDA_GPU && cuda_busy)
                            sync_cuda = true;
                        if (writer_engine == EngineType::QUALCOMM_IGPU && opencl_busy)
                            sync_opencl = true;
                    }
                }
            }

            if (sync_cuda)
            {
#ifdef USE_CUDA
                cudaDeviceSynchronize();
#endif
                cuda_busy = false;
            }
            if (sync_opencl)
            {
                if (OpenCLState::get().initialized)
                {
                    clFinish(OpenCLState::get().queue);
                }
                opencl_busy = false;
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

#ifdef DEBUG
            Debug::checkValues(ctx.inputs, ctx.inViews,
                               "(inputs) inst # " + std::to_string(idx) + " " + toString(inst) + "\n" +
                                   toString(kernel));
#endif

            if (!kernel.is_view && kernel.run)
            {
                kernel.run(ctx);
                if (current_engine == EngineType::CUDA_GPU)
                    cuda_busy = true;
                if (current_engine == EngineType::QUALCOMM_IGPU)
                    opencl_busy = true;
            }

            // Track who wrote to this memory space
            buffer_last_writer[inst.outBuffer.id.value] = current_engine;

#ifdef DEBUG
            if (cuda_busy)
            {
#ifdef USE_CUDA
                cudaDeviceSynchronize();
#endif
                cuda_busy = false;
            }
            if (opencl_busy)
            {
                if (OpenCLState::get().initialized)
                {
                    clFinish(OpenCLState::get().queue);
                }
                opencl_busy = false;
            }
            std::vector<const void *> c_outputs(ctx.outputs.begin(), ctx.outputs.end());
            Debug::checkValues(c_outputs, ctx.outViews, ctx.inputs, ctx.inViews, kernel,
                               "(output) inst # " + std::to_string(idx) + " " + toString(inst) + "\n" +
                                   toString(kernel));
#endif

            if (debugCallback)
            {
                if (outBufObj->mem_space.type == HandleType::CPP)
                {
                    debugCallback(logical_id, kernel_name, ctx, ctx.outputs[0]);
                }
            }

            // Cleanup Context
            for (const ParallelBuffer &inBuf : inst.inBuffers)
            {
                memManager.getBuffer(inBuf.mem_space)->cleanupContext(ctx);
            }
            outBufObj->cleanupContext(ctx);

            timer.tick();
        }

        // Final synchronization to ensure all pending work is completed before returning to Python/User
        if (cuda_busy)
        {
#ifdef USE_CUDA
            cudaDeviceSynchronize();
#endif
        }
        if (opencl_busy)
        {
            if (OpenCLState::get().initialized)
            {
                clFinish(OpenCLState::get().queue);
            }
        }
    }
};