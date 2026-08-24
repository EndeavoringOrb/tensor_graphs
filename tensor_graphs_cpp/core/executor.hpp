#pragma once
#include "core/debug.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/memory.hpp"
#include "core/plan/planner.hpp"
#include "core/synchronizer.hpp"
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
#ifdef TG_DEBUG
        disableTimer = false;
#endif
        ProgressTimer timer(nInst, "running", disableTimer);

        for (const auto &pair : compiled.constantStaging)
        {
            EClassId eclass_id = pair.first;
            if (compiled.nodeViews.count(eclass_id))
            {
                const TensorView &view = compiled.nodeViews.at(eclass_id);
                memManager.write(MemSpace{1, HandleType::CPP}, view.offset, pair.second->data(), pair.second->size());
            }
        }

        Synchronizer sync;

        for (uint64_t idx = 0; idx < nInst; ++idx)
        {
            const OpInstruction &inst = compiled.instructions[idx];
            const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.kernel_id);
            std::string kernel_name = kernel.opName.empty() ? toString(kernel.opType) : kernel.opName;

            const std::vector<Engine> &inst_engines = inst.engines;
            const Engine &primary_engine = inst_engines[0];

            sync.syncBefore(inst, inst_engines);

            KernelContext ctx;

#ifdef TG_USE_CUDA
            int primary_cuda_device = -1;

            for (const Engine &eng : inst_engines)
            {
                if (eng.type == EngineType::CUDA_GPU || eng.type == EngineType::CUDA_DMA)
                {
                    // The first CUDA engine found becomes our target device for kernel execution
                    if (primary_cuda_device == -1)
                    {
                        primary_cuda_device = static_cast<int>(eng.idx);
                    }
                    ctx.cuda_streams.push_back(reinterpret_cast<void *>(sync.getCudaStream(eng)));
                }
            }

            // Set the device context exactly once if any CUDA engine is involved
            if (primary_cuda_device != -1)
            {
                cudaSetDevice(primary_cuda_device);
            }
#endif

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

            DeviceBuffer *outBufObj = memManager.getBuffer(inst.outBuffer.mem_space);
            if (!outBufObj)
                Error::throw_err("Output DeviceBuffer not found");

            LogicalId logical_id;
            if (compiled.has_logical_id(inst.eclass_id))
            {
                logical_id = compiled.get_logical_id(inst.eclass_id);
            }
            const TensorView &outView = compiled.nodeViews.at(inst.eclass_id);
            outBufObj->setupOutput(ctx, outView, logical_id);

            bool issued_work = false;
            if (!kernel.is_view && kernel.run)
            {
                kernel.run(ctx);
                issued_work = true;
            }

            sync.markExecuted(inst, inst_engines, issued_work);

            if (debugCallback)
            {
                sync.syncEngines(inst_engines);
                if (outBufObj->mem_space.type == HandleType::CPP)
                {
                    debugCallback(logical_id, kernel_name, ctx, ctx.outputs[0]);
                }
#ifdef TG_USE_CUDA
                else if (outBufObj->mem_space.type == HandleType::CUDA)
                {
                    std::vector<uint8_t> host_copy(countElements(outView) * getDTypeSize(outView.dtype));
                    cudaSetDevice(outBufObj->mem_space.idx);
                    cudaMemcpy(host_copy.data(), ctx.outputs[0], host_copy.size(), cudaMemcpyDeviceToHost);
                    debugCallback(logical_id, kernel_name, ctx, host_copy.data());
                }
#endif
            }

            for (const ParallelBuffer &inBuf : inst.inBuffers)
            {
                memManager.getBuffer(inBuf.mem_space)->cleanupContext(ctx);
            }
            outBufObj->cleanupContext(ctx);

            timer.tick();
        }

        sync.syncAll();
    }
};