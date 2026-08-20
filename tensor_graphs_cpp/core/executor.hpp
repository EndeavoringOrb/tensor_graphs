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
#ifdef DEBUG
        disableTimer = false;
#endif
        ProgressTimer timer(nInst, "running", disableTimer);

        std::unordered_set<EClassId> restored_constants;
        for (const auto &inst : compiled.instructions)
        {
            for (size_t i = 0; i < inst.children.size(); ++i)
            {
                EClassId child = inst.children[i];
                if (!compiled.has_logical_id(child) && compiled.constantStaging.count(child))
                {
                    if (restored_constants.insert(child).second)
                    {
                        const ParallelBuffer &buf = inst.inBuffers[i];
                        memManager.write(buf.mem_space, buf.offset, compiled.constantStaging.at(child)->data(),
                                         compiled.constantStaging.at(child)->size());
                    }
                }
            }
        }

        Synchronizer sync;

        for (uint64_t idx = 0; idx < nInst; ++idx)
        {
            const OpInstruction &inst = compiled.instructions[idx];
            const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.kernel_id);
            std::string kernel_name = kernel.opName.empty() ? toString(kernel.opType) : kernel.opName;

            // Dynamically resolve actual engines mapped for this instruction
            std::vector<TensorNode> dummyInputs(inst.children.size());
            std::vector<MemSpace> in_mem_spaces(inst.children.size());
            for (size_t i = 0; i < inst.children.size(); ++i)
            {
                const auto &view = compiled.nodeViews.at(inst.children[i]);
                dummyInputs[i].setShape(view.getShape());
                dummyInputs[i].strides = view.strides;
                dummyInputs[i].dtype = view.dtype;
                in_mem_spaces[i] = inst.inBuffers[i].mem_space;
            }
            TensorNode dummyOutput;
            const auto &outView = compiled.nodeViews.at(inst.eclass_id);
            dummyOutput.setShape(outView.getShape());
            dummyOutput.strides = outView.strides;
            dummyOutput.dtype = outView.dtype;

            std::vector<Engine> inst_engines;
            kernel.matches(dummyInputs, dummyOutput, inst.outBuffer.mem_space, in_mem_spaces, {}, false, false, true,
                           true, &inst_engines);
            if (inst_engines.empty())
            {
                if (inst.outBuffer.mem_space.type == HandleType::CUDA)
                    inst_engines.push_back(Engine{inst.outBuffer.mem_space.idx, EngineType::CUDA_GPU});
                else
                    inst_engines.push_back(Engine{0, EngineType::CPU});
            }

            const Engine &primary_engine = inst_engines[0];

            sync.syncBefore(inst, inst_engines);

            KernelContext ctx;

#ifdef TG_USE_CUDA
            for (const Engine &eng : inst_engines)
            {
                if (eng.type == EngineType::CUDA_GPU || eng.type == EngineType::CUDA_DMA)
                {
                    cudaSetDevice(eng.idx);
                    ctx.cuda_streams.push_back(reinterpret_cast<void *>(sync.getCudaStream(eng)));
                }
            }
            if (!inst_engines.empty() &&
                (primary_engine.type == EngineType::CUDA_GPU || primary_engine.type == EngineType::CUDA_DMA))
            {
                cudaSetDevice(primary_engine.idx);
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