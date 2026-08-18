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

        // Restore bucket-local EGraph constants into the scratchpad
        std::unordered_set<EClassId> restored_constants;
        for (const auto &inst : compiled.instructions)
        {
            for (size_t i = 0; i < inst.children.size(); ++i)
            {
                EClassId child = inst.children[i];

                // If it has no LogicalId but exists in staging, it's an EGraph-generated constant
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
            EngineType current_engine = kernel.engines.empty() ? EngineType::CPU : kernel.engines[0].type;

            sync.syncBefore(inst, current_engine);

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

            bool issued_work = false;
            if (!kernel.is_view && kernel.run)
            {
                kernel.run(ctx);
                issued_work = true;
            }

            sync.markExecuted(inst, current_engine, issued_work);

#ifdef DEBUG
            sync.syncAll();

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
        sync.syncAll();
    }
};