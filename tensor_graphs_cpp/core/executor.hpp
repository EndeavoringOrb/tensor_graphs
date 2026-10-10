#pragma once
#include <cmath>
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
        bool disableTimer = false;
#ifdef TG_DEBUG
        disableTimer = false;
#endif
        ProgressTimer timer(nInst, "running", disableTimer);
        int nan_report_count = 0;

        // TODO: we should write constants for all buckets once in Session::compile
        for (const auto &pair : compiled.constantStaging)
        {
            EClassId eclass_id = pair.first;
            if (compiled.nodeViews.count(eclass_id))
            {
                const TensorView &view = compiled.nodeViews.at(eclass_id);
                memManager.write(MemSpace{1, HandleType::CPP}, view.offset, pair.second->data(), pair.second->size());
            }
        }

        const char *dump_path = std::getenv("TG_DUMP_BASE_REFS");
        const char *compare_path = std::getenv("TG_COMPARE_BASE_REFS");
        const char *check_nan_env = std::getenv("TG_CHECK_NAN");
        static int global_step = 0;
        int current_step = global_step++;
        const char *step_env = std::getenv("TG_BASE_REF_STEP");
        int target_step = (step_env && *step_env) ? std::atoi(step_env) : -1;
        if (target_step >= 0 && target_step != current_step)
        {
            dump_path = nullptr;
            compare_path = nullptr;
        }
        if (compare_path && *compare_path)
        {
            Debug::BaseRefVerifier::get().loadFromFile(compare_path);
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
                try
                {
                    kernel.run(ctx);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "[Executor ERROR at instruction " << idx << "] kernel=" << kernel_name
                              << " eclass=" << inst.eclass_id.value
                              << " logical_id=" << inst.logical_id.value
                              << " debugOrigin=" << inst.debugOrigin << std::endl;
                    std::cerr << "  children: ";
                    for (auto c : inst.children)
                        std::cerr << c.value << " ";
                    std::cerr << std::endl;
                    throw;
                }
                issued_work = true;
            }

            sync.markExecuted(inst, inst_engines, issued_work);

            if (check_nan_env && *check_nan_env && issued_work && outView.dtype == DType::FLOAT32 && nan_report_count < 10)
            {
                sync.syncEngines(inst_engines);
                const void *nan_check_ptr = nullptr;
                std::vector<uint8_t> tmp_nan_host;
                if (outBufObj->mem_space.type == HandleType::CPP)
                {
                    nan_check_ptr = ctx.outputs[0];
                }
#ifdef TG_USE_CUDA
                else if (outBufObj->mem_space.type == HandleType::CUDA)
                {
                    uint64_t sz = getStorageSpanBytes(outView);
                    tmp_nan_host.resize(sz);
                    cudaSetDevice(outBufObj->mem_space.idx);
                    cudaMemcpy(tmp_nan_host.data(), ctx.outputs[0], sz, cudaMemcpyDeviceToHost);
                    nan_check_ptr = tmp_nan_host.data();
                }
#endif
                if (nan_check_ptr)
                {
                    const float *fvals = static_cast<const float *>(nan_check_ptr);
                    uint64_t n_elems = countElements(outView);
                    bool has_nan = false;
                    for (uint64_t k = 0; k < n_elems; ++k)
                    {
                        if (std::isnan(fvals[k]))
                        {
                            has_nan = true;
                            nan_report_count++;
                            std::cerr << "[NAN DETECTED] inst=" << idx << "/" << nInst
                                      << " kernel=" << kernel_name << " eclass=" << inst.eclass_id.value
                                      << " logical=" << inst.logical_id.value
                                      << " origin=" << inst.debugOrigin
                                      << " first_nan_idx=" << k << "/" << n_elems
                                      << " out_offset=" << inst.outBuffer.offset
                                      << " out_size=" << inst.outBuffer.size << std::endl;
                            break;
                        }
                    }
                    if (has_nan)
                    {
                        for (size_t c = 0; c < inst.children.size(); ++c)
                        {
                            const TensorView &cView = compiled.nodeViews.at(inst.children[c]);
                            std::cerr << "  child[" << c << "] eclass=" << inst.children[c].value
                                      << " buf_offset=" << inst.inBuffers[c].offset
                                      << " buf_size=" << inst.inBuffers[c].size
                                      << " dtype=" << static_cast<int>(cView.dtype);
                            if (cView.dtype == DType::FLOAT32 && ctx.inputs[c])
                            {
                                const float *in_f = nullptr;
                                std::vector<uint8_t> tmp_in;
                                if (inst.inBuffers[c].mem_space.type == HandleType::CPP)
                                {
                                    in_f = static_cast<const float *>(ctx.inputs[c]);
                                }
#ifdef TG_USE_CUDA
                                else if (inst.inBuffers[c].mem_space.type == HandleType::CUDA)
                                {
                                    uint64_t in_sz = getStorageSpanBytes(cView);
                                    tmp_in.resize(in_sz);
                                    cudaSetDevice(inst.inBuffers[c].mem_space.idx);
                                    cudaMemcpy(tmp_in.data(), ctx.inputs[c], in_sz, cudaMemcpyDeviceToHost);
                                    in_f = reinterpret_cast<const float *>(tmp_in.data());
                                }
#endif
                                if (in_f)
                                {
                                    uint64_t in_n = countElements(cView);
                                    bool in_has_nan = false;
                                    for (uint64_t ik = 0; ik < in_n; ++ik)
                                    {
                                        if (std::isnan(in_f[ik]))
                                        {
                                            in_has_nan = true;
                                            break;
                                        }
                                    }
                                    std::cerr << " has_nan=" << (in_has_nan ? "YES" : "NO")
                                              << " sample[0]=" << in_f[0];
                                }
                            }
                            std::cerr << std::endl;
                        }
                    }
                }
            }

            if ((dump_path && *dump_path) || (compare_path && *compare_path))
            {
                sync.syncEngines(inst_engines);
                const void *host_ptr = nullptr;
                std::vector<uint8_t> tmp_cuda_host;
                if (outBufObj->mem_space.type == HandleType::CPP)
                {
                    host_ptr = ctx.outputs[0];
                }
#ifdef TG_USE_CUDA
                else if (outBufObj->mem_space.type == HandleType::CUDA)
                {
                    uint64_t sz = getStorageSpanBytes(outView);
                    tmp_cuda_host.resize(sz);
                    cudaSetDevice(outBufObj->mem_space.idx);
                    cudaMemcpy(tmp_cuda_host.data(), ctx.outputs[0], sz, cudaMemcpyDeviceToHost);
                    host_ptr = tmp_cuda_host.data();
                }
#endif
                if (host_ptr)
                {
                    uint32_t base_id = (inst.logical_id.value != UINT32_MAX)
                                           ? inst.logical_id.value
                                           : Debug::BaseRefVerifier::get().getBase(inst.eclass_id).value;
                    if (dump_path && *dump_path)
                    {
                        Debug::BaseRefVerifier::get().record(
                            base_id, kernel_name, inst.debugOrigin,
                            outView.getShape(), outView.strides, outView.dtype, host_ptr);
                    }
                    if (compare_path && *compare_path)
                    {
                        Debug::BaseRefVerifier::get().compare(
                            base_id, kernel_name, inst.debugOrigin,
                            outView.getShape(), outView.strides, outView.dtype, host_ptr, idx, inst.eclass_id.value);
                    }
                }
            }

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

        if (dump_path && *dump_path)
        {
            Debug::BaseRefVerifier::get().dumpToFile(dump_path);
        }
        if (compare_path && *compare_path)
        {
            std::cout << "[BaseRefVerifier SUMMARY] Matches: " << Debug::BaseRefVerifier::get().matchCount
                      << ", Mismatches: " << Debug::BaseRefVerifier::get().mismatchCount << std::endl;
        }
    }
};
