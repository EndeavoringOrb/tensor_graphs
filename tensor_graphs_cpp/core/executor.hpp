#pragma once
#include <cmath>
#include <map>
#include <fstream>
#include <chrono>
#include <cstring>
#include <unordered_map>
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

    // Host-only fast path: for a compiled graph whose buffers all live in host
    // memory or storage and whose kernels all run on the CPU, kernels run
    // synchronously in order, so each instruction's KernelContext (pointers
    // into the fixed arenas) is built once and reused.
    struct PreparedInstruction
    {
        KernelFunc run;
        KernelContext ctx;
        // Diagnostics (TG_EXEC_PROFILE): label and accumulated run time.
        std::string label;
        double seconds = 0.0;
        uint64_t calls = 0;
    };
    struct PreparedConstant
    {
        uint8_t *dst;
        const std::vector<uint8_t> *data;
    };
    struct PreparedGraph
    {
        bool hostOnly = false;
        std::vector<PreparedInstruction> instructions;
        std::vector<PreparedConstant> constants;
    };
    std::unordered_map<const CompiledGraph *, PreparedGraph> prepared;
    bool debugEnvironment = false;
    // TG_EXEC_PROFILE=<file>: time every fast-path kernel; the totals per
    // kernel and output shape are written when the executor is destroyed.
    const char *profilePath = std::getenv("TG_EXEC_PROFILE");

    PreparedGraph &prepare(const CompiledGraph &compiled)
    {
        auto it = prepared.find(&compiled);
        if (it != prepared.end())
            return it->second;
        PreparedGraph pg;
        pg.hostOnly = true;
        auto hostSpace = [](const MemSpace &ms) {
            return ms.type == HandleType::CPP || ms.type == HandleType::STORAGE;
        };
        for (const OpInstruction &inst : compiled.instructions)
        {
            bool host = hostSpace(inst.outBuffer.mem_space);
            for (const ParallelBuffer &b : inst.inBuffers)
                host = host && hostSpace(b.mem_space);
            for (const Engine &e : inst.engines)
                host = host && e.type == EngineType::CPU;
            if (!host)
            {
                pg.hostOnly = false;
                break;
            }
        }
        if (pg.hostOnly)
        {
            DeviceBuffer *hostBuffer = memManager.getBuffer(MemSpace{1, HandleType::CPP});
            for (const auto &pair : compiled.constantStaging)
            {
                if (compiled.nodeViews.count(pair.first) == 0)
                    continue;
                const TensorView &view = compiled.nodeViews.at(pair.first);
                if (!hostBuffer || !hostBuffer->getBasePtr())
                    Error::throw_err("[Executor.prepare] host arena not initialized");
                pg.constants.push_back({hostBuffer->getBasePtr() + view.offset, pair.second.get()});
            }
            for (const OpInstruction &inst : compiled.instructions)
            {
                const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.kernel_id);
                if (kernel.is_view || !kernel.run)
                    continue;
                PreparedInstruction pi;
                pi.run = kernel.run;
                if (profilePath)
                    pi.label = kernel.opName + " " + toString(compiled.nodeViews.at(inst.eclass_id).getShape());
                for (uint64_t i = 0; i < inst.children.size(); ++i)
                {
                    LogicalId logical_id;
                    if (compiled.has_logical_id(inst.children[i]))
                        logical_id = compiled.get_logical_id(inst.children[i]);
                    memManager.getBuffer(inst.inBuffers[i].mem_space)
                        ->setupInput(pi.ctx, compiled.nodeViews.at(inst.children[i]), logical_id);
                }
                LogicalId logical_id;
                if (compiled.has_logical_id(inst.eclass_id))
                    logical_id = compiled.get_logical_id(inst.eclass_id);
                memManager.getBuffer(inst.outBuffer.mem_space)
                    ->setupOutput(pi.ctx, compiled.nodeViews.at(inst.eclass_id), logical_id);
                pg.instructions.push_back(std::move(pi));
            }
        }
        return prepared.emplace(&compiled, std::move(pg)).first->second;
    }

  public:
    ~Executor()
    {
        if (!profilePath)
            return;
        std::ofstream out(profilePath, std::ios::app);
        if (!out)
            return; // destructor: nowhere to report
        for (const auto &[graph, pg] : prepared)
        {
            std::map<std::string, std::pair<double, uint64_t>> totals;
            uint64_t runs = 0;
            for (const PreparedInstruction &pi : pg.instructions)
            {
                totals[pi.label].first += pi.seconds;
                totals[pi.label].second += pi.calls;
                runs = std::max(runs, pi.calls);
            }
            if (runs == 0)
                continue;
            double total = 0.0;
            for (const auto &[label, t] : totals)
                total += t.first;
            uint64_t constant_bytes = 0;
            for (const PreparedConstant &c : pg.constants)
                constant_bytes += c.data->size();
            out << "== graph " << graph << " runs=" << runs << " ms/run=" << 1000.0 * total / runs
                << " constants=" << pg.constants.size() << " constant_bytes=" << constant_bytes << '\n';
            std::vector<std::pair<double, std::string>> sorted;
            for (const auto &[label, t] : totals)
                sorted.push_back({t.first, label + " x" + std::to_string(t.second / runs)});
            std::sort(sorted.rbegin(), sorted.rend());
            for (const auto &[seconds, label] : sorted)
                out << 1000.0 * seconds / runs << '\t' << label << '\n';
        }
    }

    Executor(MemoryManager &mm) : memManager(mm)
    {
        for (const char *name : {"TG_DUMP_BASE_REFS", "TG_COMPARE_BASE_REFS", "TG_CHECK_NAN", "TG_BASE_REF_STEP"})
        {
            const char *value = std::getenv(name);
            debugEnvironment = debugEnvironment || (value && *value);
        }
    }

    void run(const CompiledGraph &compiled, const Debug::Callback &debugCallback = nullptr)
    {
        if (!debugCallback && !debugEnvironment)
        {
            PreparedGraph &pg = prepare(compiled);
            if (pg.hostOnly)
            {
                for (const PreparedConstant &c : pg.constants)
                    std::memcpy(c.dst, c.data->data(), c.data->size());
                if (profilePath)
                {
                    for (PreparedInstruction &pi : pg.instructions)
                    {
                        const auto start = std::chrono::steady_clock::now();
                        pi.run(pi.ctx);
                        pi.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                        ++pi.calls;
                    }
                    return;
                }
                for (const PreparedInstruction &pi : pg.instructions)
                    pi.run(pi.ctx);
                return;
            }
        }
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

            if (!kernel.is_view && !kernel.requiresContiguous.empty())
            {
                for (size_t c_idx = 0; c_idx < inst.children.size(); ++c_idx)
                {
                    size_t rule_idx = std::min(c_idx, kernel.requiresContiguous.size() - 1);
                    if (kernel.requiresContiguous[rule_idx])
                    {
                        const TensorView &in_view = compiled.nodeViews.at(inst.children[c_idx]);
                        if (!isContiguous(in_view))
                        {
                            std::stringstream ss;
                            ss << "[Executor] Instruction " << idx << " (" << kernel_name << ") input #" << c_idx
                               << " (eclass " << inst.children[c_idx].value << ") is non-contiguous, but kernel requires contiguous input."
                               << " Shape=" << toString(in_view.getShape()) << " Strides=" << toString(in_view.strides);
                            Error::throw_err(ss.str());
                        }
                    }
                }
            }

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
                    bool has_bad = false;
                    for (uint64_t k = 0; k < n_elems; ++k)
                    {
                        if (std::isnan(fvals[k]) || std::isinf(fvals[k]))
                        {
                            has_bad = true;
                            nan_report_count++;
                            std::cerr << "[BAD_VAL DETECTED] inst=" << idx << "/" << nInst
                                      << " kernel=" << kernel_name << " (id=0x" << std::hex << inst.kernel_id.value << std::dec << ")"
                                      << " reqContig=" << (kernel.requiresContiguous.empty() ? -1 : (int)kernel.requiresContiguous[0])
                                      << " eclass=" << inst.eclass_id.value
                                      << " logical=" << inst.logical_id.value
                                      << " origin=" << inst.debugOrigin
                                      << " first_bad_idx=" << k << "/" << n_elems
                                      << " val=" << fvals[k]
                                      << " out_offset=" << inst.outBuffer.offset
                                      << " out_size=" << inst.outBuffer.size << std::endl;
                            break;
                        }
                    }
                    if (has_bad)
                    {
                        uint64_t target_k = 0;
                        for (uint64_t k = 0; k < n_elems; ++k) {
                            if (std::isnan(fvals[k]) || std::isinf(fvals[k])) { target_k = k; break; }
                        }
                        for (size_t c = 0; c < inst.children.size(); ++c)
                        {
                            const TensorView &cView = compiled.nodeViews.at(inst.children[c]);
                            std::cerr << "  child[" << c << "] eclass=" << inst.children[c].value
                                      << " buf_offset=" << inst.inBuffers[c].offset
                                      << " buf_size=" << inst.inBuffers[c].size
                                      << " dtype=" << static_cast<int>(cView.dtype)
                                      << " shape=[";
                            for (auto d : cView.getShape()) std::cerr << d << ",";
                            std::cerr << "] strides=[";
                            for (auto s : cView.strides) std::cerr << s << ",";
                            std::cerr << "]";
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
                                    uint64_t nan_cnt = 0, inf_cnt = 0;
                                    for (uint64_t ik = 0; ik < in_n; ++ik)
                                    {
                                        if (std::isnan(in_f[ik])) nan_cnt++;
                                        if (std::isinf(in_f[ik])) inf_cnt++;
                                    }
                                    std::cerr << " nan_cnt=" << nan_cnt << " inf_cnt=" << inf_cnt
                                              << " val_at_k=" << (target_k < in_n ? in_f[target_k] : -999999.0f)
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
