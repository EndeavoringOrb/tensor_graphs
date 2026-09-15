// tensor_graphs_cpp/test_inst.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#endif

#include "core/argparse.hpp"
#include "core/common/bench_utils.hpp"
#include "core/cost_model.hpp"
#include "core/graph.hpp"
#include "core/hardware.hpp"
#include "core/kernels.hpp"
#include "core/logging.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include "core/synchronizer.hpp"
#include "core/types.hpp"
#include "generated/build_context.gen.hpp"
#include "generated/kernels_all.gen.hpp"

#if defined(_WIN32) || defined(_WIN64)
static int64_t g_active_inst_idx = -1;
static std::string g_active_kernel_name;

inline LONG WINAPI TestInstCrashHandler(EXCEPTION_POINTERS *ep)
{
    if (!ep || !ep->ExceptionRecord)
        return EXCEPTION_CONTINUE_SEARCH;
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    if (code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_ARRAY_BOUNDS_EXCEEDED ||
        code == EXCEPTION_DATATYPE_MISALIGNMENT)
    {
        std::cerr << "\n========================================================\n"
                  << "[CRASH DETECTED IN TEST_INST] 0x" << std::hex << code << std::dec << "\n"
                  << "  Instruction Index: " << g_active_inst_idx << "\n"
                  << "  Kernel:            " << g_active_kernel_name << "\n"
                  << "  Fault Address:     0x" << std::hex << ep->ExceptionRecord->ExceptionAddress << std::dec << "\n";
        if (code == EXCEPTION_ACCESS_VIOLATION && ep->ExceptionRecord->NumberParameters >= 2)
        {
            ULONG_PTR op = ep->ExceptionRecord->ExceptionInformation[0];
            ULONG_PTR addr = ep->ExceptionRecord->ExceptionInformation[1];
            std::cerr << "  Details: Attempted to "
                      << (op == 0   ? "read from"
                          : op == 1 ? "write to"
                                    : "access")
                      << " invalid address 0x" << std::hex << addr << std::dec << "\n";
        }
        std::cerr << "========================================================\n" << std::flush;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

static std::string to_lower_str(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

static bool parse_inst_spec(const std::string &spec, uint64_t max_len, uint64_t &out_start, uint64_t &out_end)
{
    if (spec.empty())
        return false;
    auto colon = spec.find(':');
    auto dots = spec.find("..");
    try
    {
        if (colon != std::string::npos)
        {
            out_start = (colon == 0) ? 0 : std::stoull(spec.substr(0, colon));
            out_end = (colon + 1 >= spec.size()) ? max_len : std::stoull(spec.substr(colon + 1));
            return true;
        }
        if (dots != std::string::npos)
        {
            out_start = (dots == 0) ? 0 : std::stoull(spec.substr(0, dots));
            out_end = (dots + 2 >= spec.size()) ? max_len : std::stoull(spec.substr(dots + 2)) + 1;
            return true;
        }
        out_start = std::stoull(spec);
        out_end = out_start + 1;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

// TODO: dedup with main.cpp
static const float *sync_output_to_host(const float *device_ptr, uint64_t num_elements, std::vector<float> &host_buffer)
{
    const float *output_ptr = device_ptr;
#ifdef TG_USE_CUDA
    cudaPointerAttributes attrs;
    if (cudaPointerGetAttributes(&attrs, device_ptr) == cudaSuccess && attrs.type == cudaMemoryTypeDevice)
    {
        host_buffer.resize(num_elements);
        cudaMemcpy(host_buffer.data(), device_ptr, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
        output_ptr = host_buffer.data();
    }
#endif
    return output_ptr;
}

int main(int argc, char *argv[])
{
#if defined(_WIN32) || defined(_WIN64)
    AddVectoredExceptionHandler(1, TestInstCrashHandler);
#endif
    System::get();

    ArgParser parser("test_inst",
                     "Test / benchmark a specific instruction or set of instructions from a cache file in isolation.");
    parser.add_option({"--cache", "-c"}, "Path to compiled graph cache .bin file.", "");
    parser.add_option({"--inst", "-i"}, "Instruction index or range (e.g. 30402 or 30390:30405).", "");
    parser.add_option({"--op"}, "Filter by op name or op type (case-insensitive substring).", "");
    parser.add_option({"--bucket", "-b"}, "Bucket index in cache file (default: 0).", "0");
    parser.add_option({"--iters"}, "Number of execution iterations.", "1");
    parser.add_option({"--warmup"}, "Number of warmup iterations.", "0");
    parser.add_flag({"--dry-run"}, "Print instruction details and pre-flight check without executing.");
    parser.add_flag({"--isolated"}, "Run with isolated buffer extents instead of full arena.");
    parser.add_flag({"--check-nan"}, "Check output buffer for NaN or Inf values.");
    parser.add_flag({"--verbose", "-v"}, "Print sample input/output tensor values.");
    parser.add_positional("pos1", "Cache path, instruction index, or op filter.", "");
    parser.add_positional("pos2", "Instruction index or op filter.", "");

    std::vector<std::string> remaining_args;
    if (!parser.parse(argc, argv, &remaining_args))
    {
        return 1;
    }

    std::string cache_path = parser.get_option("--cache");
    std::string inst_spec = parser.get_option("--inst");
    std::string op_filter = parser.get_option("--op");
    int bucket_idx = std::stoi(parser.get_option("--bucket"));
    int iters = std::max(1, std::stoi(parser.get_option("--iters")));
    int warmup = std::max(0, std::stoi(parser.get_option("--warmup")));
    bool dry_run = parser.get_flag("--dry-run");
    bool isolated = parser.get_flag("--isolated");
    bool check_nan = parser.get_flag("--check-nan");
    bool verbose = parser.get_flag("--verbose");

    // Positionals fallback parsing
    std::string p1 = parser.get_positional("pos1");
    std::string p2 = parser.get_positional("pos2");

    if (cache_path.empty())
    {
        if (!p1.empty() && (p1.rfind(".bin") != std::string::npos || std::filesystem::exists(p1)))
        {
            cache_path = p1;
            p1 = "";
        }
        else if (!p2.empty() && (p2.rfind(".bin") != std::string::npos || std::filesystem::exists(p2)))
        {
            cache_path = p2;
            p2 = "";
        }
    }

    std::string remaining_filter = !p1.empty() ? p1 : p2;
    if (!remaining_filter.empty())
    {
        uint64_t dummy_s, dummy_e;
        if (inst_spec.empty() && parse_inst_spec(remaining_filter, 100000000, dummy_s, dummy_e))
        {
            inst_spec = remaining_filter;
        }
        else if (op_filter.empty())
        {
            op_filter = remaining_filter;
        }
    }

    if (cache_path.empty())
    {
        // Try locating any .bin in dirty_region_caches/
        if (std::filesystem::exists("dirty_region_caches"))
        {
            for (const auto &entry : std::filesystem::directory_iterator("dirty_region_caches"))
            {
                if (entry.path().extension() == ".bin")
                {
                    cache_path = entry.path().string();
                    std::cout << "[test_inst] Defaulting to cache file: " << cache_path << "\n";
                    break;
                }
            }
        }
    }

    if (cache_path.empty() || !std::filesystem::exists(cache_path))
    {
        std::cerr << "[test_inst Error] Cache file not found: '" << cache_path << "'\n";
        return 1;
    }

    std::cout << "[test_inst] Loading compiled graph from: " << cache_path << "\n";
    CacheFile cache = loadCacheFile(cache_path, false);
    if (!cache.isValid)
    {
        std::cerr << "[test_inst Error] Failed to load cache file: " << cache.invalidReason << "\n";
        return 1;
    }

    if (bucket_idx < 0 || static_cast<size_t>(bucket_idx) >= cache.compiledGraphs.size())
    {
        std::cerr << "[test_inst Error] Bucket index " << bucket_idx << " out of range (0.."
                  << cache.compiledGraphs.size() - 1 << ")\n";
        return 1;
    }

    const CompiledGraph &compiled = cache.compiledGraphs[bucket_idx];
    uint64_t n_instructions = compiled.instructions.size();

    std::cout << "[test_inst] Bucket " << bucket_idx << " loaded (" << n_instructions << " instructions).\n";

    // Filter matching instructions
    std::vector<uint64_t> target_indices;

    if (!inst_spec.empty())
    {
        uint64_t start_idx = 0, end_idx = 0;
        if (parse_inst_spec(inst_spec, n_instructions, start_idx, end_idx))
        {
            start_idx = std::min(start_idx, n_instructions);
            end_idx = std::min(end_idx, n_instructions);
            for (uint64_t i = start_idx; i < end_idx; ++i)
            {
                target_indices.push_back(i);
            }
        }
        else
        {
            std::cerr << "[test_inst Error] Invalid instruction specification: '" << inst_spec << "'\n";
            return 1;
        }
    }

    if (!op_filter.empty())
    {
        std::string filter_lower = to_lower_str(op_filter);
        std::vector<uint64_t> filtered;
        auto to_search = target_indices.empty() ? [&]() {
            std::vector<uint64_t> all_idx(n_instructions);
            for (uint64_t i = 0; i < n_instructions; ++i)
                all_idx[i] = i;
            return all_idx;
        }()
                                                : target_indices;

        for (uint64_t idx : to_search)
        {
            const auto &inst = compiled.instructions[idx];
            if (inst.kernel_id.value == 0 || !KernelRegistry::get().hasKernel(inst.kernel_id))
                continue;
            const auto &kernel = KernelRegistry::get().getKernel(inst.kernel_id);
            std::string opName = to_lower_str(kernel.opName);
            std::string opType = to_lower_str(toString(kernel.opType));
            std::string kernelName = to_lower_str(kernel.getName());
            std::string origin = to_lower_str(inst.debugOrigin);

            if (opName.find(filter_lower) != std::string::npos || opType.find(filter_lower) != std::string::npos ||
                kernelName.find(filter_lower) != std::string::npos || origin.find(filter_lower) != std::string::npos)
            {
                filtered.push_back(idx);
            }
        }
        target_indices = std::move(filtered);
    }

    if (target_indices.empty())
    {
        std::cout << "[test_inst] No instructions matched the criteria.\n";
        return 0;
    }

    std::cout << "[test_inst] Selected " << target_indices.size() << " instruction(s) to inspect / run.\n\n";

    // Determine memory peak requirements
    std::unordered_map<MemSpace, uint64_t> peakSizes;
    auto source_set = isolated ? target_indices : [&]() {
        std::vector<uint64_t> all_i(n_instructions);
        for (uint64_t i = 0; i < n_instructions; ++i)
            all_i[i] = i;
        return all_i;
    }();

    for (uint64_t idx : source_set)
    {
        const auto &inst = compiled.instructions[idx];
        if (inst.outBuffer.mem_space.type != HandleType::STORAGE && inst.outBuffer.offset >= 0)
        {
            uint64_t extent = static_cast<uint64_t>(inst.outBuffer.offset) + inst.outBuffer.size;
            peakSizes[inst.outBuffer.mem_space] = std::max(peakSizes[inst.outBuffer.mem_space], extent);
        }
        for (const auto &inBuf : inst.inBuffers)
        {
            if (inBuf.mem_space.type != HandleType::STORAGE && inBuf.offset >= 0)
            {
                uint64_t extent = static_cast<uint64_t>(inBuf.offset) + inBuf.size;
                peakSizes[inBuf.mem_space] = std::max(peakSizes[inBuf.mem_space], extent);
            }
        }
        if (compiled.nodeViews.count(inst.eclass_id))
        {
            const auto &v = compiled.nodeViews.at(inst.eclass_id);
            uint64_t extent = v.offset + countElements(v.getShape()) * getDTypeSize(v.dtype);
            peakSizes[inst.outBuffer.mem_space] = std::max(peakSizes[inst.outBuffer.mem_space], extent);
        }
        for (EClassId c : inst.children)
        {
            if (compiled.nodeViews.count(c))
            {
                const auto &v = compiled.nodeViews.at(c);
                uint64_t extent = v.offset + countElements(v.getShape()) * getDTypeSize(v.dtype);
                peakSizes[MemSpace{1, HandleType::CPP}] = std::max(peakSizes[MemSpace{1, HandleType::CPP}], extent);
            }
        }
    }

    std::cout << "[test_inst] Initializing memory manager (" << (isolated ? "ISOLATED" : "FULL ARENA") << ")...\n";
    for (const auto &pair : peakSizes)
    {
        std::cout << "  - " << pair.first << ": " << pair.second << " bytes (" << (pair.second / (1024.0 * 1024.0))
                  << " MB)\n";
    }

    MemoryManager memManager(peakSizes);
    memManager.init(peakSizes);

    // Populate CPU memory with non-zero 1.0f pattern
    DeviceBuffer *cpuBuf = memManager.getBuffer(MemSpace{1, HandleType::CPP});
    if (cpuBuf && cpuBuf->getBasePtr())
    {
        float one = 1.0f;
        uint32_t one_bits;
        std::memcpy(&one_bits, &one, 4);
        uint32_t *p32 = reinterpret_cast<uint32_t *>(cpuBuf->getBasePtr());
        for (uint64_t k = 0; k < cpuBuf->sizeBytes / 4; ++k)
        {
            p32[k] = one_bits;
        }
    }

    // Write staged constants
    for (const auto &pair : compiled.constantStaging)
    {
        EClassId eclass_id = pair.first;
        if (compiled.nodeViews.count(eclass_id))
        {
            const TensorView &view = compiled.nodeViews.at(eclass_id);
            memManager.write(MemSpace{1, HandleType::CPP}, view.offset, pair.second->data(), pair.second->size());
        }
    }
    for (const auto &pair : cache.constants)
    {
        LogicalId logicalId = pair.first;
        auto it = compiled.logical_to_eclass.find(logicalId);
        if (it != compiled.logical_to_eclass.end() && compiled.nodeViews.count(it->second))
        {
            const TensorView &view = compiled.nodeViews.at(it->second);
            memManager.write(MemSpace{1, HandleType::CPP}, view.offset, pair.second->data(), pair.second->size());
        }
    }

    // Test loop
    Synchronizer sync;
    int passed = 0;
    int failed = 0;

    for (uint64_t inst_idx : target_indices)
    {
        const OpInstruction &inst = compiled.instructions[inst_idx];
        if (inst.kernel_id.value == 0 || !KernelRegistry::get().hasKernel(inst.kernel_id))
        {
            std::cout << "Skipping inst #" << inst_idx << " (Kernel ID 0x" << std::hex << inst.kernel_id.value
                      << std::dec << " not registered).\n";
            continue;
        }

        const KernelEntry &kernel = KernelRegistry::get().getKernel(inst.kernel_id);
        std::string kernelName = kernel.opName.empty() ? toString(kernel.opType) : kernel.opName;

#if defined(_WIN32) || defined(_WIN64)
        g_active_inst_idx = inst_idx;
        g_active_kernel_name = kernelName;
#endif

        std::cout << "\n================================================================================\n";
        std::cout << "Instruction [" << inst_idx << " / " << n_instructions << "]\n";
        std::cout << "Kernel:       " << kernelName << " (0x" << std::hex << inst.kernel_id.value << std::dec << ")\n";
        std::cout << "OpType:       " << toString(kernel.opType) << (kernel.is_view ? " [VIEW]" : "") << "\n";
        std::cout << "DebugOrigin:  " << (inst.debugOrigin.empty() ? "N/A" : inst.debugOrigin) << "\n";

        // Output details
        const TensorView &outView = compiled.nodeViews.at(inst.eclass_id);
        uint64_t out_extent = getRequiredBufferSize(outView) * getDTypeSize(outView.dtype);
        std::cout << "Output:\n";
        std::cout << "  EClass: " << inst.eclass_id.value << " | LogicalId: " << inst.logical_id.value << "\n";
        std::cout << "  Buffer: ID " << inst.outBuffer.id.value << " | " << inst.outBuffer.mem_space << " | Offset: 0x"
                  << std::hex << inst.outBuffer.offset << std::dec << " (" << inst.outBuffer.offset
                  << ") | Size: " << inst.outBuffer.size << " B\n";
        std::cout << "  View:   " << toString(outView.dtype) << " " << toString(outView.getShape())
                  << " | Strides: " << toString(outView.strides) << " | ViewOffset: 0x" << std::hex << outView.offset
                  << std::dec << " | Extent: " << out_extent << " B\n";

        // Input details
        std::cout << "Inputs (" << inst.children.size() << "):\n";
        bool preflight_ok = true;

        for (uint64_t i = 0; i < inst.children.size(); ++i)
        {
            EClassId c_id = inst.children[i];
            const TensorView &inView = compiled.nodeViews.at(c_id);
            const ParallelBuffer &inBuf = inst.inBuffers[i];
            uint64_t in_extent = getRequiredBufferSize(inView) * getDTypeSize(inView.dtype);

            std::string const_desc = "";
            if (compiled.constantStaging.count(c_id))
            {
                const auto &cd = *compiled.constantStaging.at(c_id);
                if (inView.dtype == DType::INT32 && cd.size() >= 4)
                {
                    int32_t val;
                    std::memcpy(&val, cd.data(), 4);
                    const_desc = " [Constant: " + std::to_string(val) + "]";
                }
            }

            std::cout << "  [" << i << "] EClass: " << c_id.value << " | Buf ID " << inBuf.id.value << " | "
                      << inBuf.mem_space << " | Offset: 0x" << std::hex << inBuf.offset << std::dec
                      << " | Size: " << inBuf.size << " B" << const_desc << "\n";
            std::cout << "      View: " << toString(inView.dtype) << " " << toString(inView.getShape())
                      << " | Strides: " << toString(inView.strides) << " | ViewOffset: 0x" << std::hex << inView.offset
                      << std::dec << " | Extent: " << in_extent << " B\n";

            // Bounds check
            if (inBuf.mem_space.type != HandleType::STORAGE)
            {
                if (inBuf.offset < 0)
                {
                    std::cerr << "  [PRE-FLIGHT ERROR] Input #" << i << " buffer has negative offset (" << inBuf.offset
                              << ")!\n";
                    preflight_ok = false;
                }
                if (inView.offset < static_cast<uint64_t>(inBuf.offset))
                {
                    std::cerr << "  [PRE-FLIGHT ERROR] Input #" << i << " view offset (0x" << std::hex << inView.offset
                              << ") starts before buffer offset (0x" << inBuf.offset << std::dec << ")!\n";
                    preflight_ok = false;
                }
                if (inView.offset + in_extent > static_cast<uint64_t>(inBuf.offset) + inBuf.size)
                {
                    std::cerr << "  [PRE-FLIGHT ERROR] Input #" << i << " view extent (0x" << std::hex
                              << (inView.offset + in_extent) << ") overflows buffer bounds (0x"
                              << (inBuf.offset + inBuf.size) << std::dec << ")!\n";
                    preflight_ok = false;
                }
            }
        }

        // Check output bounds
        if (inst.outBuffer.mem_space.type != HandleType::STORAGE)
        {
            if (inst.outBuffer.offset < 0)
            {
                std::cerr << "  [PRE-FLIGHT ERROR] Output buffer has negative offset (" << inst.outBuffer.offset
                          << ")!\n";
                preflight_ok = false;
            }
            if (outView.offset < static_cast<uint64_t>(inst.outBuffer.offset))
            {
                std::cerr << "  [PRE-FLIGHT ERROR] Output view offset (0x" << std::hex << outView.offset
                          << ") starts before buffer offset (0x" << inst.outBuffer.offset << std::dec << ")!\n";
                preflight_ok = false;
            }
            if (outView.offset + out_extent > static_cast<uint64_t>(inst.outBuffer.offset) + inst.outBuffer.size)
            {
                std::cerr << "  [PRE-FLIGHT ERROR] Output view extent (0x" << std::hex << (outView.offset + out_extent)
                          << ") overflows buffer bounds (0x" << (inst.outBuffer.offset + inst.outBuffer.size)
                          << std::dec << ")!\n";
                preflight_ok = false;
            }
        }

        // Specific check for CONCAT kernel
        if (kernel.opType == OpType::CONCAT)
        {
            if (inst.children.size() < 2)
            {
                std::cerr << "  [PRE-FLIGHT ERROR] CONCAT instruction has fewer than 2 children!\n";
                preflight_ok = false;
            }
        }

        if (preflight_ok)
        {
            std::cout << "Pre-flight Bounds Check: [PASSED]\n";
        }
        else
        {
            std::cerr << "Pre-flight Bounds Check: [FAILED - POTENTIAL MEMORY SAFETY HAZARD]\n";
        }

        if (dry_run)
        {
            std::cout << "[Dry-Run Mode: Skipping Kernel Execution]\n";
            continue;
        }

        // Prepare context and run
        try
        {
            sync.syncBefore(inst, inst.engines);
            KernelContext ctx;

#ifdef TG_USE_CUDA
            int primary_cuda_device = -1;
            for (const Engine &eng : inst.engines)
            {
                if (eng.type == EngineType::CUDA_GPU || eng.type == EngineType::CUDA_DMA)
                {
                    if (primary_cuda_device == -1)
                        primary_cuda_device = static_cast<int>(eng.idx);
                    ctx.cuda_streams.push_back(reinterpret_cast<void *>(sync.getCudaStream(eng)));
                }
            }
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

                LogicalId lid = compiled.has_logical_id(inst.children[i]) ? compiled.get_logical_id(inst.children[i])
                                                                          : LogicalId{UINT32_MAX};
                inBufObj->setupInput(ctx, inView, lid);
            }

            DeviceBuffer *outBufObj = memManager.getBuffer(inst.outBuffer.mem_space);
            if (!outBufObj)
                Error::throw_err("Output DeviceBuffer not found");

            LogicalId out_lid = compiled.has_logical_id(inst.eclass_id) ? compiled.get_logical_id(inst.eclass_id)
                                                                        : LogicalId{UINT32_MAX};
            outBufObj->setupOutput(ctx, outView, out_lid);

            std::cout << "Pointer Diagnostics:\n";
            std::cout << "  out_ptr: " << ctx.outputs[0] << "\n";
            for (size_t i = 0; i < ctx.inputs.size(); ++i)
            {
                std::cout << "  in_ptr[" << i << "]: " << ctx.inputs[i];
                if (i == 0 && kernel.opType == OpType::CONCAT && ctx.inputs[0])
                {
                    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[0]);
                    std::cout << " (axis=" << axis << ")";
                }
                std::cout << "\n";
            }

            if (!kernel.is_view && kernel.run)
            {
                std::cout << "Executing kernel (" << iters << " iters, " << warmup << " warmup)..." << std::flush;

                for (int w = 0; w < warmup; ++w)
                {
                    kernel.run(ctx);
                }
                sync.syncAll();

                auto t_start = std::chrono::high_resolution_clock::now();
                for (int it = 0; it < iters; ++it)
                {
                    kernel.run(ctx);
                }
                sync.syncAll();
                auto t_end = std::chrono::high_resolution_clock::now();

                double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                double avg_ms = total_ms / iters;
                std::cout << " [PASSED] Avg Runtime: " << std::fixed << std::setprecision(4) << avg_ms << " ms\n";

                if (check_nan && ctx.outputs[0] && outView.dtype == DType::FLOAT32)
                {
                    std::vector<float> host_out;
                    const float *host_ptr = sync_output_to_host(static_cast<const float *>(ctx.outputs[0]),
                                                                countElements(outView.getShape()), host_out);
                    uint64_t n_elems = countElements(outView.getShape());
                    uint64_t nan_count = 0;
                    for (uint64_t k = 0; k < n_elems; ++k)
                    {
                        if (std::isnan(host_ptr[k]) || std::isinf(host_ptr[k]))
                            nan_count++;
                    }
                    if (nan_count > 0)
                    {
                        std::cerr << "  [CHECK-NAN WARNING] Found " << nan_count
                                  << " NaN/Inf values in output tensor!\n";
                    }
                    else
                    {
                        std::cout << "  [CHECK-NAN] Output values are all finite.\n";
                    }
                }

                if (verbose && ctx.outputs[0])
                {
                    std::cout << "  Output Preview (first 8 values): [";
                    if (outView.dtype == DType::FLOAT32)
                    {
                        std::vector<float> host_out;
                        const float *host_ptr = sync_output_to_host(static_cast<const float *>(ctx.outputs[0]),
                                                                    countElements(outView.getShape()), host_out);
                        uint64_t n_elems = std::min<uint64_t>(8, countElements(outView.getShape()));
                        for (uint64_t k = 0; k < n_elems; ++k)
                        {
                            if (k > 0)
                                std::cout << ", ";
                            std::cout << host_ptr[k];
                        }
                    }
                    else if (outView.dtype == DType::INT32)
                    {
                        const int32_t *host_ptr = static_cast<const int32_t *>(ctx.outputs[0]);
                        uint64_t n_elems = std::min<uint64_t>(8, countElements(outView.getShape()));
                        for (uint64_t k = 0; k < n_elems; ++k)
                        {
                            if (k > 0)
                                std::cout << ", ";
                            std::cout << host_ptr[k];
                        }
                    }
                    std::cout << "...]\n";
                }
            }
            else
            {
                std::cout << "Instruction is a View operation (no device computation executed).\n";
            }

            for (const ParallelBuffer &inBuf : inst.inBuffers)
            {
                memManager.getBuffer(inBuf.mem_space)->cleanupContext(ctx);
            }
            outBufObj->cleanupContext(ctx);
            passed++;
        }
        catch (const std::exception &e)
        {
            std::cerr << "\n[EXECUTION FAILED] Exception in instruction " << inst_idx << ": " << e.what() << "\n";
            failed++;
        }
    }

    std::cout << "\n--------------------------------------------------------------------------------\n";
    std::cout << "Summary: " << passed << " passed, " << failed << " failed.\n";
    std::cout << "================================================================================\n";

    return (failed > 0) ? 1 : 0;
}