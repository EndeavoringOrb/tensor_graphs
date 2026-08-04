#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#endif

#include "core/argparse.hpp"
#include "core/common/bench_utils.hpp"
#include "core/cost_model.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include "core/types.hpp"
#include "generated/build_context.gen.hpp"
#include "generated/kernels_all.gen.hpp"

int main(int argc, char *argv[])
{
    ArgParser parser("bench", "Benchmark registered kernels.");
    parser.add_option({"--skip", "-s"}, "Number of kernels to skip.", "0");
    parser.add_flag({"--list", "-l"}, "Only list the configurations, do not benchmark.");
    parser.add_option({"--cache"}, "Path to cache file. If provided, benchmark only configurations from this cache.",
                      "");
    parser.add_positional("targetKernel", "Bench only kernels whose name contain this string.", "");

    if (!parser.parse(argc, argv))
    {
        return 1;
    }

    int skipCount = std::stoi(parser.get_option("--skip"));
    bool listOnly = parser.get_flag("--list");
    std::string cachePath = parser.get_option("--cache");
    std::string targetKernel = parser.get_positional("targetKernel");

    std::filesystem::create_directories("benchmarks");
    std::string callsPath = "benchmarks/calls.bin";
    std::string recordsPath = "benchmarks/records.bin";

    if (!targetKernel.empty())
    {
        std::cout << "Filtering benchmarks for kernel containing: " << targetKernel << std::endl;
    }

    CostModel costModel;
    costModel.load(recordsPath);

    std::unordered_set<std::string> recordedKeys;
    std::ifstream recordsFile(recordsPath, std::ios::binary);
    if (recordsFile.is_open())
    {
        BinaryReader br(recordsFile);
        while (recordsFile.peek() != EOF)
        {
            Record r;
            br.read(r);
            r.runTime = 0.0f;
            recordedKeys.insert(serializeToString(r));
        }
    }

    std::vector<Record> toBenchmark;
    std::unordered_set<std::string> seenCalls;

    std::vector<Record> candidates;
    if (!cachePath.empty())
    {
        auto recordsByUid = getRecordsFromCache(cachePath);
        for (auto &kv : recordsByUid)
        {
            for (auto &r : kv.second)
            {
                candidates.push_back(std::move(r));
            }
        }
    }
    else
    {
        std::ifstream callsFile(callsPath, std::ios::binary);
        if (!callsFile.is_open())
        {
            std::cerr << "No calls file found at " << callsPath
                      << ". Enable TENSOR_GRAPHS_LOG_COST_CALLS and run an inference pass first." << std::endl;
            return 0;
        }

        BinaryReader br(callsFile);
        while (callsFile.peek() != EOF)
        {
            Record r;
            br.read(r);
            candidates.push_back(std::move(r));
        }
    }

    for (Record &r : candidates)
    {
        r.runTime = 0.0f;
        r.buildContextId = BUILD_CONTEXT_ID;
        std::string key = serializeToString(r);

        if (recordedKeys.find(key) == recordedKeys.end() && seenCalls.find(key) == seenCalls.end())
        {
            seenCalls.insert(key);
            if (r.hwTag == HW_TAG && KernelRegistry::get().hasKernel(r.kernelId))
            {
                const auto &kernel = KernelRegistry::get().getKernel(r.kernelId);
                std::string name = kernel.opName.empty() ? toString(kernel.opType) : kernel.opName;

                if (!targetKernel.empty() && name.find(targetKernel) == std::string::npos)
                    continue;

                toBenchmark.push_back(std::move(r));
            }
        }
    }

    if (toBenchmark.empty())
    {
        std::cout << "No kernels match the filters or all already benchmarked." << std::endl;
        return 0;
    }

    for (uint32_t i = 0; i < toBenchmark.size(); i++)
    {
        Record &r = toBenchmark[i];
        float cost = costModel.estimateCost(r.kernelId, r.outputShape, r.outputStrides, r.outputDType, r.inputShapes,
                                            r.inputStrides, r.inputDTypes, r.inputConstants);
        r.runTime = std::isinf(cost) ? -1.0f : cost;
    }

    std::stable_sort(toBenchmark.begin(), toBenchmark.end(), [&](const Record &ra, const Record &rb) {
        float costA = ra.runTime;
        float costB = rb.runTime;

        if (std::abs(costA - costB) < 1e-7)
        {
            bool isRefA = KernelRegistry::get().getKernel(ra.kernelId).isReference;
            bool isRefB = KernelRegistry::get().getKernel(rb.kernelId).isReference;
            if (isRefA != isRefB)
                return !isRefA;

            auto getVolume = [](const Record &r) {
                uint64_t v = 1;
                for (uint32_t d : r.outputShape)
                    v *= d;
                return v;
            };
            return getVolume(ra) < getVolume(rb);
        }
        return costA < costB;
    });

    uint64_t startIdx = (skipCount > (int)toBenchmark.size()) ? toBenchmark.size() : (uint64_t)std::max(0, skipCount);

    if (startIdx > 0)
    {
        std::cout << "Skipping the first " << startIdx << " kernels..." << std::endl;
    }

    std::cout << (listOnly ? "Listing " : "Benchmarking ") << toBenchmark.size() - startIdx << " configurations..."
              << std::endl;

    std::ofstream outFile;
    if (!listOnly)
    {
        outFile.open(recordsPath, std::ios::app | std::ios::binary);
    }
    BinaryWriter bw(outFile);

    for (uint64_t i = startIdx; i < toBenchmark.size(); ++i)
    {
        Record &r = toBenchmark[i];
        uint64_t kernelId = r.kernelId.value;
        const KernelEntry &kernel = KernelRegistry::get().getKernel(r.kernelId);

        std::cout << "[" << (i + 1) << "/" << toBenchmark.size() << "][";
        for (uint64_t bidx = 0; bidx < kernel.engines.size(); ++bidx)
        {
            if (bidx > 0)
                std::cout << ",";
            std::cout << toString(kernel.engines[bidx].type);
        }
        std::cout << "] " << kernel.opName << (kernel.opName.empty() ? toString(kernel.opType) : "") << " (0x"
                  << std::hex << kernelId << std::dec << ")"
                  << " est " << std::to_string(r.runTime) << " ms\n";

        for (uint64_t idx = 0; idx < r.inputShapes.size(); ++idx)
        {
            std::cout << "  In  #" << idx << ": dtype=" << toString(r.inputDTypes[idx])
                      << ", shape=" << toString(r.inputShapes[idx]) << ", strides=" << toString(r.inputStrides[idx])
                      << "\n";
        }

        std::cout << "  Out #0: dtype=" << toString(r.outputDType) << ", shape=" << toString(r.outputShape)
                  << ", strides=" << toString(r.outputStrides) << "\n";

        if (listOnly)
        {
            continue;
        }

        try
        {
            std::vector<TensorNode> dummyInputs(r.inputShapes.size());
            for (uint64_t idx = 0; idx < r.inputShapes.size(); ++idx)
            {
                dummyInputs[idx].setShape(r.inputShapes[idx]);
                dummyInputs[idx].strides = r.inputStrides[idx];
                dummyInputs[idx].dtype = r.inputDTypes[idx];
            }

            TensorNode dummyOutput;
            dummyOutput.setShape(r.outputShape);
            dummyOutput.strides = r.outputStrides;
            dummyOutput.dtype = r.outputDType;

            if (!kernel.matches(dummyInputs, dummyOutput, r.output_mem_space, r.input_mem_spaces, r.engines, false,
                                false, false, true))
            {
                std::cerr << "Skipping kernel " << kernel.getName() << " (0x" << std::hex << kernelId
                          << "): record fails matches() validity check." << std::endl;
                continue;
            }

            PreparedKernel pk;
            pk.prepare(kernel, r);

            std::cout << "  Benchmarking..." << std::flush;

            // Warmup
            if (!kernel.is_view)
            {
                pk.updateStorageContext(kernel, r, 0);
                pk.run(kernel);
                pk.synchronize();
            }

            int iters = 8;
            std::vector<float> latencies;
            latencies.reserve(iters);
            for (int it = 0; it < iters; ++it)
            {
                if (!kernel.is_view)
                {
                    pk.updateStorageContext(kernel, r, it + 1);
                }

                auto iterStart = std::chrono::high_resolution_clock::now();
                if (!kernel.is_view)
                {
                    pk.run(kernel);
                }
                pk.synchronize();
                auto iterEnd = std::chrono::high_resolution_clock::now();
                float iterMs = std::chrono::duration<float, std::milli>(iterEnd - iterStart).count();
                latencies.push_back(iterMs);
                if (it != 0)
                {
                    std::cout << ",";
                }
                std::cout << " " << iterMs;
            }

            std::sort(latencies.begin(), latencies.end());

            float runtimeMs = 0.0f;
            if (iters > 0)
            {
                if (iters % 2 == 0)
                {
                    runtimeMs = (latencies[iters / 2 - 1] + latencies[iters / 2]) / 2.0f;
                }
                else
                {
                    runtimeMs = latencies[iters / 2];
                }
            }

            r.runTime = runtimeMs;
            r.buildContextId = BUILD_CONTEXT_ID;
            bw.write(r);
            if (outFile.is_open())
            {
                outFile.flush();
            }

            std::cout << "\n  Benchmarked -> " << runtimeMs << " ms" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to benchmark kernel " << kernelId << ": " << e.what() << std::endl;
        }
    }

    std::cout << "Benchmarking complete." << std::endl;
    return 0;
}