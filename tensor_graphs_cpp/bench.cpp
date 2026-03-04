#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <filesystem>
#include <cstring>

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/cost_model.hpp"

// Explicitly register all kernels
#include "kernels/fused/tanh/F32_1D.hpp"
#include "kernels/reference/add/F32_ND.hpp"
#include "kernels/reference/arange/I32_ND.hpp"
#include "kernels/reference/cast/BF16_F32_ND.hpp"
#include "kernels/reference/cast/I32_F32_ND.hpp"
#include "kernels/reference/concat/F32_ND.hpp"
#include "kernels/reference/cos/F32_ND.hpp"
#include "kernels/reference/div/F32_ND.hpp"
#include "kernels/reference/dot/F32_3D.hpp"
#include "kernels/reference/fill/F32_ND.hpp"
#include "kernels/reference/gather/F32_I32_ND.hpp"
#include "kernels/reference/max/F32_ND.hpp"
#include "kernels/reference/mul/F32_ND.hpp"
#include "kernels/reference/neg/F32_ND.hpp"
#include "kernels/reference/permute/F32_ND.hpp"
#include "kernels/reference/pow/F32_ND.hpp"
#include "kernels/reference/repeat/F32_ND.hpp"
#include "kernels/reference/reshape/ND.hpp"
#include "kernels/reference/sin/F32_ND.hpp"
#include "kernels/reference/slice/F32_ND.hpp"
#include "kernels/reference/sum/F32_ND.hpp"
#include "kernels/reference/triu/F32_ND.hpp"

using json = nlohmann::json;

int main()
{
    std::filesystem::create_directories("benchmarks");

    std::string callsPath = "benchmarks/calls.jsonl";
    std::string recordsPath = "benchmarks/records.jsonl";

    // Build a registry of already benchmarked kernels to skip redundancy
    std::unordered_set<std::string> recordedKeys;
    std::ifstream recordsFile(recordsPath);
    if (recordsFile.is_open())
    {
        std::string line;
        while (std::getline(recordsFile, line))
        {
            if (line.empty())
                continue;
            try
            {
                auto j = json::parse(line);
                json keyObj;
                keyObj["kernelId"] = j["kernelId"];
                keyObj["inputShapes"] = j["inputShapes"];
                keyObj["outputShapes"] = j["outputShapes"];
                recordedKeys.insert(keyObj.dump());
            }
            catch (...)
            {
            }
        }
    }

    std::ifstream callsFile(callsPath);
    if (!callsFile.is_open())
    {
        std::cerr << "No calls file found at " << callsPath << ". Enable TENSOR_GRAPHS_LOG_COST_CALLS and run an inference pass first." << std::endl;
        return 0;
    }

    std::vector<json> toBenchmark;
    std::unordered_set<std::string> seenCalls;

    std::string line;
    while (std::getline(callsFile, line))
    {
        if (line.empty())
            continue;
        try
        {
            auto j = json::parse(line);
            json keyObj;
            keyObj["kernelId"] = j["kernelId"];
            keyObj["inputShapes"] = j["inputShapes"];
            keyObj["outputShapes"] = j["outputShapes"];
            std::string key = keyObj.dump();

            if (recordedKeys.find(key) == recordedKeys.end() && seenCalls.find(key) == seenCalls.end())
            {
                seenCalls.insert(key);
                toBenchmark.push_back(j);
            }
        }
        catch (...)
        {
        }
    }

    if (toBenchmark.empty())
    {
        std::cout << "All calls already benchmarked or no new kernels to test." << std::endl;
        return 0;
    }

    std::ofstream outFile(recordsPath, std::ios::app);
    std::cout << "Benchmarking " << toBenchmark.size() << " configurations..." << std::endl;

    for (size_t i = 0; i < toBenchmark.size(); ++i)
    {
        auto &call = toBenchmark[i];
        uint32_t kernelId = call["kernelId"].get<uint32_t>();

        Record r = call.get<Record>();

        try
        {
            const KernelEntry &kernel = KernelRegistry::get().getKernel(kernelId);

            std::vector<std::vector<uint8_t>> inData(r.inputShapes.size());
            std::vector<const void *> inPtrs(r.inputShapes.size());
            std::vector<TensorView> inViews(r.inputShapes.size());

            for (size_t idx = 0; idx < r.inputShapes.size(); ++idx)
            {
                uint64_t elements = countElements(r.inputShapes[idx]);
                if (elements == 0)
                    elements = 1;
                uint64_t bytes = elements * getDTypeSize(r.inputDTypes[idx]);
                inData[idx].resize(bytes);

                // Prioritize providing the explicit constants if the planner saved them,
                // to avoid index-out-of-bounds or zero-division runtime crashes inside reference kernels
                if (idx < r.inputConstants.size() && !r.inputConstants[idx].empty() && r.inputConstants[idx].size() == bytes)
                {
                    std::memcpy(inData[idx].data(), r.inputConstants[idx].data(), bytes);
                }
                else
                {
                    // Pre-fill everything with safely predictable "1" values otherwise
                    if (r.inputDTypes[idx] == DType::FLOAT32)
                    {
                        float *fptr = reinterpret_cast<float *>(inData[idx].data());
                        for (size_t k = 0; k < elements; ++k)
                            fptr[k] = 1.0f;
                    }
                    else if (r.inputDTypes[idx] == DType::INT32)
                    {
                        int32_t *iptr = reinterpret_cast<int32_t *>(inData[idx].data());
                        for (size_t k = 0; k < elements; ++k)
                            iptr[k] = 1;
                    }
                    else if (r.inputDTypes[idx] == DType::BF16)
                    {
                        uint16_t *bptr = reinterpret_cast<uint16_t *>(inData[idx].data());
                        for (size_t k = 0; k < elements; ++k)
                            bptr[k] = 0x3F80;
                    }
                    else
                    {
                        std::memset(inData[idx].data(), 1, bytes);
                    }
                }

                inPtrs[idx] = inData[idx].data();

                inViews[idx].shape = r.inputShapes[idx];
                inViews[idx].strides = TensorView::calcContiguousStrides(r.inputShapes[idx]);
                inViews[idx].baseOffset = 0;
                inViews[idx].dtype = r.inputDTypes[idx];
            }

            std::vector<std::vector<uint8_t>> outData(r.outputShapes.size());
            std::vector<void *> outPtrs(r.outputShapes.size());
            std::vector<TensorView> outViews(r.outputShapes.size());

            for (size_t idx = 0; idx < r.outputShapes.size(); ++idx)
            {
                uint64_t elements = countElements(r.outputShapes[idx]);
                if (elements == 0)
                    elements = 1;
                uint64_t bytes = elements * getDTypeSize(r.outputDTypes[idx]);
                outData[idx].resize(bytes);

                outPtrs[idx] = outData[idx].data();

                outViews[idx].shape = r.outputShapes[idx];
                outViews[idx].strides = TensorView::calcContiguousStrides(r.outputShapes[idx]);
                outViews[idx].baseOffset = 0;
                outViews[idx].dtype = r.outputDTypes[idx];
            }

            // Warmup
            kernel.run(inPtrs, outPtrs, inViews, outViews);

            int iters = 15;
            auto start = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it)
            {
                kernel.run(inPtrs, outPtrs, inViews, outViews);
            }
            auto end = std::chrono::high_resolution_clock::now();

            float runtimeMs = std::chrono::duration<float, std::milli>(end - start).count() / iters;

            call["runTime"] = runtimeMs;
            outFile << call.dump() << "\n";
            outFile.flush();

            std::cout << "[" << (i + 1) << "/" << toBenchmark.size() << "] "
                      << kernel.opName << (kernel.opName.empty() ? toString(kernel.opType) : "")
                      << " (Kernel " << kernelId << ") -> "
                      << runtimeMs << " ms" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to benchmark kernel " << kernelId << ": " << e.what() << std::endl;
        }
    }

    std::cout << "Benchmarking complete." << std::endl;
    return 0;
}