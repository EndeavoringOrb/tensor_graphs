#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <filesystem>
#include <cstring>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/cost_model.hpp"

#include "generated/kernels_all.gen.hpp"
#include "generated/build_context.gen.hpp"

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

            auto j = json::parse(line);
            json keyObj;
            keyObj["buildContextId"] = j["buildContextId"];
            keyObj["hwTag"] = j["hwTag"];
            keyObj["inputConstants"] = j["inputConstants"];
            keyObj["inputDTypes"] = j["inputDTypes"];
            keyObj["outputDTypes"] = j["outputDTypes"];
            keyObj["kernelUid"] = j["kernelUid"];
            keyObj["inputShapes"] = j["inputShapes"];
            keyObj["outputShapes"] = j["outputShapes"];
            std::string key = keyObj.dump();
            recordedKeys.insert(key);
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
    std::ostringstream ss;
    ss << "0x" << std::hex << BUILD_CONTEXT_ID;
    std::string BUILD_CONTEXT_ID_STRING = ss.str();

    std::string line;
    while (std::getline(callsFile, line))
    {
        if (line.empty())
            continue;
        auto j = json::parse(line);
        json keyObj;
        keyObj["buildContextId"] = j["buildContextId"];
        keyObj["hwTag"] = j["hwTag"];
        keyObj["inputConstants"] = j["inputConstants"];
        keyObj["inputDTypes"] = j["inputDTypes"];
        keyObj["outputDTypes"] = j["outputDTypes"];
        keyObj["kernelUid"] = j["kernelUid"];
        keyObj["inputShapes"] = j["inputShapes"];
        keyObj["outputShapes"] = j["outputShapes"];
        std::string key = keyObj.dump();

        if (recordedKeys.find(key) == recordedKeys.end() && seenCalls.find(key) == seenCalls.end())
        {
            seenCalls.insert(key);
            std::string valA = j["hwTag"].get<std::string>();
            std::string valB = j["buildContextId"].get<std::string>();
            bool checkA = valA == HW_TAG;
            bool checkB = valB == BUILD_CONTEXT_ID_STRING;
            if (checkA && checkB)
            {
                toBenchmark.push_back(j);
            }
        }
        else
        {
            int a = 5;
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
        uint64_t kernelUid = std::stoull(call["kernelUid"].get<std::string>(), nullptr, 16);

        Record r = call.get<Record>();

        try
        {
            const KernelEntry &kernel = KernelRegistry::get().getKernel(kernelUid);

            std::vector<std::vector<uint8_t>> inData(r.inputShapes.size());
            std::vector<const void *> inPtrs(r.inputShapes.size());
            std::vector<TensorView> inViews(r.inputShapes.size());

            // TODO: if kernel.backend not in available backends, skip benchmarking it
#ifdef USE_CUDA
            bool isInputCuda = kernel.backend == Backend::CUDA;
            bool isOutputCuda = kernel.backend == Backend::CUDA;

            if (kernel.opType == OpType::COPY_TO)
            {
                if (kernel.backend == Backend::CUDA)
                {
                    isInputCuda = false;
                    isOutputCuda = true;
                }
                else
                {
                    isInputCuda = true;
                    isOutputCuda = false;
                }
            }
#else
            bool isInputCuda = false;
            bool isOutputCuda = false;
#endif

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

                if (isInputCuda)
                {
#ifdef USE_CUDA
                    void *d_ptr;
                    cudaMalloc(&d_ptr, bytes);
                    cudaMemcpy(d_ptr, inData[idx].data(), bytes, cudaMemcpyHostToDevice);
                    inPtrs[idx] = d_ptr;
#endif
                }
                else
                {
                    inPtrs[idx] = inData[idx].data();
                }

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

                if (isOutputCuda)
                {
#ifdef USE_CUDA
                    void *d_ptr;
                    cudaMalloc(&d_ptr, bytes);
                    outPtrs[idx] = d_ptr;
#endif
                }
                else
                {
                    outPtrs[idx] = outData[idx].data();
                }

                outViews[idx].shape = r.outputShapes[idx];
                outViews[idx].strides = TensorView::calcContiguousStrides(r.outputShapes[idx]);
                outViews[idx].baseOffset = 0;
                outViews[idx].dtype = r.outputDTypes[idx];
            }

            std::cout << "[" << (i + 1) << "/" << toBenchmark.size() << "] "
                      << kernel.opName << (kernel.opName.empty() ? toString(kernel.opType) : "")
                      << " (Kernel 0x" << std::hex << kernelUid << std::dec << ")" << std::flush;

            // Warmup
            kernel.run(inPtrs, outPtrs, inViews, outViews);

            int iters = 1; // TODO: make this an arg, default to 15
            auto start = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it)
            {
                kernel.run(inPtrs, outPtrs, inViews, outViews);
            }
#ifdef USE_CUDA
            if (isInputCuda || isOutputCuda)
            {
                cudaDeviceSynchronize();
            }
#endif
            auto end = std::chrono::high_resolution_clock::now();

#ifdef USE_CUDA
            for (size_t idx = 0; idx < inPtrs.size(); ++idx)
            {
                if (isInputCuda)
                    cudaFree(const_cast<void *>(inPtrs[idx]));
            }
            for (size_t idx = 0; idx < outPtrs.size(); ++idx)
            {
                if (isOutputCuda)
                    cudaFree(outPtrs[idx]);
            }
#endif

            float runtimeMs = std::chrono::duration<float, std::milli>(end - start).count() / iters;

            call["runTime"] = runtimeMs;
            outFile << call.dump() << "\n";
            outFile.flush();

            std::cout << " -> " << runtimeMs << " ms" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to benchmark kernel " << kernelUid << ": " << e.what() << std::endl;
        }
    }

    std::cout << "Benchmarking complete." << std::endl;
    return 0;
}