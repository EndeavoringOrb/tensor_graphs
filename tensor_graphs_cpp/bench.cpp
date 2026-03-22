#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <filesystem>
#include <cstring>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "core/types.hpp"
#include "core/memory.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/cost_model.hpp"
#include "core/misc.hpp"

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
            Record r = j.get<Record>();
            json keyObj;
            std::stringstream uid_ss, build_ss;
            uid_ss << "0x" << std::hex << r.kernelUid;
            build_ss << "0x" << std::hex << r.buildContextId;
            keyObj["buildContextId"] = build_ss.str();
            keyObj["hwTag"] = r.hwTag;
            keyObj["inputConstants"] = r.inputConstants;
            keyObj["inputDTypes"] = r.inputDTypes;
            keyObj["outputDTypes"] = r.outputDTypes;
            keyObj["kernelUid"] = uid_ss.str();
            keyObj["inputShapes"] = r.inputShapes;
            keyObj["outputShapes"] = r.outputShapes;
            keyObj["inputStrides"] = r.inputStrides;
            keyObj["outputStrides"] = r.outputStrides;
            keyObj["backends"] = r.backends;
            keyObj["inputBackends"] = r.inputBackends;
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
        Record r = j.get<Record>();
        json keyObj;
        std::stringstream uid_ss, build_ss;
        uid_ss << "0x" << std::hex << r.kernelUid;
        build_ss << "0x" << std::hex << r.buildContextId;
        keyObj["buildContextId"] = build_ss.str();
        keyObj["hwTag"] = r.hwTag;
        keyObj["inputConstants"] = r.inputConstants;
        keyObj["inputDTypes"] = r.inputDTypes;
        keyObj["outputDTypes"] = r.outputDTypes;
        keyObj["kernelUid"] = uid_ss.str();
        keyObj["inputShapes"] = r.inputShapes;
        keyObj["outputShapes"] = r.outputShapes;
        keyObj["inputStrides"] = r.inputStrides;
        keyObj["outputStrides"] = r.outputStrides;
        keyObj["backends"] = r.backends;
        keyObj["inputBackends"] = r.inputBackends;
        std::string key = keyObj.dump();

        if (recordedKeys.find(key) == recordedKeys.end() && seenCalls.find(key) == seenCalls.end())
        {
            seenCalls.insert(key);
            std::string valA = j["hwTag"].get<std::string>();
            std::string valB = j["buildContextId"].get<std::string>();
            bool checkA = valA == HW_TAG;
            bool checkB = valB == BUILD_CONTEXT_ID_STRING;
            bool hasKernel = KernelRegistry::get().hasKernel(r.kernelUid);
            if (checkA && checkB && hasKernel)
            {
                toBenchmark.push_back(j);
            }
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
            // SAFETY CHECK: Ensure the Record from JSON matches the Kernel's expectations
            if (r.inputShapes.size() < (kernel.inplace ? 1 : kernel.numInputs) && !kernel.isReference)
            {
                std::cerr << "Skipping kernel " << kernel.opName << ": Record has "
                          << r.inputShapes.size() << " inputs, kernel requires " << kernel.numInputs << std::endl;
                continue;
            }

            std::vector<std::vector<uint8_t>> inData(r.inputShapes.size());
            std::vector<const void *> inPtrs(r.inputShapes.size(), nullptr);
            std::vector<TensorView> inViews(r.inputShapes.size());

            std::vector<std::vector<uint8_t>> outData(r.outputShapes.size());
            std::vector<void *> outPtrs(r.outputShapes.size(), nullptr);
            std::vector<TensorView> outViews(r.outputShapes.size());

#ifdef USE_CUDA
            std::vector<bool> inIsCuda(r.inputShapes.size(), false);
            bool runCuda = false;
            for (Backend b : kernel.backends)
            {
                if (b == Backend::CUDA)
                    runCuda = true;
            }

            for (size_t idx = 0; idx < r.inputShapes.size(); ++idx)
            {
                if (idx < kernel.inputBackends.size())
                    inIsCuda[idx] = (kernel.inputBackends[idx] == Backend::CUDA);
                else
                    inIsCuda[idx] = runCuda;
            }
            bool isOutputCuda = runCuda;

            if (kernel.opType == OpType::COPY_TO)
            {
                if (runCuda)
                {
                    inIsCuda.assign(inIsCuda.size(), false);
                    isOutputCuda = true;
                }
                else
                {
                    inIsCuda.assign(inIsCuda.size(), true);
                    isOutputCuda = false;
                }
            }
#else
            std::vector<bool> inIsCuda(r.inputShapes.size(), false);
            bool isOutputCuda = false;
#endif

            // RAII wrapper to guarantee device memory is freed even if an exception occurs
            struct CudaCleanup
            {
                std::vector<const void *> &inPtrs;
                std::vector<void *> &outPtrs;
                const std::vector<bool> &inIsCuda;
                bool isOutputCuda;
                ~CudaCleanup()
                {
#ifdef USE_CUDA
                    for (size_t idx = 0; idx < inPtrs.size(); ++idx)
                    {
                        if (idx < inIsCuda.size() && inIsCuda[idx] && inPtrs[idx])
                            cudaFree(const_cast<void *>(inPtrs[idx]));
                    }
                    for (size_t idx = 0; idx < outPtrs.size(); ++idx)
                    {
                        if (isOutputCuda && outPtrs[idx])
                            cudaFree(outPtrs[idx]);
                    }
#endif
                }
            } cleanup{inPtrs, outPtrs, inIsCuda, isOutputCuda};

            for (size_t idx = 0; idx < r.inputShapes.size(); ++idx)
            {
                uint64_t maxIndex = 0;
                for (size_t d = 0; d < r.inputShapes[idx].size(); ++d)
                {
                    if (r.inputShapes[idx][d] > 0)
                    {
                        maxIndex += (r.inputShapes[idx][d] - 1) * r.inputStrides[idx][d];
                    }
                }
                uint64_t elements = r.inputShapes[idx].empty() ? 1 : maxIndex + 1;

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
                        if (kernel.opType == OpType::PERMUTE || kernel.opName.find("Permute") != std::string::npos) // TODO: make the check based on reference graph instead of name
                        {
                            if (idx == 1 && r.inputShapes.size() > 0 && r.outputShapes.size() > 0 && 
                                r.inputShapes[0].size() == r.outputShapes[0].size() && elements == r.inputShapes[0].size()) 
                            {
                                std::vector<bool> used(elements, false);
                                for (size_t k = 0; k < elements; ++k) {
                                    size_t found_d = k; // default fallback
                                    for(size_t d = 0; d < elements; ++d) {
                                        if(!used[d] && r.inputShapes[0][d] == r.outputShapes[0][k]) {
                                            found_d = d;
                                            break;
                                        }
                                    }
                                    used[found_d] = true;
                                    iptr[k] = found_d;
                                }
                            }
                            else 
                            {
                                for (size_t k = 0; k < elements; ++k)
                                    iptr[k] = k;
                            }
                        }
                        else
                        {
                            for (size_t k = 0; k < elements; ++k)
                                iptr[k] = 1;
                        }
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

                if (inIsCuda[idx])
                {
#ifdef USE_CUDA
                    void *d_ptr = nullptr;
                    cudaError_t err = cudaMalloc(&d_ptr, bytes);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
                    }
                    if (bytes > 0 && inData[idx].data())
                    {
                        err = cudaMemcpy(d_ptr, inData[idx].data(), bytes, cudaMemcpyHostToDevice);
                        if (err != cudaSuccess)
                        {
                            cudaFree(d_ptr);
                            throw std::runtime_error("cudaMemcpy failed: " + std::string(cudaGetErrorString(err)));
                        }
                    }
                    inPtrs[idx] = d_ptr;
#endif
                }
                else
                {
                    inPtrs[idx] = inData[idx].data();
                }

                inViews[idx].shape = r.inputShapes[idx];
                inViews[idx].strides = r.inputStrides[idx];
                inViews[idx].baseOffset = 0;
                inViews[idx].dtype = r.inputDTypes[idx];
            }

            for (size_t idx = 0; idx < r.outputShapes.size(); ++idx)
            {
                uint64_t maxIndex = 0;
                for (size_t d = 0; d < r.outputShapes[idx].size(); ++d)
                {
                    if (r.outputShapes[idx][d] > 0)
                    {
                        maxIndex += (r.outputShapes[idx][d] - 1) * r.outputStrides[idx][d];
                    }
                }
                uint64_t elements = r.outputShapes[idx].empty() ? 1 : maxIndex + 1;

                if (elements == 0)
                    elements = 1;
                uint64_t bytes = elements * getDTypeSize(r.outputDTypes[idx]);
                outData[idx].resize(bytes);

                if (isOutputCuda)
                {
#ifdef USE_CUDA
                    void *d_ptr = nullptr;
                    cudaError_t err = cudaMalloc(&d_ptr, bytes);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
                    }
                    outPtrs[idx] = d_ptr;
#endif
                }
                else
                {
                    outPtrs[idx] = outData[idx].data();
                }

                outViews[idx].shape = r.outputShapes[idx];
                outViews[idx].strides = r.outputStrides[idx];
                outViews[idx].baseOffset = 0;
                outViews[idx].dtype = r.outputDTypes[idx];
            }

            std::cout << "[" << (i + 1) << "/" << toBenchmark.size() << "][";
            for (size_t bidx = 0; bidx < kernel.backends.size(); ++bidx)
            {
                if (bidx > 0)
                    std::cout << ",";
                std::cout << toString(kernel.backends[bidx]);
            }
            std::cout << "] "
                      << kernel.opName << (kernel.opName.empty() ? toString(kernel.opType) : "")
                      << " (0x" << std::hex << kernelUid << std::dec << ")"
                      << ", Out DType: " << toString(outViews[0].dtype) // Added DType
                      << ", Out Shape: " << toString(outViews[0].shape)
                      << ", Out Strides: " << toString(outViews[0].strides) // Added Strides
                      << std::flush;

            // Warmup
            kernel.run(inPtrs, outPtrs, inViews, outViews);

            int iters = 1; // TODO: make this an arg, default to 15
            auto start = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it)
            {
                kernel.run(inPtrs, outPtrs, inViews, outViews);
            }
#ifdef USE_CUDA
            bool anyInputCuda = std::any_of(inIsCuda.begin(), inIsCuda.end(), [](bool b){ return b; });
            if (anyInputCuda || isOutputCuda)
            {
                cudaError_t err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    // Critical: if a kernel crashed, we must reset the error state 
                    // or subsequent cudaMalloc/cudaMemcpy calls will fail.
                    // However, illegal memory access is often unrecoverable without context reset.
                    throw std::runtime_error("CUDA Synchronization failed: " + std::string(cudaGetErrorString(err)));
                }
            }
#endif
            auto end = std::chrono::high_resolution_clock::now();

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