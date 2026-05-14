#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <chrono>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include <cmath>

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

int main(int argc, char* argv[])
{
    int skipCount = 0;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if ((arg == "-s" || arg == "--skip") && i + 1 < argc)
        {
            skipCount = std::atoi(argv[++i]);
        }
    }

    std::filesystem::create_directories("benchmarks");

    std::string callsPath = "benchmarks/calls.bin";
    std::string recordsPath = "benchmarks/records.bin";

    // 1. Initialize CostModel and load existing records to provide estimates for sorting
    CostModel costModel;
    costModel.load(recordsPath);

    // Build a registry of already benchmarked kernels to skip redundancy
    std::unordered_set<std::string> recordedKeys;
    std::ifstream recordsFile(recordsPath, std::ios::binary);
    if (recordsFile.is_open())
    {
        BinaryReader br(recordsFile);
        while (recordsFile.peek() != EOF)
        {
            Record r;
            br.read(r);
            r.runTime = 0.0f; // Normalize for skip comparison
            recordedKeys.insert(serializeToString(r));
        }
    }

    std::ifstream callsFile(callsPath, std::ios::binary);
    if (!callsFile.is_open())
    {
        std::cerr << "No calls file found at " << callsPath << ". Enable TENSOR_GRAPHS_LOG_COST_CALLS and run an inference pass first." << std::endl;
        return 0;
    }

    std::vector<Record> toBenchmark;
    std::unordered_set<std::string> seenCalls;

    BinaryReader br(callsFile);
    while (callsFile.peek() != EOF)
    {
        Record r;
        br.read(r);

        r.runTime = 0.0f;
        r.buildContextId = BUILD_CONTEXT_ID;
        std::string key = serializeToString(r);

        if (recordedKeys.find(key) == recordedKeys.end() && seenCalls.find(key) == seenCalls.end())
        {
            seenCalls.insert(key);
            if (r.hwTag == HW_TAG && KernelRegistry::get().hasKernel(r.kernelUid))
            {
                toBenchmark.push_back(std::move(r));
            }
        }
    }

    if (toBenchmark.empty())
    {
        std::cout << "All calls already benchmarked or no new kernels to test." << std::endl;
        return 0;
    }

    for (uint32_t i = 0; i < toBenchmark.size(); i++)
    {
        Record &r = toBenchmark[i];
        float cost = costModel.estimateCost(
            r.kernelUid,
            r.outputShapes[0],
            r.outputStrides[0],
            r.outputDTypes[0],
            r.inputShapes,
            r.inputStrides,
            r.inputDTypes,
            r.inputConstants);
        r.runTime = std::isinf(cost) ? -1.0f : cost;
    }

    // 2. Sort kernels by cost (cheapest first).
    // Fallback to element count for kernels with no previous data (inf cost).
    std::stable_sort(toBenchmark.begin(), toBenchmark.end(), [&](const Record &ra, const Record &rb)
                     {
                          float costA = ra.runTime;
                          float costB = rb.runTime;

                          if (std::abs(costA - costB) < 1e-7) {
                             // Tie-break: Prioritize optimized kernels over reference kernels
                             bool isRefA = KernelRegistry::get().getKernel(ra.kernelUid).isReference;
                             bool isRefB = KernelRegistry::get().getKernel(rb.kernelUid).isReference;
                             if (isRefA != isRefB) return !isRefA;
                             uint64_t sizeA = 1;
                             for (size_t i = 0; i < ra.outputShapes.size(); i++) {
                                 for (uint32_t dim : ra.outputShapes[i]) {
                                     sizeA *= dim;
                                 }
                             }
                             uint64_t sizeB = 1;
                             for (size_t i = 0; i < rb.outputShapes.size(); i++) {
                                 for (uint32_t dim : rb.outputShapes[i]) {
                                     sizeB *= dim;
                                 }
                             }
                             
                             return sizeA < sizeB;
                          }

                          return costA < costB; });

    std::ofstream outFile(recordsPath, std::ios::app | std::ios::binary);
    BinaryWriter bw(outFile);
    size_t startIdx = (skipCount > (int)toBenchmark.size()) ? toBenchmark.size() : (size_t)std::max(0, skipCount);
    if (startIdx > 0)
    {
        std::cout << "Skipping the first " << startIdx << " kernels..." << std::endl;
    }
    std::cout << "Benchmarking " << toBenchmark.size() - startIdx << " configurations..." << std::endl;

    for (size_t i = startIdx; i < toBenchmark.size(); ++i)
    {
        Record &r = toBenchmark[i];
        uint64_t kernelUid = r.kernelUid;

        try
        {
            const KernelEntry &kernel = KernelRegistry::get().getKernel(kernelUid);

            // Build dummy nodes for validation against centralized matching logic
            std::vector<TensorNode> dummyInputs(r.inputShapes.size());
            for (size_t idx = 0; idx < r.inputShapes.size(); ++idx)
            {
                dummyInputs[idx].setShape(r.inputShapes[idx]);
                dummyInputs[idx].strides = r.inputStrides[idx];
                dummyInputs[idx].dtype = r.inputDTypes[idx];

                Backend b = Backend::CPU;
                size_t ruleIdx = idx;
                if (kernel.isVariadic)
                {
                    ruleIdx = (idx == r.inputShapes.size() - 1) ? (kernel.inputBackends.empty() ? 0 : kernel.inputBackends.size() - 1) : 0;
                }

                if (!r.inputBackends.empty() && ruleIdx < r.inputBackends.size() && !r.inputBackends[ruleIdx].empty())
                    b = r.inputBackends[ruleIdx][0];
                dummyInputs[idx].backend = b;
            }

            TensorNode dummyOutput;
            if (!r.outputShapes.empty())
            {
                dummyOutput.setShape(r.outputShapes[0]);
                dummyOutput.strides = r.outputStrides[0];
                dummyOutput.dtype = r.outputDTypes[0];
                dummyOutput.backend = r.backends.empty() ? Backend::CPU : r.backends[0];
            }

            // Run central match that safely ignores OOB array evaluations
            if (!kernel.matches(dummyInputs, dummyOutput))
            {
                std::cerr << "Skipping kernel " << kernel.getName() << " (0x" << std::hex << kernelUid << "): record fails matches() validity check." << std::endl;
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
                size_t ruleIdx = idx;
                if (kernel.isVariadic)
                {
                    ruleIdx = (idx == r.inputShapes.size() - 1) ? (kernel.inputBackends.empty() ? 0 : kernel.inputBackends.size() - 1) : 0;
                }

                if (ruleIdx < kernel.inputBackends.size())
                {
                    bool hasCuda = false;
                    for (Backend b : kernel.inputBackends[ruleIdx])
                    {
                        if (b == Backend::CUDA)
                            hasCuda = true;
                    }
                    inIsCuda[idx] = hasCuda;
                }
                else
                {
                    inIsCuda[idx] = runCuda;
                }
            }
            bool isOutputCuda = runCuda;
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
                        if (kernel.opType == OpType::PERMUTE || kernel.opName.find("Permute") != std::string::npos)
                        {
                            if (idx == 1 && r.inputShapes.size() > 0 && r.outputShapes.size() > 0 &&
                                r.inputShapes[0].size() == r.outputShapes[0].size() && elements == r.inputShapes[0].size())
                            {
                                std::vector<bool> used(elements, false);
                                for (size_t k = 0; k < elements; ++k)
                                {
                                    size_t found_d = k; // default fallback
                                    for (size_t d = 0; d < elements; ++d)
                                    {
                                        if (!used[d] && r.inputShapes[0][d] == r.outputShapes[0][k])
                                        {
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
                        else if (kernel.opType == OpType::CONCAT || kernel.opName.find("Concat") != std::string::npos)
                        {
                            if (idx == r.inputShapes.size() - 1)
                            {
                                int32_t concat_axis = -1;
                                if (!r.inputShapes.empty() && !r.outputShapes.empty())
                                {
                                    for (size_t d = 0; d < r.outputShapes[0].size(); ++d)
                                    {
                                        if (r.outputShapes[0][d] != r.inputShapes[0][d])
                                        {
                                            concat_axis = (int32_t)d;
                                            break;
                                        }
                                    }
                                }
                                if (concat_axis == -1)
                                    concat_axis = 0;
                                for (size_t k = 0; k < elements; ++k)
                                    iptr[k] = concat_axis;
                            }
                            else
                            {
                                for (size_t k = 0; k < elements; ++k)
                                    iptr[k] = 1;
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

                inViews[idx].setShape(r.inputShapes[idx]);
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

                outViews[idx].setShape(r.outputShapes[idx]);
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
            std::cout << "] " << kernel.opName << (kernel.opName.empty() ? toString(kernel.opType) : "")
                      << " (0x" << std::hex << kernelUid << std::dec << ")"
                      << " est " << std::to_string(r.runTime) << " ms\n";

            // Print All Inputs
            for (size_t idx = 0; idx < inViews.size(); ++idx)
            {
                std::cout << "  In  #" << idx << ": dtype=" << toString(inViews[idx].dtype)
                          << ", shape=" << toString(inViews[idx].getShape())
                          << ", strides=" << toString(inViews[idx].strides) << "\n";
            }

            // Print All Outputs
            for (size_t idx = 0; idx < outViews.size(); ++idx)
            {
                std::cout << "  Out #" << idx << ": dtype=" << toString(outViews[idx].dtype)
                          << ", shape=" << toString(outViews[idx].getShape())
                          << ", strides=" << toString(outViews[idx].strides) << "\n";
            }
            std::cout << "  Benchmarking..." << std::flush;

            // Warmup
            if (!kernel.isView)
            {
                kernel.run(inPtrs, outPtrs, inViews, outViews);
#ifdef USE_CUDA
                cudaDeviceSynchronize();
#endif
            }

            int iters = 8;
            std::vector<float> latencies;
            latencies.reserve(iters);
            for (int it = 0; it < iters; ++it)
            {
                auto iterStart = std::chrono::high_resolution_clock::now();
                if (!kernel.isView)
                {
                    kernel.run(inPtrs, outPtrs, inViews, outViews);
                }
#ifdef USE_CUDA
                bool anyInputCuda = std::any_of(inIsCuda.begin(), inIsCuda.end(), [](bool b)
                                                { return b; });
                if (anyInputCuda || isOutputCuda)
                {
                    cudaError_t err = cudaDeviceSynchronize();
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error("CUDA Synchronization failed: " + std::string(cudaGetErrorString(err)));
                    }
                }
#endif
                auto iterEnd = std::chrono::high_resolution_clock::now();
                float iterMs = std::chrono::duration<float, std::milli>(iterEnd - iterStart).count();
                latencies.push_back(iterMs);
            }
            // Calculate Median
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