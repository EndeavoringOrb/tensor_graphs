// tensor_graphs_cpp/bench.cpp
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

#ifdef TG_OS_WINDOWS
#include <io.h>
#include <fcntl.h>
#include <share.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif

// =============================================================================
// Extensible Unified RAII Device Buffer
// =============================================================================
struct BenchBuffer
{
    Backend backend = Backend::CPU;
    uint64_t bytes = 0;
    std::vector<uint8_t> hostData;
    void *devicePtr = nullptr;
    cl_mem clMem = nullptr;

    BenchBuffer() = default;
    ~BenchBuffer()
    {
        free();
    }

    // Disable copy
    BenchBuffer(const BenchBuffer &) = delete;
    BenchBuffer &operator=(const BenchBuffer &) = delete;

    // Support move
    BenchBuffer(BenchBuffer &&o) noexcept
    {
        *this = std::move(o);
    }

    BenchBuffer &operator=(BenchBuffer &&o) noexcept
    {
        if (this != &o)
        {
            free();
            backend = o.backend;
            bytes = o.bytes;
            hostData = std::move(o.hostData);
            devicePtr = o.devicePtr;
            clMem = o.clMem;
            o.devicePtr = nullptr;
            o.clMem = nullptr;
            o.bytes = 0;
        }
        return *this;
    }

    void allocate(Backend b, uint64_t size)
    {
        free();
        backend = b;
        bytes = size == 0 ? 1 : size;

        hostData.resize(bytes, 0);

        if (backend == Backend::CUDA)
        {
#ifdef USE_CUDA
            cudaError_t err = cudaMalloc(&devicePtr, bytes);
            if (err != cudaSuccess)
            {
                Error::throw_err("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
            }
#else
            Error::throw_err("CUDA backend requested but USE_CUDA is not defined.");
#endif
        }
        else if (backend == Backend::OPENCL)
        {
            OpenCLState::get().init();
            cl_context ctx = OpenCLState::get().context;
            if (!ctx)
            {
                Error::throw_err("OpenCL context not initialized.");
            }
            devicePtr = hostData.data();
            cl_int err;
            clMem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, bytes, devicePtr, &err);
            if (err != CL_SUCCESS || !clMem)
            {
                Error::throw_err("clCreateBuffer failed to allocate memory of size " + std::to_string(bytes) + ". Error: " + std::to_string(err));
            }
        }
        else
        {
            devicePtr = hostData.data();
        }
    }

    void upload()
    {
        if (backend == Backend::CUDA)
        {
#ifdef USE_CUDA
            cudaError_t err = cudaMemcpy(devicePtr, hostData.data(), bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                Error::throw_err("cudaMemcpy HostToDevice failed: " + std::string(cudaGetErrorString(err)));
            }
#endif
        }
        else if (backend == Backend::OPENCL)
        {
            if (devicePtr != hostData.data() && devicePtr && !hostData.empty())
            {
                std::memcpy(devicePtr, hostData.data(), bytes);
            }
        }
    }

    void download()
    {
        if (backend == Backend::CUDA)
        {
#ifdef USE_CUDA
            cudaError_t err = cudaMemcpy(hostData.data(), devicePtr, bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                Error::throw_err("cudaMemcpy DeviceToHost failed: " + std::string(cudaGetErrorString(err)));
            }
#endif
        }
        else if (backend == Backend::OPENCL)
        {
            if (devicePtr != hostData.data() && devicePtr && !hostData.empty())
            {
                std::memcpy(hostData.data(), devicePtr, bytes);
            }
        }
    }

    void free()
    {
        if (devicePtr)
        {
            if (backend == Backend::CUDA)
            {
#ifdef USE_CUDA
                cudaFree(devicePtr);
#endif
            }
            else if (backend == Backend::OPENCL)
            {
                if (clMem)
                {
                    clReleaseMemObject(clMem);
                    clMem = nullptr;
                }
            }
            devicePtr = nullptr;
        }
        bytes = 0;
        hostData.clear();
    }

    const void *getReadPtr() const
    {
        return (backend == Backend::STORAGE) ? nullptr : devicePtr;
    }

    void *getWritePtr()
    {
        return devicePtr;
    }
};

// =============================================================================
// Helper Functions
// =============================================================================
struct StorageFiles
{
    std::vector<std::string> paths;
    std::vector<int> fds;

    StorageFiles() = default;

    StorageFiles(const StorageFiles &) = delete;
    StorageFiles &operator=(const StorageFiles &) = delete;

    StorageFiles(StorageFiles &&other) noexcept
        : paths(std::move(other.paths)), fds(std::move(other.fds)) {}

    StorageFiles &operator=(StorageFiles &&other) noexcept
    {
        if (this != &other)
        {
            clear();
            paths = std::move(other.paths);
            fds = std::move(other.fds);
        }
        return *this;
    }

    void clear()
    {
        for (int fd : fds)
        {
            if (fd >= 0)
            {
#ifdef TG_OS_WINDOWS
                _close(fd);
#else
                close(fd);
#endif
            }
        }
        fds.clear();
        for (const auto &path : paths)
        {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
        paths.clear();
    }

    ~StorageFiles()
    {
        clear();
    }
};

StorageFiles createStorageInputs(const Record &r, const KernelEntry &kernel, int runIdx)
{
    StorageFiles sf;
    std::vector<char> dummyBuf(1024 * 1024, 0);

    for (size_t idx = 0; idx < r.inputShapes.size(); ++idx)
    {
        size_t ruleIdx = idx;
        if (kernel.isVariadic)
        {
            ruleIdx = (idx == r.inputShapes.size() - 1) ? (kernel.inputBackends.empty() ? 0 : kernel.inputBackends.size() - 1) : 0;
        }
        Backend b = Backend::CPU;
        if (!r.inputBackends.empty() && ruleIdx < r.inputBackends.size() && !r.inputBackends[ruleIdx].empty())
            b = r.inputBackends[ruleIdx][0];

        if (b == Backend::STORAGE)
        {
            uint64_t elements = countElements(r.inputShapes[idx]);
            uint64_t bytes = elements * getDTypeSize(r.inputDTypes[idx]);
            if (bytes == 0)
            {
                Error::throw_err("[createStorageInputs] got 0 bytes for file size");
            }

            std::string path = "benchmarks/dummy_storage_" + std::to_string(sf.fds.size()) + "_" + std::to_string(r.kernelUid) + "_run_" + std::to_string(runIdx) + ".bin";
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out.is_open())
            {
                std::cerr << "Failed to create dummy storage file: " << path << std::endl;
                continue;
            }

            uint64_t written = 0;
            while (written < bytes)
            {
                uint64_t toWrite = std::min<uint64_t>(dummyBuf.size(), bytes - written);
                out.write(dummyBuf.data(), toWrite);
                written += toWrite;
            }
            out.close();
            sf.paths.push_back(path);

            int fd = -1;
#ifdef TG_OS_WINDOWS
            _wsopen_s(&fd, std::filesystem::path(path).c_str(), _O_RDONLY | _O_BINARY, _SH_DENYNO, 0);
#else
            fd = open(path.c_str(), O_RDONLY);
#endif
            if (fd < 0)
            {
                std::cerr << "Failed to open dummy storage file for reading: " << path << std::endl;
            }
            sf.fds.push_back(fd);
        }
    }
    return sf;
}

void synchronizeBackend(Backend backend)
{
    if (backend == Backend::CUDA)
    {
#ifdef USE_CUDA
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            Error::throw_err("CUDA Synchronization failed: " + std::string(cudaGetErrorString(err)));
        }
#endif
    }
    else if (backend == Backend::OPENCL)
    {
        clFinish(OpenCLState::get().queue);
    }
}

// =============================================================================
// Entry Point
// =============================================================================
int main(int argc, char *argv[])
{
    int skipCount = 0;
    std::string targetKernel = "";
    bool listOnly = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if ((arg == "-s" || arg == "--skip") && i + 1 < argc)
        {
            skipCount = std::atoi(argv[++i]);
        }
        else if (arg == "-l" || arg == "--list")
        {
            listOnly = true;
        }
        else if (targetKernel.empty() && arg[0] != '-')
        {
            targetKernel = arg;
        }
    }

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
                const auto &kernel = KernelRegistry::get().getKernel(r.kernelUid);
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
        float cost = costModel.estimateCost(
            r.kernelUid, r.outputShapes[0], r.outputStrides[0], r.outputDTypes[0],
            r.inputShapes, r.inputStrides, r.inputDTypes, r.inputConstants);
        r.runTime = std::isinf(cost) ? -1.0f : cost;
    }

    std::stable_sort(toBenchmark.begin(), toBenchmark.end(), [&](const Record &ra, const Record &rb)
                     {
        float costA = ra.runTime;
        float costB = rb.runTime;

        if (std::abs(costA - costB) < 1e-7) {
            bool isRefA = KernelRegistry::get().getKernel(ra.kernelUid).isReference;
            bool isRefB = KernelRegistry::get().getKernel(rb.kernelUid).isReference;
            if (isRefA != isRefB) return !isRefA;

            auto getVolume = [](const Record& r) {
                uint64_t v = 1;
                for (const auto& shape : r.outputShapes)
                    for (uint32_t d : shape) v *= d;
                return v;
            };
            return getVolume(ra) < getVolume(rb);
        }
        return costA < costB; });

    size_t startIdx = (skipCount > (int)toBenchmark.size()) ? toBenchmark.size() : (size_t)std::max(0, skipCount);

    if (startIdx > 0)
    {
        std::cout << "Skipping the first " << startIdx << " kernels..." << std::endl;
    }

    std::cout << (listOnly ? "Listing " : "Benchmarking ") << toBenchmark.size() - startIdx << " configurations..." << std::endl;

    std::ofstream outFile;
    if (!listOnly)
    {
        outFile.open(recordsPath, std::ios::app | std::ios::binary);
    }
    BinaryWriter bw(outFile);

    for (size_t i = startIdx; i < toBenchmark.size(); ++i)
    {
        Record &r = toBenchmark[i];
        uint64_t kernelUid = r.kernelUid;
        const KernelEntry &kernel = KernelRegistry::get().getKernel(kernelUid);

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

        for (size_t idx = 0; idx < r.inputShapes.size(); ++idx)
        {
            std::cout << "  In  #" << idx << ": dtype=" << toString(r.inputDTypes[idx])
                      << ", shape=" << toString(r.inputShapes[idx])
                      << ", strides=" << toString(r.inputStrides[idx]) << "\n";
        }

        for (size_t idx = 0; idx < r.outputShapes.size(); ++idx)
        {
            std::cout << "  Out #" << idx << ": dtype=" << toString(r.outputDTypes[idx])
                      << ", shape=" << toString(r.outputShapes[idx])
                      << ", strides=" << toString(r.outputStrides[idx]) << "\n";
        }

        if (listOnly)
        {
            continue;
        }

        try
        {
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

            if (!kernel.matches(dummyInputs, dummyOutput))
            {
                std::cerr << "Skipping kernel " << kernel.getName() << " (0x" << std::hex << kernelUid << "): record fails matches() validity check." << std::endl;
                continue;
            }

            // Unified RAII buffer lists
            std::vector<BenchBuffer> inputBuffers(r.inputShapes.size());
            std::vector<const void *> inPtrs(r.inputShapes.size(), nullptr);
            std::vector<TensorView> inViews(r.inputShapes.size());

            std::vector<BenchBuffer> outputBuffers(r.outputShapes.size());
            std::vector<void *> outPtrs(r.outputShapes.size(), nullptr);
            std::vector<TensorView> outViews(r.outputShapes.size());

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

                size_t ruleIdx = idx;
                if (kernel.isVariadic)
                {
                    ruleIdx = (idx == r.inputShapes.size() - 1) ? (kernel.inputBackends.empty() ? 0 : kernel.inputBackends.size() - 1) : 0;
                }
                Backend b = Backend::CPU;
                if (!r.inputBackends.empty() && ruleIdx < r.inputBackends.size() && !r.inputBackends[ruleIdx].empty())
                    b = r.inputBackends[ruleIdx][0];

                inputBuffers[idx].allocate(b, bytes);

                if (idx < r.inputConstants.size() && !r.inputConstants[idx].empty() && r.inputConstants[idx].size() == bytes)
                {
                    std::memcpy(inputBuffers[idx].hostData.data(), r.inputConstants[idx].data(), bytes);
                }
                else
                {
                    if (r.inputDTypes[idx] == DType::FLOAT32)
                    {
                        float *fptr = reinterpret_cast<float *>(inputBuffers[idx].hostData.data());
                        for (size_t k = 0; k < elements; ++k)
                            fptr[k] = 1.0f;
                    }
                    else if (r.inputDTypes[idx] == DType::INT32)
                    {
                        int32_t *iptr = reinterpret_cast<int32_t *>(inputBuffers[idx].hostData.data());
                        if (kernel.opType == OpType::PERMUTE || kernel.opName.find("Permute") != std::string::npos)
                        {
                            if (idx == 1 && r.inputShapes.size() > 0 && r.outputShapes.size() > 0 &&
                                r.inputShapes[0].size() == r.outputShapes[0].size() && elements == r.inputShapes[0].size())
                            {
                                std::vector<bool> used(elements, false);
                                for (size_t k = 0; k < elements; ++k)
                                {
                                    size_t found_d = k;
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
                        uint16_t *bptr = reinterpret_cast<uint16_t *>(inputBuffers[idx].hostData.data());
                        for (size_t k = 0; k < elements; ++k)
                            bptr[k] = 0x3F80;
                    }
                    else
                    {
                        std::memset(inputBuffers[idx].hostData.data(), 1, bytes);
                    }
                }

                inputBuffers[idx].upload();
                inPtrs[idx] = inputBuffers[idx].getReadPtr();

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

                Backend outBackend = r.backends.empty() ? Backend::CPU : r.backends[0];
                outputBuffers[idx].allocate(outBackend, bytes);

                if (kernel.inplace && idx == 0)
                {
                    std::memcpy(outputBuffers[idx].hostData.data(), inputBuffers[0].hostData.data(), std::min(inputBuffers[0].bytes, outputBuffers[idx].bytes));
                    outputBuffers[idx].upload();
                }

                outPtrs[idx] = outputBuffers[idx].getWritePtr();

                outViews[idx].setShape(r.outputShapes[idx]);
                outViews[idx].strides = r.outputStrides[idx];
                outViews[idx].baseOffset = 0;
                outViews[idx].dtype = r.outputDTypes[idx];
            }

            KernelContext ctx;
            ctx.inputs = inPtrs;
            ctx.outputs = outPtrs;
            ctx.inViews = inViews;
            ctx.outViews = outViews;
            ctx.fd.assign(inPtrs.size(), -1);

            for (size_t idx = 0; idx < inputBuffers.size(); ++idx)
            {
                ctx.cl_inputs.push_back(inputBuffers[idx].clMem);
            }
            for (size_t idx = 0; idx < outputBuffers.size(); ++idx)
            {
                ctx.cl_outputs.push_back(outputBuffers[idx].clMem);
            }

            StorageFiles sf;

            auto updateStorageContext = [&](int runIdx)
            {
                sf = createStorageInputs(r, kernel, runIdx);
                size_t storageInIdx = 0;
                for (size_t idx = 0; idx < r.inputShapes.size(); ++idx)
                {
                    size_t ruleIdx = idx;
                    if (kernel.isVariadic)
                    {
                        ruleIdx = (idx == r.inputShapes.size() - 1) ? (kernel.inputBackends.empty() ? 0 : kernel.inputBackends.size() - 1) : 0;
                    }
                    Backend b = Backend::CPU;
                    if (!r.inputBackends.empty() && ruleIdx < r.inputBackends.size() && !r.inputBackends[ruleIdx].empty())
                        b = r.inputBackends[ruleIdx][0];

                    if (b == Backend::STORAGE)
                    {
                        if (storageInIdx < sf.fds.size())
                        {
                            ctx.fd[idx] = sf.fds[storageInIdx++];
                        }
                    }
                }
            };

            bool anyCuda = std::any_of(inputBuffers.begin(), inputBuffers.end(), [](const BenchBuffer &b)
                                       { return b.backend == Backend::CUDA; }) ||
                           std::any_of(outputBuffers.begin(), outputBuffers.end(), [](const BenchBuffer &b)
                                       { return b.backend == Backend::CUDA; });
            bool anyOpenCL = std::any_of(inputBuffers.begin(), inputBuffers.end(), [](const BenchBuffer &b)
                                         { return b.backend == Backend::OPENCL; }) ||
                             std::any_of(outputBuffers.begin(), outputBuffers.end(), [](const BenchBuffer &b)
                                         { return b.backend == Backend::OPENCL; });

            std::cout << "  Benchmarking..." << std::flush;

            // Warmup
            if (!kernel.isView)
            {
                updateStorageContext(0);
                kernel.run(ctx);
                if (anyCuda)
                    synchronizeBackend(Backend::CUDA);
                if (anyOpenCL)
                    synchronizeBackend(Backend::OPENCL);
            }

            int iters = 8;
            std::vector<float> latencies;
            latencies.reserve(iters);
            for (int it = 0; it < iters; ++it)
            {
                if (!kernel.isView)
                {
                    updateStorageContext(it + 1);
                }

                auto iterStart = std::chrono::high_resolution_clock::now();
                if (!kernel.isView)
                {
                    kernel.run(ctx);
                }
                if (anyCuda)
                    synchronizeBackend(Backend::CUDA);
                if (anyOpenCL)
                    synchronizeBackend(Backend::OPENCL);
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
            std::cerr << "Failed to benchmark kernel " << kernelUid << ": " << e.what() << std::endl;
        }
    }

    std::cout << "Benchmarking complete." << std::endl;
    return 0;
}