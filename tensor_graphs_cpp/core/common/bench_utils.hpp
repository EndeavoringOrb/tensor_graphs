// File: tensor_graphs_cpp/common/bench_utils.hpp
#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
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

inline StorageFiles createStorageInputs(const Record &r, const KernelEntry &kernel, int runIdx)
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

inline void synchronizeBackend(Backend backend)
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