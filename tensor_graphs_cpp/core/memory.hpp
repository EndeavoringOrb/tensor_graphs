#pragma once
#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "core/hardware.hpp"
#include "core/loaders/resolver.hpp"
#include "core/types.hpp"

#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#endif

struct DeviceBuffer
{
    MemSpace mem_space;
    uint64_t sizeBytes;

    DeviceBuffer(MemSpace ms, uint64_t size) : mem_space(ms), sizeBytes((size + 4095) & ~4095ULL)
    {
    }
    virtual ~DeviceBuffer() = default;

    virtual void init() = 0;
    virtual void resize(uint64_t newSizeBytes) = 0;
    virtual void freeArena() = 0;
    virtual void write(uint64_t offset, const void *data, uint64_t size) = 0;

    virtual void setupInput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) = 0;
    virtual void setupOutput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) = 0;
    virtual void cleanupContext(KernelContext &ctx) = 0;
    virtual uint8_t *getBasePtr()
    {
        return nullptr;
    }
};

struct StorageBuffer : public DeviceBuffer
{
    StorageBuffer(MemSpace ms, uint64_t size) : DeviceBuffer(ms, size)
    {
    }
    void init() override
    {
    }
    void resize(uint64_t newSizeBytes) override
    {
        sizeBytes = newSizeBytes;
    }
    void freeArena() override
    {
    }
    void write(uint64_t offset, const void *data, uint64_t size) override
    {
        Error::throw_err("Cannot write to StorageBuffer");
    }
    void setupInput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
        if (logicalId == LogicalId{UINT32_MAX} || logicalId.value == UINT32_MAX)
        {
            Error::throw_err("StorageBuffer::setupInput: logicalId is uninitialized (UINT32_MAX). "
                             "Check that storage EClass was properly mapped to its source weight LogicalId.");
        }
        TensorMetadata meta = TensorResolver::get().getNodeMeta(logicalId);
        TensorView v = view;
        v.offset = meta.dataOffsetStart + view.offset;
        ctx.inViews.push_back(v);
        ctx.inputs.push_back(nullptr);
        ctx.fd.push_back(TensorResolver::get().getNodeFd(logicalId));
        ctx.cl_inputs.push_back(nullptr);
    }
    void setupOutput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
        Error::throw_err("Cannot use StorageBuffer as output");
    }
    void cleanupContext(KernelContext &ctx) override
    {
    }
};

struct CppBuffer : public DeviceBuffer
{
    std::vector<uint8_t> cpu_arena;
    uint8_t *arena_ptr = nullptr;

    CppBuffer(MemSpace ms, uint64_t size) : DeviceBuffer(ms, size)
    {
    }
    ~CppBuffer() override
    {
        freeArena();
    }
    void init() override
    {
        if (arena_ptr)
            return;
        uint64_t allocSize = std::max<uint64_t>(4096ULL, sizeBytes);
        cpu_arena.resize(allocSize + 4096);
        uintptr_t ptr = reinterpret_cast<uintptr_t>(cpu_arena.data());
        arena_ptr = reinterpret_cast<uint8_t *>((ptr + 4095) & ~4095ULL);
    }
    void resize(uint64_t newSizeBytes) override
    {
        freeArena();
        sizeBytes = std::max<uint64_t>(4096ULL, (newSizeBytes + 4095) & ~4095ULL);
        init();
    }
    void freeArena() override
    {
        arena_ptr = nullptr;
        cpu_arena.clear();
    }
    void write(uint64_t offset, const void *data, uint64_t size) override
    {
        if (!arena_ptr)
            init();
        std::memcpy(arena_ptr + offset, data, size);
    }
    void setupInput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
        if (!arena_ptr)
            init();
        TensorView v = view;
        ctx.inViews.push_back(v);
        ctx.inputs.push_back(arena_ptr + v.offset);
        ctx.fd.push_back(-1);
        ctx.cl_inputs.push_back(nullptr);
    }
    void setupOutput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
        if (!arena_ptr)
            init();
        TensorView v = view;
        ctx.outViews.push_back(v);
        ctx.outputs.push_back(arena_ptr + v.offset);
        ctx.cl_outputs.push_back(nullptr);
    }
    void cleanupContext(KernelContext &ctx) override
    {
    }
    uint8_t *getBasePtr() override
    {
        return arena_ptr;
    }
};

#ifdef TG_USE_CUDA
struct CudaBuffer : public DeviceBuffer
{
    uint8_t *arena_ptr = nullptr;

    CudaBuffer(MemSpace ms, uint64_t size) : DeviceBuffer(ms, size)
    {
    }
    ~CudaBuffer() override
    {
        freeArena();
    }
    void init() override
    {
        if (arena_ptr)
            return;
        uint64_t allocSize = std::max<uint64_t>(4096ULL, sizeBytes);
        cudaSetDevice(mem_space.idx);
        if (HardwareCaps::get().has_unified_memory)
        {
            cudaError_t err = cudaMallocManaged(&arena_ptr, allocSize);
            if (err != cudaSuccess)
                Error::throw_err("cudaMallocManaged failed for device " + std::to_string(mem_space.idx) + ": " +
                                 cudaGetErrorString(err));
        }
        else
        {
            cudaError_t err = cudaMalloc(&arena_ptr, allocSize);
            if (err != cudaSuccess)
                Error::throw_err("cudaMalloc failed for device " + std::to_string(mem_space.idx) + ": " +
                                 cudaGetErrorString(err));
        }
    }
    void resize(uint64_t newSizeBytes) override
    {
        freeArena();
        sizeBytes = std::max<uint64_t>(4096ULL, (newSizeBytes + 4095) & ~4095ULL);
        init();
    }
    void freeArena() override
    {
        if (arena_ptr)
        {
            cudaSetDevice(mem_space.idx);
            cudaFree(arena_ptr);
            arena_ptr = nullptr;
        }
    }
    void write(uint64_t offset, const void *data, uint64_t size) override
    {
        if (!arena_ptr)
            init();
        cudaSetDevice(mem_space.idx);
        cudaError_t err = cudaMemcpy(arena_ptr + offset, data, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            Error::throw_err("cudaMemcpy HostToDevice failed on device " + std::to_string(mem_space.idx) + ": " +
                             cudaGetErrorString(err));
    }
    void setupInput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
        if (!arena_ptr)
            init();
        TensorView v = view;
        ctx.inViews.push_back(v);
        ctx.inputs.push_back(arena_ptr + v.offset);
        ctx.fd.push_back(-1);
        ctx.cl_inputs.push_back(nullptr);
    }
    void setupOutput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
        if (!arena_ptr)
            init();
        TensorView v = view;
        ctx.outViews.push_back(v);
        ctx.outputs.push_back(arena_ptr + v.offset);
        ctx.cl_outputs.push_back(nullptr);
    }
    void cleanupContext(KernelContext &ctx) override
    {
    }
    uint8_t *getBasePtr() override
    {
        return arena_ptr;
    }
};
#endif

#ifdef TG_USE_OPENCL
struct OpenCLBuffer : public DeviceBuffer
{
    cl_mem arena_ptr_cl_mem = nullptr;

    OpenCLBuffer(MemSpace ms, uint64_t size) : DeviceBuffer(ms, size)
    {
    }
    ~OpenCLBuffer() override
    {
        freeArena();
    }
    void init() override
    {
        if (arena_ptr_cl_mem)
            return;
        uint64_t allocSize = std::max<uint64_t>(4096ULL, sizeBytes);
        OpenCLState::get().init();
        cl_context ctx = OpenCLState::get().context;
        cl_int err;
        arena_ptr_cl_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, allocSize, nullptr, &err);
        if (err != CL_SUCCESS)
            Error::throw_err("clCreateBuffer failed with error: " + std::to_string(err));
    }
    void resize(uint64_t newSizeBytes) override
    {
        freeArena();
        sizeBytes = std::max<uint64_t>(4096ULL, (newSizeBytes + 4095) & ~4095ULL);
        init();
    }
    void freeArena() override
    {
        if (arena_ptr_cl_mem)
        {
            clReleaseMemObject(arena_ptr_cl_mem);
            arena_ptr_cl_mem = nullptr;
        }
    }
    void write(uint64_t offset, const void *data, uint64_t size) override
    {
        Error::throw_err("writeInput is not supported on OpenCLBuffer");
    }
    void setupInput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
        if (!arena_ptr_cl_mem)
            init();
        TensorView v = view;
        ctx.inViews.push_back(v);
        ctx.inputs.push_back(nullptr);
        ctx.fd.push_back(-1);

        uint64_t size = getRequiredBufferSize(view) * getDTypeSize(view.dtype);
        if (size == 0)
            size = 1;

        cl_mem buf = nullptr;
        for (uint64_t i = 0; i < ctx.cl_inputs.size(); i++)
        {
            if (ctx.inViews[i].offset == v.offset && ctx.cl_inputs[i] != nullptr)
            {
                buf = ctx.cl_inputs[i];
                clRetainMemObject(buf);
                break;
            }
        }
        if (!buf)
        {
            cl_buffer_region region;
            region.origin = v.offset;
            region.size = size;
            cl_int err;
            buf = clCreateSubBuffer(arena_ptr_cl_mem, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
            if (err != CL_SUCCESS)
                Error::throw_err("clCreateSubBuffer failed in setupInput (offset=" + std::to_string(v.offset) +
                                 "): error " + std::to_string(err));
        }
        ctx.cl_inputs.push_back(buf);
    }
    void setupOutput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
        if (!arena_ptr_cl_mem)
            init();
        TensorView v = view;
        ctx.outViews.push_back(v);
        ctx.outputs.push_back(nullptr);

        uint64_t size = getRequiredBufferSize(view) * getDTypeSize(view.dtype);
        if (size == 0)
            size = 1;

        cl_mem buf = nullptr;
        for (uint64_t i = 0; i < ctx.cl_inputs.size(); i++)
        {
            if (ctx.inViews[i].offset == v.offset && ctx.cl_inputs[i] != nullptr)
            {
                buf = ctx.cl_inputs[i];
                clRetainMemObject(buf);
                break;
            }
        }
        if (!buf)
        {
            cl_buffer_region region;
            region.origin = v.offset;
            region.size = size;
            cl_int err;
            buf = clCreateSubBuffer(arena_ptr_cl_mem, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
            if (err != CL_SUCCESS)
                Error::throw_err("clCreateSubBuffer failed in setupOutput (offset=" + std::to_string(v.offset) +
                                 "): error " + std::to_string(err));
        }
        ctx.cl_outputs.push_back(buf);
    }
    void cleanupContext(KernelContext &ctx) override
    {
        for (cl_mem &sub : ctx.cl_inputs)
        {
            if (sub)
            {
                clReleaseMemObject(sub);
                sub = nullptr;
            }
        }
        for (cl_mem &sub : ctx.cl_outputs)
        {
            if (sub)
            {
                clReleaseMemObject(sub);
                sub = nullptr;
            }
        }
    }
    uint8_t *getBasePtr() override
    {
        Error::throw_err("getBasePtr not implemented for opencl");
    }
};
#endif // TG_USE_OPENCL

struct MemoryManager
{
    std::unordered_map<MemSpace, std::unique_ptr<DeviceBuffer>> buffers;

    MemoryManager(std::unordered_map<MemSpace, uint64_t> bufferSizes = {})
    {
        if (bufferSizes.empty())
        {
            bufferSizes = System::get().getBufferSizes();
        }

        buffers[MemSpace{0, HandleType::STORAGE}] =
            std::make_unique<StorageBuffer>(MemSpace{0, HandleType::STORAGE}, 0);

        for (auto &pair : bufferSizes)
        {
            MemSpace ms = pair.first;
            uint64_t size = pair.second;

            if (ms.type == HandleType::STORAGE)
            {
                buffers[ms] = std::make_unique<StorageBuffer>(ms, size);
            }
            else if (ms.type == HandleType::CPP)
            {
                buffers[ms] = std::make_unique<CppBuffer>(ms, size);
            }
            else if (ms.type == HandleType::OPENCL)
            {
#ifdef TG_USE_OPENCL
                buffers[ms] = std::make_unique<OpenCLBuffer>(ms, size);
#else
                Error::throw_err("OPENCL requested but TG_USE_OPENCL not defined");
#endif
            }
            else if (ms.type == HandleType::CUDA)
            {
#ifdef TG_USE_CUDA
                buffers[ms] = std::make_unique<CudaBuffer>(ms, size);
#else
                Error::throw_err("CUDA requested but TG_USE_CUDA not defined");
#endif
            }
        }
    }

    void resizeBuffer(MemSpace ms, uint64_t newSizeBytes)
    {
        auto it = buffers.find(ms);
        if (it != buffers.end() && it->second)
        {
            it->second->resize(newSizeBytes);
        }
        else
        {
            Error::throw_err("Buffer not initialized for resize in MemSpace(idx=" + std::to_string(ms.idx) +
                             ", type=" + toString(ms.type) + ")");
        }
    }

    void init(const std::unordered_map<MemSpace, uint64_t> &peakSizes = {})
    {
        for (const auto &pair : peakSizes)
        {
            auto it = buffers.find(pair.first);
            if (it != buffers.end() && it->second)
            {
                it->second->resize(pair.second);
            }
        }
        for (auto &pair : buffers)
        {
            if (pair.second)
                pair.second->init();
        }
    }

    void write(MemSpace ms, uint64_t offset, const void *data, uint64_t size)
    {
        auto it = buffers.find(ms);
        if (it != buffers.end() && it->second)
        {
            it->second->write(offset, data, size);
        }
        else
        {
            Error::throw_err("Buffer not initialized for MemSpace(idx=" + std::to_string(ms.idx) +
                             ", type=" + toString(ms.type) + ")");
        }
    }

    DeviceBuffer *getBuffer(MemSpace ms)
    {
        auto it = buffers.find(ms);
        if (it != buffers.end())
            return it->second.get();
        return nullptr;
    }

    std::unordered_map<MemSpace, uint64_t> getMemCaps() const
    {
        std::unordered_map<MemSpace, uint64_t> sizes;
        for (const auto &pair : buffers)
        {
            if (pair.second)
            {
                sizes[pair.first] = pair.second->sizeBytes;
            }
        }
        return sizes;
    }
};