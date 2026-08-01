#pragma once
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "core/hardware.hpp"
#include "core/loaders/loader.hpp"
#include "core/types.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
#include <CL/cl.h>

struct OpenCLState
{
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_device_id device = nullptr;
    bool initialized = false;

    void init()
    {
        if (initialized)
            return;
        cl_uint numPlatforms = 0;
        clGetPlatformIDs(0, nullptr, &numPlatforms);
        if (numPlatforms == 0)
            return;
        std::vector<cl_platform_id> platforms(numPlatforms);
        clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

        for (auto plat : platforms)
        {
            cl_uint numDevices = 0;
            clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
            if (numDevices > 0)
            {
                std::vector<cl_device_id> devices(numDevices);
                clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr);
                device = devices[0];
                break;
            }
        }
        if (device)
        {
            cl_int err;
            context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
            initialized = true;
        }
    }

    static OpenCLState &get()
    {
        static OpenCLState instance;
        return instance;
    }
};

struct DeviceBuffer;

struct InterruptManager
{
    static inline std::vector<DeviceBuffer *> buffers;
    static inline std::mutex mtx;
    static inline volatile sig_atomic_t g_interrupted = 0;

    static void registerBuffer(DeviceBuffer *buf)
    {
        std::lock_guard<std::mutex> lock(mtx);
        buffers.push_back(buf);
    }
    static void unregisterBuffer(DeviceBuffer *buf)
    {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = std::find(buffers.begin(), buffers.end(), buf);
        if (it != buffers.end())
            buffers.erase(it);
    }
    static void cleanup();
    static void handleSigInt(int signum)
    {
        std::cerr << "\n[TensorGraph] Caught interrupt signal (" << signum << "). Cleaning up..." << std::endl;
        g_interrupted = 1;
    }
    static bool isInterrupted()
    {
        return g_interrupted != 0;
    }
    static void hook()
    {
        static bool hooked = false;
        if (!hooked)
        {
            std::signal(SIGINT, handleSigInt);
            hooked = true;
        }
    }
};

struct DeviceBuffer
{
    MemSpace mem_space;
    uint64_t sizeBytes;

    DeviceBuffer(MemSpace ms, uint64_t size) : mem_space(ms), sizeBytes((size + 4095) & ~4095ULL)
    {
    }
    virtual ~DeviceBuffer() = default;

    virtual void init() = 0;
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
    void freeArena() override
    {
    }
    void write(uint64_t offset, const void *data, uint64_t size) override
    {
        Error::throw_err("Cannot write to StorageBuffer");
    }
    void setupInput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
        TensorMetadata meta = FileRegistry::get().getNodeMeta(logicalId);
        TensorView v = view;
        v.offset = meta.dataOffsetStart;
        ctx.inViews.push_back(v);
        ctx.inputs.push_back(nullptr);
        ctx.fd.push_back(FileRegistry::get().getNodeFd(logicalId));
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
        InterruptManager::registerBuffer(this);
        InterruptManager::hook();
    }
    ~CppBuffer() override
    {
        InterruptManager::unregisterBuffer(this);
        freeArena();
    }
    void init() override
    {
        cpu_arena.resize(sizeBytes + 4096);
        uintptr_t ptr = reinterpret_cast<uintptr_t>(cpu_arena.data());
        arena_ptr = reinterpret_cast<uint8_t *>((ptr + 4095) & ~4095ULL);
    }
    void freeArena() override
    {
        arena_ptr = nullptr;
        cpu_arena.clear();
    }
    void write(uint64_t offset, const void *data, uint64_t size) override
    {
        std::memcpy(arena_ptr + offset, data, size);
    }
    void setupInput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
        TensorView v = view;
        ctx.inViews.push_back(v);
        ctx.inputs.push_back(arena_ptr + v.offset);
        ctx.fd.push_back(-1);
        ctx.cl_inputs.push_back(nullptr);
    }
    void setupOutput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
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

#ifdef USE_CUDA
struct CudaBuffer : public DeviceBuffer
{
    uint8_t *arena_ptr = nullptr;

    CudaBuffer(MemSpace ms, uint64_t size) : DeviceBuffer(ms, size)
    {
        InterruptManager::registerBuffer(this);
        InterruptManager::hook();
    }
    ~CudaBuffer() override
    {
        InterruptManager::unregisterBuffer(this);
        freeArena();
    }
    void init() override
    {
        if (HardwareCaps::get().has_unified_memory)
        {
            cudaError_t err = cudaMallocManaged(&arena_ptr, sizeBytes);
            if (err != cudaSuccess)
                Error::throw_err("cudaMallocManaged failed");
        }
        else
        {
            cudaError_t err = cudaMalloc(&arena_ptr, sizeBytes);
            if (err != cudaSuccess)
                Error::throw_err("cudaMalloc failed");
        }
    }
    void freeArena() override
    {
        if (arena_ptr)
        {
            cudaFree(arena_ptr);
            arena_ptr = nullptr;
        }
    }
    void write(uint64_t offset, const void *data, uint64_t size) override
    {
        Error::throw_err("writeInput is not supported on CudaBuffer");
    }
    void setupInput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
        TensorView v = view;
        ctx.inViews.push_back(v);
        ctx.inputs.push_back(arena_ptr + v.offset);
        ctx.fd.push_back(-1);
        ctx.cl_inputs.push_back(nullptr);
    }

    void setupOutput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
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

struct OpenCLBuffer : public DeviceBuffer
{
    cl_mem arena_ptr_cl_mem = nullptr;

    OpenCLBuffer(MemSpace ms, uint64_t size) : DeviceBuffer(ms, size)
    {
        InterruptManager::registerBuffer(this);
        InterruptManager::hook();
    }
    ~OpenCLBuffer() override
    {
        InterruptManager::unregisterBuffer(this);
        freeArena();
    }
    void init() override
    {
        OpenCLState::get().init();
        cl_context ctx = OpenCLState::get().context;
        cl_int err;
        arena_ptr_cl_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeBytes, nullptr, &err);
        if (err != CL_SUCCESS)
            Error::throw_err("clCreateBuffer failed");
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
        TensorView v = view;
        ctx.inViews.push_back(v);
        ctx.inputs.push_back(nullptr);
        ctx.fd.push_back(-1);

        uint64_t size = countElements(view) * getDTypeSize(view.dtype);
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
                Error::throw_err("clCreateSubBuffer failed");
        }
        ctx.cl_inputs.push_back(buf);
    }
    void setupOutput(KernelContext &ctx, const TensorView &view, LogicalId logicalId) override
    {
        TensorView v = view;
        ctx.outViews.push_back(v);
        ctx.outputs.push_back(nullptr);

        uint64_t size = countElements(view) * getDTypeSize(view.dtype);
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
                Error::throw_err("clCreateSubBuffer failed");
        }
        ctx.cl_outputs.push_back(buf);
    }
    void cleanupContext(KernelContext &ctx) override
    {
        for (cl_mem sub : ctx.cl_inputs)
        {
            if (sub)
                clReleaseMemObject(sub);
        }
        for (cl_mem sub : ctx.cl_outputs)
        {
            if (sub)
                clReleaseMemObject(sub);
        }
    }
    uint8_t *getBasePtr() override
    {
        return nullptr;
    }
};

struct MemoryManager
{
    std::unordered_map<MemSpace, std::unique_ptr<DeviceBuffer>> buffers;

    MemoryManager(std::unordered_map<MemSpace, uint64_t> bufferSizes)
    {
        // Register default storage buffer space
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
                buffers[ms] = std::make_unique<OpenCLBuffer>(ms, size);
            }
            else if (ms.type == HandleType::CUDA)
            {
#ifdef USE_CUDA
                buffers[ms] = std::make_unique<CudaBuffer>(ms, size);
#else
                Error::throw_err("CUDA requested but USE_CUDA not defined");
#endif
            }
        }
    }

    void init()
    {
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

inline void InterruptManager::cleanup()
{
    std::lock_guard<std::mutex> lock(mtx);
    for (auto *buf : buffers)
    {
        buf->freeArena();
    }
    buffers.clear();
}