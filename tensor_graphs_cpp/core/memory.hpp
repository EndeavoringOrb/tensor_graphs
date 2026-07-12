// tensor_graphs_cpp/core/memory.hpp
#pragma once
#include "core/types.hpp"
#include "core/hardware.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <cstring>
#include <stdexcept>
#include <mutex>
#include <csignal>
#include <cstdlib>
#include <algorithm>
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

// Add global variables to hold OpenCL Context and Queue
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
                device = devices[0]; // Take the first GPU
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

// Forward declarations
struct DeviceBuffer;

/**
 * InterruptManager handles SIGINT (Ctrl+C) to ensure hardware resources
 * (like CUDA memory) are freed properly before the process exits.
 */
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
        {
            buffers.erase(it);
        }
    }

    static void cleanup(); // Implemented at the bottom of the file

    static void handleSigInt(int signum); // Implemented at the bottom of the file

    static bool isInterrupted()
    {
        return g_interrupted != 0;
    }

    static void resetInterruptFlag()
    {
        g_interrupted = 0;
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

struct MemBlock
{
    uint64_t offset;
    uint64_t sizeBytes;
    uint32_t nodeId; // Use UINT32_MAX to denote a free segment
    StorageType storageType;
    uint32_t refCount;
    bool isLocked;
    float cost; // Track compute cost

    bool isFree() const
    {
        return nodeId == UINT32_MAX;
    }
};

struct DeviceBuffer
{
    Backend backend;
    std::vector<uint8_t> cpu_arena;
    uint8_t *arena_ptr = nullptr;
    cl_mem arena_ptr_cl_mem = nullptr;
    uint64_t sizeBytes;
    bool initialized = false;

    std::list<MemBlock> blocks;
    std::unordered_map<uint32_t, std::list<MemBlock>::iterator> allocationMap;

    void freeArena()
    {
#ifdef USE_CUDA
        if (arena_ptr != nullptr && backend == Backend::CUDA)
        {
            cudaFree(arena_ptr);
            arena_ptr = nullptr;
        }
#endif
        if (arena_ptr_cl_mem != nullptr && backend == Backend::OPENCL)
        {
            clEnqueueUnmapMemObject(OpenCLState::get().queue, arena_ptr_cl_mem, arena_ptr, 0, nullptr, nullptr);
            clReleaseMemObject(arena_ptr_cl_mem);
            arena_ptr_cl_mem = nullptr;
        }
        arena_ptr = nullptr;
    }

    DeviceBuffer(Backend b, uint64_t _sizeBytes) : backend(b), sizeBytes(_sizeBytes)
    {
        // Align overall size to 4096 bytes to facilitate OpenCL zero-copy page alignment
        sizeBytes = (sizeBytes + 4095) & ~4095ULL; // TODO: make this a compile time variable somewhere that is based on hardware query instead of vibes (CL_DEVICE_MEM_BASE_ADDR_ALIGN)

        MemBlock initialFree;
        initialFree.offset = 0;
        initialFree.sizeBytes = sizeBytes;
        initialFree.nodeId = UINT32_MAX;
        initialFree.cost = 0.0f;
        initialFree.isLocked = false;
        blocks.push_back(initialFree);

        InterruptManager::registerBuffer(this);
        InterruptManager::hook();
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    DeviceBuffer(DeviceBuffer &&other) noexcept
        : backend(other.backend), cpu_arena(std::move(other.cpu_arena)),
          arena_ptr(other.arena_ptr), arena_ptr_cl_mem(other.arena_ptr_cl_mem), sizeBytes(other.sizeBytes),
          initialized(other.initialized),
          blocks(std::move(other.blocks)), allocationMap(std::move(other.allocationMap))
    {
        other.arena_ptr = nullptr;
        other.arena_ptr_cl_mem = nullptr;
        InterruptManager::unregisterBuffer(&other);
        InterruptManager::registerBuffer(this);
    }

    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
    {
        if (this != &other)
        {
            freeArena();

            backend = other.backend;
            cpu_arena = std::move(other.cpu_arena);
            arena_ptr = other.arena_ptr;
            arena_ptr_cl_mem = other.arena_ptr_cl_mem;
            sizeBytes = other.sizeBytes;
            initialized = other.initialized;
            blocks = std::move(other.blocks);
            allocationMap = std::move(other.allocationMap);

            other.arena_ptr = nullptr;
            other.arena_ptr_cl_mem = nullptr;
            InterruptManager::unregisterBuffer(&other);
            InterruptManager::registerBuffer(this);
        }
        return *this;
    }

    ~DeviceBuffer()
    {
        InterruptManager::unregisterBuffer(this);
        freeArena();
    }

    void reset()
    {
        blocks.clear();
        allocationMap.clear();
        // Re-add the initial free block spanning the entire arena
        MemBlock initialFree;
        initialFree.offset = 0;
        initialFree.sizeBytes = sizeBytes;
        initialFree.nodeId = UINT32_MAX;
        initialFree.cost = 0.0f;
        initialFree.isLocked = false;
        blocks.push_back(initialFree);
    }

    void mergeFreeBlocks()
    {
        auto it = blocks.begin();
        while (it != blocks.end())
        {
            auto nextIt = std::next(it);
            if (nextIt != blocks.end() && it->isFree() && nextIt->isFree())
            {
                it->sizeBytes += nextIt->sizeBytes;
                blocks.erase(nextIt);
            }
            else
            {
                ++it;
            }
        }
    }

    void freeAllocation(std::list<MemBlock>::iterator it)
    {
        if (it == blocks.end() || it->isFree())
            return;

        for (auto mapIt = allocationMap.begin(); mapIt != allocationMap.end();)
        {
            if (mapIt->second == it)
            {
                mapIt = allocationMap.erase(mapIt);
            }
            else
            {
                ++mapIt;
            }
        }

        it->nodeId = UINT32_MAX;
        it->isLocked = false;
        mergeFreeBlocks();
    }

    void init()
    {
        if (initialized)
            return;
        auto &caps = HardwareCaps::get();

        if (backend == Backend::OPENCL)
        {
            OpenCLState::get().init();
            cl_context ctx = OpenCLState::get().context;
            cl_command_queue queue = OpenCLState::get().queue;
            if (!ctx || !queue)
            {
                Error::throw_err("[DeviceBuffer] Failed to initialize OpenCL Context/Queue");
            }

            cl_int err;
            // 1. Allocate cache-coherent physical memory (zero-copy) via the driver
            arena_ptr_cl_mem = clCreateBuffer(
                ctx,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                sizeBytes,
                nullptr,
                &err);

            if (err != CL_SUCCESS || !arena_ptr_cl_mem)
            {
                Error::throw_err("[DeviceBuffer] clCreateBuffer failed to allocate memory of size " +
                                 std::to_string(sizeBytes) + ". Error: " + std::to_string(err));
            }

            // 2. Map it permanently into the CPU's virtual address space
            arena_ptr = (uint8_t*)clEnqueueMapBuffer(
                queue,
                arena_ptr_cl_mem,
                CL_TRUE, // blocking
                CL_MAP_READ | CL_MAP_WRITE,
                0,
                sizeBytes,
                0, nullptr, nullptr,
                &err);

            if (err != CL_SUCCESS || !arena_ptr)
            {
                Error::throw_err("[DeviceBuffer] clEnqueueMapBuffer failed. Error: " + std::to_string(err));
            }
        }
#ifdef USE_CUDA
        else if (backend == Backend::CUDA)
        {
            if (caps.has_unified_memory)
            {
                // Physically shared memory
                cudaError_t err = cudaMallocManaged(&arena_ptr, sizeBytes);
                if (err != cudaSuccess)
                {
                    Error::throw_err("[DeviceBuffer] cudaMallocManaged failed: " + std::string(cudaGetErrorString(err)));
                }
            }
            else
            {
                cudaError_t err = cudaMalloc(&arena_ptr, sizeBytes);
                if (err != cudaSuccess)
                {
                    Error::throw_err("[DeviceBuffer] cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
                }
            }
        }
#endif // USE_CUDA
        else if (backend == Backend::CPU)
        {
            cpu_arena.resize(sizeBytes + 4096);
            uintptr_t ptr = reinterpret_cast<uintptr_t>(cpu_arena.data());
            arena_ptr = reinterpret_cast<uint8_t *>((ptr + 4095) & ~4095ULL);
        }
        else
        {
            Error::throw_err("Unknown backend");
        }
        initialized = true;
    }

    void write(uint32_t nodeId, const void *data, uint64_t size)
    {
        auto it = allocationMap.find(nodeId);
        if (it != allocationMap.end())
        {
#ifdef USE_CUDA
            if (backend == Backend::CUDA)
            {
                cudaMemcpy(arena_ptr + it->second->offset, data, size, cudaMemcpyHostToDevice);
            }
            else
            {
                std::memcpy(arena_ptr + it->second->offset, data, size);
            }
#else
            std::memcpy(arena_ptr + it->second->offset, data, size);
#endif
        }
        else
        {
            Error::throw_err("Cannot write to unallocated node (nodeId=" + std::to_string(nodeId) + ",size=" + std::to_string(size) + ")");
        }
    }

    void defrag()
    {
        uint64_t offset = 0;
        auto it = blocks.begin();
        while (it != blocks.end())
        {
            if (it->isFree())
            {
                it = blocks.erase(it);
            }
            else
            {
                if (it->offset > offset)
                {
#ifdef USE_CUDA
                    if (backend == Backend::CUDA)
                    {
                        cudaMemcpy(arena_ptr + offset, arena_ptr + it->offset, it->sizeBytes, cudaMemcpyDeviceToDevice);
                    }
                    else
                    {
                        std::memmove(arena_ptr + offset, arena_ptr + it->offset, it->sizeBytes);
                    }
#else
                    std::memmove(arena_ptr + offset, arena_ptr + it->offset, it->sizeBytes);
#endif
                    it->offset = offset;
                }
                offset += it->sizeBytes;
                ++it;
            }
        }

        if (offset < sizeBytes)
        {
            MemBlock freeBlock;
            freeBlock.offset = offset;
            freeBlock.sizeBytes = sizeBytes - offset;
            freeBlock.nodeId = UINT32_MAX;
            freeBlock.storageType = StorageType::TRANSIENT;
            freeBlock.refCount = 0;
            freeBlock.isLocked = false;
            freeBlock.cost = 0.0f;
            blocks.push_back(freeBlock);
        }
    }

    std::list<MemBlock>::iterator findFreeSlot(uint64_t _sizeBytes, bool tryDefrag = true)
    {
        for (auto it = blocks.begin(); it != blocks.end(); ++it)
        {
            if (it->isFree() && it->sizeBytes >= _sizeBytes)
            {
                return it;
            }
        }
        if (tryDefrag) // TODO: store a boolean on DeviceBuffer that tracks if memory has been changed since last defrag, if not then we don't need to defrag again
        {
            mergeFreeBlocks();
            defrag();
            return findFreeSlot(_sizeBytes, false);
        }
        return blocks.end();
    }

    uint64_t allocate(uint32_t nodeId, uint64_t _sizeBytes, StorageType storageType, int32_t refCount, float cost)
    {
        // Align to 4096 bytes for standard OpenCL zero-copy page alignment requirements
        _sizeBytes = (_sizeBytes + 4095) & ~4095ULL;

        // 1. If it's already cached, lock it and update
        auto mapIt = allocationMap.find(nodeId);
        if (mapIt != allocationMap.end())
        {
            auto blockIt = mapIt->second;
            blockIt->refCount = refCount;
            blockIt->isLocked = true;
            blockIt->cost = cost;
            return blockIt->offset;
        }

        // 2. See if there is space already available
        auto slotIt = findFreeSlot(_sizeBytes);

        // 3. If no space, allocation failed.
        if (slotIt == blocks.end())
        {
            Error::throw_err<MemoryAllocationError>("Cannot allocate: Not enough space on " + toString(backend), _sizeBytes);
        }

        // 4. Claim the free slot
        if (slotIt->sizeBytes > _sizeBytes)
        {
            // Split the block (leaves leftovers as free space naturally)
            MemBlock newAlloc;
            newAlloc.offset = slotIt->offset;
            newAlloc.sizeBytes = _sizeBytes;
            newAlloc.nodeId = nodeId;
            newAlloc.storageType = storageType;
            newAlloc.refCount = refCount;
            newAlloc.isLocked = true;
            newAlloc.cost = cost;

            auto insertedIt = blocks.insert(slotIt, newAlloc);
            allocationMap[nodeId] = insertedIt;

            // Shrink the leftover free block
            slotIt->offset += _sizeBytes;
            slotIt->sizeBytes -= _sizeBytes;
            return newAlloc.offset;
        }
        else
        {
            // Exact size match; overwrite free properties
            slotIt->nodeId = nodeId;
            slotIt->storageType = storageType;
            slotIt->refCount = refCount;
            slotIt->isLocked = true;
            slotIt->cost = cost;
            allocationMap[nodeId] = slotIt;
            return slotIt->offset;
        }
    }

    uint64_t getOffset(uint32_t nodeId) const
    {
        auto it = allocationMap.find(nodeId);
        if (it == allocationMap.end())
        {
            Error::throw_err("[DeviceBuffer.getOffset] Node " + std::to_string(nodeId) + " not found in allocation map");
        }
        return it->second->offset;
    }
};

struct MemoryManager
{
    std::unordered_map<Backend, DeviceBuffer> buffers;
    std::unordered_map<uint32_t, uint32_t> aliasMap;
    std::unordered_map<uint32_t, uint32_t> aliasRefCounts;
    std::unordered_map<uint32_t, StorageType> aliasStorageTypes;

    MemoryManager(std::unordered_map<Backend, uint64_t> bufferSizes)
    {
        buffers.reserve(bufferSizes.size());
        for (auto &bufSize : bufferSizes)
        {
            buffers.emplace(bufSize.first, DeviceBuffer(bufSize.first, bufSize.second));
        }
    }

    void init()
    {
        for (auto &buf : buffers)
        {
            buf.second.init();
            buf.second.reset();
        }
        // Clear stale alias entries left over from any previous execution cycle.
        // Multiple sessions sharing a MemoryManager have overlapping node-ID spaces
        // (each graph numbers its nodes independently from 0). A transient alias
        // created during session N's execution and not fully cleaned up will
        // silently redirect writes for session N+1's newly-allocated persistent
        // nodes — whose IDs coincidentally collide — to already-freed addresses.
        aliasMap.clear();
        aliasRefCounts.clear();
        aliasStorageTypes.clear();
    }

    void addAlias(Backend backend, uint32_t srcId, uint32_t dstId, uint32_t additionalRefs, StorageType storageType = StorageType::TRANSIENT)
    {
        if (srcId == dstId)
            return;
        aliasMap[dstId] = srcId;
        aliasRefCounts[dstId] = additionalRefs;
        aliasStorageTypes[dstId] = storageType;
    }

    uint64_t allocate(Backend backend, uint32_t nodeId, uint64_t sizeBytes, StorageType storageType, int32_t refCount = 0, float cost = 0.0f)
    {
        auto it = buffers.find(backend);
        if (it == buffers.end())
            Error::throw_err("[MemoryManager.allocate] DeviceBuffer not initialized for backend " + toString(backend));

        return it->second.allocate(nodeId, sizeBytes, storageType, refCount, cost);
    }

    void write(Backend backend, uint32_t nodeId, const void *data, uint64_t size)
    {
        auto it = buffers.find(backend);
        if (it == buffers.end())
            Error::throw_err("[MemoryManager.write] DeviceBuffer not initialized for backend " + toString(backend));

        uint32_t targetId = nodeId;
        while (aliasMap.find(targetId) != aliasMap.end())
        {
            targetId = aliasMap.at(targetId);
        }

        buffers.at(backend).write(targetId, data, size);
    }

    void release(Backend backend, uint32_t nodeId)
    {
        // Check if this is an alias
        auto aliasIt = aliasMap.find(nodeId);
        if (aliasIt != aliasMap.end())
        {
            auto refIt = aliasRefCounts.find(nodeId);
            if (refIt != aliasRefCounts.end() && refIt->second > 0)
            {
                refIt->second--;
                // Only deallocate if the reference count is actually zero
                if (refIt->second == 0)
                {
                    auto storageIt = aliasStorageTypes.find(nodeId);
                    // Ensure we only recurse if the alias is TRANSIENT
                    if (storageIt == aliasStorageTypes.end() || storageIt->second == StorageType::TRANSIENT)
                    {
                        uint32_t targetId = aliasIt->second;

                        // Clean up the alias metadata before recursing to the underlying ID
                        aliasRefCounts.erase(nodeId);
                        if (storageIt != aliasStorageTypes.end())
                            aliasStorageTypes.erase(nodeId);
                        aliasMap.erase(nodeId);

                        release(backend, targetId);
                    }
                }
            }
            return;
        }

        auto bufIt = buffers.find(backend);
        if (bufIt == buffers.end())
            Error::throw_err("[MemoryManager.release] DeviceBuffer not initialized for backend " + toString(backend));

        auto &buf = buffers.at(backend);
        auto it = buf.allocationMap.find(nodeId);
        if (it != buf.allocationMap.end())
        {
            if (it->second->storageType == StorageType::TRANSIENT)
            {
                if (it->second->refCount > 0)
                {
                    it->second->refCount--;
                    // Transients are reclaimed as soon as the last consumer releases them.
                    if (it->second->refCount == 0)
                    {
                        buf.freeAllocation(it->second);
                    }
                }
            }
        }
    }

    void transferOwnership(Backend backend, uint32_t srcId, uint32_t dstId)
    {
        if (srcId == dstId)
            return;

        auto dstAliasIt = aliasMap.find(dstId);
        if (dstAliasIt != aliasMap.end())
        {
            aliasMap.erase(dstAliasIt);
            aliasRefCounts.erase(dstId);
            aliasStorageTypes.erase(dstId);
        }

        auto &buf = buffers.at(backend);

        auto aliasIt = aliasMap.find(srcId);
        if (aliasIt != aliasMap.end())
        {
            aliasMap[dstId] = aliasIt->second;
            aliasRefCounts[dstId] = aliasRefCounts[srcId];
            aliasStorageTypes[dstId] = aliasStorageTypes[srcId];
            aliasMap.erase(aliasIt);
            aliasRefCounts.erase(srcId);
            aliasStorageTypes.erase(srcId);

            auto dstIt = buf.allocationMap.find(dstId);
            if (dstIt != buf.allocationMap.end())
            {
                dstIt->second->nodeId = UINT32_MAX;
                dstIt->second->isLocked = false;
                buf.allocationMap.erase(dstIt);
                buf.mergeFreeBlocks();
            }
            return;
        }

        auto srcIt = buf.allocationMap.find(srcId);
        if (srcIt != buf.allocationMap.end())
        {
            auto blockIt = srcIt->second;

            if (blockIt->storageType == StorageType::PERSISTENT || blockIt->storageType == StorageType::PINNED)
            {
                // Cannot transfer ownership of a persistent/pinned block (it must survive!)
                // Create an alias to share the inplace memory safely instead.
                addAlias(backend, srcId, dstId, 0, StorageType::TRANSIENT);
            }
            else
            {
                auto dstIt = buf.allocationMap.find(dstId);
                if (dstIt != buf.allocationMap.end())
                {
                    dstIt->second->nodeId = UINT32_MAX;
                    dstIt->second->isLocked = false;
                    buf.allocationMap.erase(dstIt);
                    buf.mergeFreeBlocks();
                }

                buf.allocationMap.erase(srcIt);

                // Update node identity
                blockIt->nodeId = dstId;
                buf.allocationMap[dstId] = blockIt;
            }
        }
        else
        {
            Error::throw_err("[MemoryManager.transferOwnership] Source ID not found in allocation map");
        }
    }

    uint32_t resolveAlias(uint32_t id)
    {
        while (aliasMap.find(id) != aliasMap.end())
        {
            id = aliasMap.at(id);
        }
        return id;
    }

    TensorView getView(const TensorNode &node, const uint32_t overrideId) const
    {
        uint32_t targetId = overrideId;
        while (aliasMap.find(targetId) != aliasMap.end())
        {
            targetId = aliasMap.at(targetId);
        }

        // Cross-backend lookup for Unified Memory
        // TODO: is there a better way to do this?
        Backend actualBackend = node.backend;
        if (buffers.find(Backend::CUDA) != buffers.end() && buffers.at(Backend::CUDA).allocationMap.count(targetId))
        {
            actualBackend = Backend::CUDA;
        }
        else if (buffers.find(Backend::CPU) != buffers.end() && buffers.at(Backend::CPU).allocationMap.count(targetId))
        {
            actualBackend = Backend::CPU;
        }
        else if (buffers.find(Backend::OPENCL) != buffers.end() && buffers.at(Backend::OPENCL).allocationMap.count(targetId))
        {
            actualBackend = Backend::OPENCL;
        }

        const DeviceBuffer &buf = buffers.at(actualBackend);
        uint64_t arenaOffset = buf.getOffset(targetId);
        return TensorView(node, arenaOffset + node.viewOffset * getDTypeSize(node.dtype));
    }

    bool has(Backend backend, uint32_t nodeId) const
    {
        uint32_t targetId = nodeId;
        auto aliasIt = aliasMap.find(targetId);
        while (aliasIt != aliasMap.end())
        {
            targetId = aliasIt->second;
            aliasIt = aliasMap.find(targetId);
        }

        // 1. Try the expected backend buffer first
        auto it = buffers.find(backend);
        if (it != buffers.end() && it->second.allocationMap.count(targetId))
        {
            return true;
        }

        // 2. Check other backends (required for Unified Memory views spanning backends)
        for (const auto &pair : buffers)
        {
            if (pair.first == backend)
                continue;
            if (pair.second.allocationMap.count(targetId))
                return true;
        }

        return false;
    }

    uint64_t getCapacity(Backend backend) const
    {
        return buffers.at(backend).sizeBytes;
    }

    std::unordered_map<Backend, uint64_t> getBufferSizes() const
    {
        std::unordered_map<Backend, uint64_t> sizes;
        for (const auto &pair : buffers)
        {
            sizes[pair.first] = pair.second.sizeBytes;
        }
        return sizes;
    }

    MemBlock &getBlock(Backend backend, uint32_t nodeId)
    {
        auto it = buffers.find(backend);
        if (it == buffers.end())
        {
            Error::throw_err("[MemoryManager.getBlock] Backend buffer not initialized");
        }

        uint32_t targetId = nodeId;
        while (aliasMap.find(targetId) != aliasMap.end())
        {
            targetId = aliasMap.at(targetId);
        }

        const DeviceBuffer &buf = it->second;
        auto bufIt = buf.allocationMap.find(targetId);
        if (bufIt == buf.allocationMap.end())
        {
            Error::throw_err("[MemoryManager.getBlock] Buffer allocation map doesn't have targetId " + std::to_string(targetId));
        }
        return *bufIt->second;
    }
};

// TODO: Can this be moved inside InterruptManager?
inline void InterruptManager::cleanup()
{
    std::lock_guard<std::mutex> lock(mtx);
    for (auto *buf : buffers)
    {
        buf->freeArena(); // Requires full definition of DeviceBuffer
    }
    buffers.clear();
}

inline void InterruptManager::handleSigInt(int signum)
{
    std::cerr << "\n[TensorGraph] Caught interrupt signal (" << signum << "). Cleaning up..." << std::endl;
    g_interrupted = 1; // Just set the flag - cleanup happens in main thread
}