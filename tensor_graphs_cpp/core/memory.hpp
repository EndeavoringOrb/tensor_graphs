#pragma once
#include "core/types.hpp"
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

inline float calculateSavedCost(
    const std::unordered_set<uint32_t> &cacheState,
    const std::unordered_map<uint32_t, std::vector<uint32_t>> &parentMap,
    const std::unordered_map<uint32_t, float> &nodeCosts)
{
    std::unordered_set<uint32_t> savedNodes;
    std::vector<uint32_t> stack;

    // A node in the cache saves itself AND all of its recursive ancestors.
    for (uint32_t node : cacheState)
    {
        if (savedNodes.insert(node).second)
        {
            stack.push_back(node);
        }
    }

    while (!stack.empty())
    {
        uint32_t curr = stack.back();
        stack.pop_back();

        auto it = parentMap.find(curr);
        if (it != parentMap.end())
        {
            for (uint32_t p : it->second)
            {
                if (savedNodes.insert(p).second)
                {
                    stack.push_back(p);
                }
            }
        }
    }

    float totalCost = 0.0f;
    for (uint32_t node : savedNodes)
    {
        auto it = nodeCosts.find(node);
        if (it != nodeCosts.end())
        {
            totalCost += it->second;
        }
    }
    return totalCost;
}

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
    }

    DeviceBuffer(Backend b, uint64_t _sizeBytes) : backend(b), sizeBytes(_sizeBytes)
    {
        MemBlock initialFree;
        initialFree.offset = 0;
        initialFree.sizeBytes = _sizeBytes;
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
          arena_ptr(other.arena_ptr), sizeBytes(other.sizeBytes),
          initialized(other.initialized),
          blocks(std::move(other.blocks)), allocationMap(std::move(other.allocationMap))
    {
        other.arena_ptr = nullptr;
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
            sizeBytes = other.sizeBytes;
            initialized = other.initialized;
            blocks = std::move(other.blocks);
            allocationMap = std::move(other.allocationMap);

            other.arena_ptr = nullptr;
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

        allocationMap.erase(it->nodeId);
        it->nodeId = UINT32_MAX;
        it->storageType = StorageType::TRANSIENT;
        it->refCount = 0;
        it->isLocked = false;
        it->cost = 0.0f;
        mergeFreeBlocks();
    }

    void init()
    {
        if (initialized)
            return;

#ifdef USE_CUDA
        if (backend == Backend::CUDA)
        {
            cudaError_t err = cudaMalloc(&arena_ptr, sizeBytes);
            if (err != cudaSuccess)
            {
                Error::throw_err("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
            }
        }
        else if (backend == Backend::CPU)
        {
            cpu_arena.resize(sizeBytes);
            arena_ptr = cpu_arena.data();
        }
        else
        {
            Error::throw_err("Unknown backend");
        }
#else
        if (backend == Backend::CPU)
        {
            cpu_arena.resize(sizeBytes);
            arena_ptr = cpu_arena.data();
        }
        else if (backend == Backend::CUDA)
        {
            Error::throw_err("CUDA backend not supported without USE_CUDA");
        }
        else
        {
            Error::throw_err("Unknown backend");
        }
#endif
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
            Error::throw_err("Cannot write to unallocated node");
        }
    }

    const uint8_t *read(uint32_t nodeId) const
    {
        auto it = allocationMap.find(nodeId);
        if (it != allocationMap.end())
        {
#ifdef USE_CUDA
            if (backend == Backend::CUDA)
            {
                cudaDeviceSynchronize();
            }
#endif
            return arena_ptr + it->second->offset;
        }
        return nullptr;
    }

    void defrag()
    {
        uint64_t offset = 0;
        for (auto it = blocks.begin(); it != blocks.end(); ++it)
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
                    std::memcpy(arena_ptr + offset, arena_ptr + it->offset, it->sizeBytes);
                }
#else
                std::memcpy(arena_ptr + offset, arena_ptr + it->offset, it->sizeBytes);
#endif
            }
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

    uint64_t allocate(uint32_t nodeId, uint64_t _sizeBytes, StorageType storageType, int32_t refCount, float cost,
                      const std::unordered_map<uint32_t, std::vector<uint32_t>> *parentMap = nullptr,
                      const std::unordered_map<uint32_t, float> *nodeCosts = nullptr)
    {
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
            Error::throw_err<MemoryAllocationError>("Cannot allocate: Not enough space.", _sizeBytes);
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
        }
    }

    uint64_t allocate(Backend backend, uint32_t nodeId, uint64_t sizeBytes, StorageType storageType, int32_t refCount = 0, float cost = 0.0f, const std::unordered_map<uint32_t, std::vector<uint32_t>> *parentMap = nullptr, const std::unordered_map<uint32_t, float> *nodeCosts = nullptr)
    {
        auto it = buffers.find(backend);
        if (it == buffers.end())
            Error::throw_err("[MemoryManager.allocate] DeviceBuffer not initialized for backend " + toString(backend));

        return it->second.allocate(nodeId, sizeBytes, storageType, refCount, cost, parentMap, nodeCosts);
    }

    void write(Backend backend, uint32_t nodeId, const void *data, uint64_t size)
    {
        auto it = buffers.find(backend);
        if (it == buffers.end())
            Error::throw_err("[MemoryManager.write] DeviceBuffer not initialized for backend " + toString(backend));

        buffers.at(backend).write(nodeId, data, size);
    }

    const uint8_t *read(Backend backend, uint32_t nodeId) const
    {
        auto it = buffers.find(backend);
        if (it == buffers.end())
            Error::throw_err("[MemoryManager.read] DeviceBuffer not initialized for backend " + toString(backend));

        return buffers.at(backend).read(nodeId);
    }

    void release(Backend backend, uint32_t nodeId)
    {
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

        auto &buf = buffers.at(backend);
        auto srcIt = buf.allocationMap.find(srcId);
        if (srcIt != buf.allocationMap.end())
        {
            auto dstIt = buf.allocationMap.find(dstId);
            if (dstIt != buf.allocationMap.end())
            {
                dstIt->second->nodeId = UINT32_MAX;
                dstIt->second->isLocked = false;
                buf.allocationMap.erase(dstIt);
                buf.mergeFreeBlocks();
            }

            auto blockIt = srcIt->second;
            buf.allocationMap.erase(srcIt);

            // Update node identity
            blockIt->nodeId = dstId;
            buf.allocationMap[dstId] = blockIt;
        }
        else
        {
            Error::throw_err("[MemoryManager.transferOwnership] Source ID not found in allocation map");
        }
    }

    TensorView getView(const TensorNode &node, const CompiledGraph &compiled = {}) const
    {
        auto it = buffers.find(node.backend);
        if (it == buffers.end())
        {
            Error::throw_err("[MemoryManager.getView] Backend buffer not initialized");
        }

        const DeviceBuffer &buf = it->second;
        uint64_t arenaOffset = buf.getOffset(compiled.getLogicalId(node.id));

        // If the node already has view metadata (shape/strides), we use it.
        // Otherwise, we create a default contiguous view based on the node's shape.
        TensorView view;
        if (node.view.shape.empty())
        {
            view.baseOffset = arenaOffset;
            view.shape = node.shape;
            view.strides = TensorView::calcContiguousStrides(node.shape);
            view.dtype = node.dtype;
        }
        else
        {
            view = node.view;
            // The view's baseOffset stored in the node is usually relative to the
            // start of the allocation. We add the arena-relative offset here.
            view.baseOffset += arenaOffset;
        }

        return view;
    }

    /**
     * Creates a TensorView for a specific node and backend with a custom shape.
     * This is typically used when a node needs to be interpreted as a different shape
     * (like a reshape operation) or to initialize the view for a newly allocated input.
     *
     * @param backend The backend (device) where the tensor resides.
     * @param nodeId The unique identifier for the tensor node.
     * @param shape The desired shape for the view.
     * @return A TensorView containing the physical arena offset, shape, and contiguous strides.
     */
    TensorView getView(Backend backend, const uint32_t nodeId, std::vector<uint32_t> shape, DType dtype) const
    {
        // 1. Find the device-specific buffer
        auto it = buffers.find(backend);
        if (it == buffers.end())
        {
            std::stringstream ss;
            ss << "[MemoryManager.getView] Backend " << backend << " not initialized.";
            Error::throw_err(ss.str());
        }

        const DeviceBuffer &buf = it->second;

        // 2. Find the physical allocation block for this node
        auto allocIt = buf.allocationMap.find(nodeId);
        if (allocIt == buf.allocationMap.end())
        {
            std::stringstream ss;
            ss << "[MemoryManager.getView] Node ID " << nodeId << " is not currently allocated on " << backend;
            Error::throw_err(ss.str());
        }

        // 3. Construct the view
        TensorView view;
        // The baseOffset is the start of the block within the physical DeviceBuffer arena
        view.baseOffset = allocIt->second->offset;
        view.shape = std::move(shape);
        // Standard contiguous layout calculation (row-major)
        view.strides = TensorView::calcContiguousStrides(view.shape);
        view.dtype = dtype;

        return view;
    }

    bool has(Backend backend, uint32_t nodeId) const
    {
        const DeviceBuffer &buf = buffers.at(backend);
        return (buf.allocationMap.find(nodeId) != buf.allocationMap.end());
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
