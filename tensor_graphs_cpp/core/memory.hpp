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
    int32_t refCount;
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

    // Sparse representation
    std::unordered_map<uint32_t, std::vector<uint8_t>> sparseData;

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
          initialized(other.initialized), sparseData(std::move(other.sparseData)),
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
            sparseData = std::move(other.sparseData);
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
        else
        {
            Error::throw_err("CUDA backend not supported without USE_CUDA");
        }
#endif
        if (backend == Backend::CPU)
        {
            cpu_arena.resize(sizeBytes);
            arena_ptr = cpu_arena.data();
        }
        else
        {
            throw std::runtime_error("CUDA backend not supported without USE_CUDA");
        }
#endif
        initialized = true;

        for (const auto &pair : sparseData)
        {
            uint32_t nodeId = pair.first;
            auto it = allocationMap.find(nodeId);
            if (it != allocationMap.end())
            {
#ifdef USE_CUDA
                if (backend == Backend::CUDA)
                {
                    cudaMemcpy(arena_ptr + it->second->offset, pair.second.data(), pair.second.size(), cudaMemcpyHostToDevice);
                }
                else
                {
                    std::memcpy(arena_ptr + it->second->offset, pair.second.data(), pair.second.size());
                }
#else
                std::memcpy(arena_ptr + it->second->offset, pair.second.data(), pair.second.size());
#endif
            }
        }
        sparseData.clear();
    }

    void write(uint32_t nodeId, const void *data, uint64_t size)
    {
        if (!initialized)
        {
            std::vector<uint8_t> buf(size);
            std::memcpy(buf.data(), data, size);
            sparseData[nodeId] = std::move(buf);
        }
        else
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
    }

    const uint8_t *read(uint32_t nodeId) const
    {
        if (!initialized)
        {
            auto it = sparseData.find(nodeId);
            if (it != sparseData.end())
            {
                return it->second.data();
            }
            return nullptr;
        }
        else
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
    }

    void unload(uint32_t nodeId)
    {
        if (!initialized)
        {
            sparseData.erase(nodeId);
        }
    }

    std::list<MemBlock>::iterator findFreeSlot(uint64_t _sizeBytes)
    {
        for (auto it = blocks.begin(); it != blocks.end(); ++it)
        {
            if (it->isFree() && it->sizeBytes >= _sizeBytes)
            {
                return it;
            }
        }
        return blocks.end();
    }

    bool tryEvict(uint64_t needed,
                  const std::unordered_map<uint32_t, std::vector<uint32_t>> *parentMap = nullptr,
                  const std::unordered_map<uint32_t, float> *nodeCosts = nullptr,
                  const std::unordered_set<uint32_t> &globalCacheState = {})
    {
        auto left = blocks.begin();
        auto right = blocks.begin();
        uint64_t currentSize = 0;

        auto bestLeft = blocks.end();
        auto bestRight = blocks.end();
        float bestCostLost = -1.0f;
        uint64_t bestSize = std::numeric_limits<uint64_t>::max();

        float currentTotalSaved = 0.0f;
        if (parentMap && nodeCosts)
        {
            currentTotalSaved = calculateSavedCost(globalCacheState, *parentMap, *nodeCosts);
        }

        while (right != blocks.end())
        {
            // Locked blocks act as an impenetrable wall. Reset the window.
            if (right->isLocked)
            {
                right++;
                left = right;
                currentSize = 0;
                continue;
            }

            currentSize += right->sizeBytes;

            // Once the window meets our size requirement, evaluate and shrink
            while (currentSize >= needed)
            {
                float costLost = 0.0f;
                if (parentMap && nodeCosts)
                {
                    std::unordered_set<uint32_t> windowNodes;
                    auto tempIt = left;
                    while (tempIt != std::next(right))
                    {
                        if (!tempIt->isFree())
                        {
                            windowNodes.insert(tempIt->nodeId);
                        }
                        tempIt++;
                    }

                    // Create hypothetical state simulating what remains if window is evicted
                    std::unordered_set<uint32_t> cacheStateWithoutWindow = globalCacheState;
                    for (uint32_t wn : windowNodes)
                    {
                        cacheStateWithoutWindow.erase(wn);
                    }

                    float newSaved = calculateSavedCost(cacheStateWithoutWindow, *parentMap, *nodeCosts);
                    costLost = currentTotalSaved - newSaved;
                }
                else
                {
                    auto tempIt = left;
                    while (tempIt != std::next(right))
                    {
                        if (!tempIt->isFree())
                            costLost += tempIt->cost;
                        tempIt++;
                    }
                }

                // Save it if it is strictly cheaper (we lose less cached compute), OR same cost but wastes less physical space
                if (bestCostLost < 0 || costLost < bestCostLost || (costLost == bestCostLost && currentSize < bestSize))
                {
                    bestCostLost = costLost;
                    bestSize = currentSize;
                    bestLeft = left;
                    bestRight = right;
                }

                // Break early if we've shrunk it to a single block
                if (left == right)
                    break;

                // Shrink from the left
                currentSize -= left->sizeBytes;
                left++;
            }
            right++;
        }

        if (bestLeft != blocks.end())
        {
            auto it = bestLeft;
            uint64_t mergeOffset = bestLeft->offset;
            auto end_evict = std::next(bestRight);

            while (it != end_evict)
            {
                if (!it->isFree())
                {
                    allocationMap.erase(it->nodeId);
                }
                it++;
            }

            MemBlock mergedFree;
            mergedFree.offset = mergeOffset;
            mergedFree.sizeBytes = bestSize;
            mergedFree.nodeId = UINT32_MAX;
            mergedFree.cost = 0.0f;
            mergedFree.isLocked = false;

            blocks.insert(bestLeft, mergedFree);
            blocks.erase(bestLeft, end_evict);

            return true;
        }

        return false;
    }

    uint64_t allocate(uint32_t nodeId, uint64_t _sizeBytes, StorageType storageType, int32_t refCount, float cost,
                      const std::unordered_map<uint32_t, std::vector<uint32_t>> *parentMap = nullptr,
                      const std::unordered_map<uint32_t, float> *nodeCosts = nullptr,
                      const std::unordered_set<uint32_t> &globalCacheState = {})
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

        // 3. If no space, try evict
        if (slotIt == blocks.end())
        {
            if (tryEvict(_sizeBytes, parentMap, nodeCosts, globalCacheState))
            {
                slotIt = findFreeSlot(_sizeBytes);
            }
        }

        // 4. If still no space, eviction failed.
        if (slotIt == blocks.end())
        {
            Error::throw_err<MemoryAllocationError>("Cannot allocate: Not enough contiguous space.", _sizeBytes);
        }

        // 5. Claim the free slot
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

struct InterruptManager
{
    static inline std::vector<DeviceBuffer *> buffers;
    static inline std::mutex mtx;

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

    static void cleanup(); // Implemented at the bottom of memory.hpp

    static void handleSigInt(int signum)
    {
        std::cerr << "\n[TensorGraph] Caught interrupt signal (" << signum << "). Cleaning up..." << std::endl;
        cleanup();
        std::exit(signum);
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

    std::unordered_set<uint32_t> getGlobalCacheState() const
    {
        std::unordered_set<uint32_t> state;
        for (const auto &pair : buffers)
        {
            for (const auto &alloc : pair.second.allocationMap)
            {
                state.insert(alloc.first);
            }
        }
        return state;
    }

    uint64_t allocate(Backend backend, uint32_t nodeId, uint64_t sizeBytes, StorageType storageType, int32_t refCount = 0, float cost = 0.0f, const std::unordered_map<uint32_t, std::vector<uint32_t>> *parentMap = nullptr, const std::unordered_map<uint32_t, float> *nodeCosts = nullptr)
    {
        auto it = buffers.find(backend);
        if (it == buffers.end())
            Error::throw_err("Backend buffer not initialized in MemoryManager");

        std::unordered_set<uint32_t> globalCacheState;
        if (parentMap && nodeCosts)
        {
            globalCacheState = getGlobalCacheState();
        }

        return it->second.allocate(nodeId, sizeBytes, storageType, refCount, cost, parentMap, nodeCosts, globalCacheState);
    }

    void write(Backend backend, uint32_t nodeId, const void *data, uint64_t size)
    {
        buffers.at(backend).write(nodeId, data, size);
    }

    const uint8_t *read(Backend backend, uint32_t nodeId) const
    {
        return buffers.at(backend).read(nodeId);
    }

    void unload(Backend backend, uint32_t nodeId)
    {
        buffers.at(backend).unload(nodeId);
    }

    void release(Backend backend, uint32_t nodeId)
    {
        auto &buf = buffers.at(backend);
        auto it = buf.allocationMap.find(nodeId);
        if (it != buf.allocationMap.end())
        {
            if (it->second->storageType == StorageType::TRANSIENT)
            {
                if (it->second->refCount > 0)
                {
                    it->second->refCount--;
                    // If no one else needs this node, unlock it for eviction
                    if (it->second->refCount == 0)
                    {
                        it->second->isLocked = false;
                    }
                }
            }
        }
    }

    void transferOwnership(Backend backend, uint32_t srcId, uint32_t dstId)
    {
        auto &buf = buffers.at(backend);
        auto srcIt = buf.allocationMap.find(srcId);
        if (srcIt != buf.allocationMap.end())
        {
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

    TensorView getView(const TensorNode &node) const
    {
        auto it = buffers.find(node.backend);
        if (it == buffers.end())
        {
            Error::throw_err("[MemoryManager.getView] Backend buffer not initialized");
        }

        const DeviceBuffer &buf = it->second;
        uint64_t arenaOffset = buf.getOffset(node.id);

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
};

// TODO: Can this be moved inside InterruptManager?
inline void InterruptManager::cleanup()
{
    std::lock_guard<std::mutex> lock(mtx);
    for (auto *buf : buffers)
    {
        buf->freeArena();
    }
    buffers.clear();
}