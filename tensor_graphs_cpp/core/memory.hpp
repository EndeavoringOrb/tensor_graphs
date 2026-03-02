#pragma once
#include "core/types.hpp"
#include <vector>
#include <unordered_map>
#include <list>
#include <cstring>
#include <stdexcept>

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
    std::vector<uint8_t> arena;
    uint64_t sizeBytes;
    bool initialized = false;

    // Sparse representation for writing weights/constants before huge physical memory allocation
    std::unordered_map<uint32_t, std::vector<uint8_t>> sparseData;

    // Unified physical memory layout representation
    std::list<MemBlock> blocks;
    // O(1) lookup map pointing directly to the list node for fast updates
    std::unordered_map<uint32_t, std::list<MemBlock>::iterator> allocationMap;

    DeviceBuffer(uint64_t _sizeBytes) : sizeBytes(_sizeBytes)
    {
        // Initialize with one massive free block
        MemBlock initialFree;
        initialFree.offset = 0;
        initialFree.sizeBytes = _sizeBytes;
        initialFree.nodeId = UINT32_MAX;
        initialFree.cost = 0.0f;
        initialFree.isLocked = false;
        blocks.push_back(initialFree);
    }

    void init()
    {
        if (initialized)
            return;
        arena.resize(sizeBytes);
        initialized = true;

        // Copy over all sparse data to their allocated offsets in the arena
        for (const auto &pair : sparseData)
        {
            uint32_t nodeId = pair.first;
            auto it = allocationMap.find(nodeId);
            if (it != allocationMap.end())
            {
                std::memcpy(arena.data() + it->second->offset, pair.second.data(), pair.second.size());
            }
        }
        sparseData.clear(); // Free up redundant metadata storage
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
                std::memcpy(arena.data() + it->second->offset, data, size);
            }
            else
            {
                throw std::runtime_error("Cannot write to unallocated node");
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
                return arena.data() + it->second->offset;
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

    bool tryEvict(uint64_t needed)
    {
        auto left = blocks.begin();
        auto right = blocks.begin();
        uint64_t currentSize = 0;
        float currentCost = 0.0f;

        auto bestLeft = blocks.end();
        auto bestRight = blocks.end();
        float bestCost = std::numeric_limits<float>::infinity();
        uint64_t bestSize = std::numeric_limits<uint64_t>::max();

        while (right != blocks.end())
        {
            // Locked blocks act as an impenetrable wall. Reset the window.
            if (right->isLocked)
            {
                right++;
                left = right;
                currentSize = 0;
                currentCost = 0.0f;
                continue;
            }

            currentSize += right->sizeBytes;
            currentCost += right->cost;

            // Once the window meets our size requirement, evaluate and shrink
            while (currentSize >= needed)
            {
                // Save it if it is strictly cheaper, OR same cost but wastes less physical space
                if (currentCost < bestCost || (currentCost == bestCost && currentSize < bestSize))
                {
                    bestCost = currentCost;
                    bestSize = currentSize;
                    bestLeft = left;
                    bestRight = right;
                }

                // Break early if we've shrunk it to a single block
                if (left == right)
                    break;

                // Shrink from the left
                currentSize -= left->sizeBytes;
                currentCost -= left->isFree() ? 0.0f : left->cost;
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

    // float cost is the cumulative cost to recompute. TODO: when node is evicted, update its children's cost
    uint64_t allocate(uint32_t nodeId, uint64_t _sizeBytes, StorageType storageType, int32_t refCount, float cost)
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
            if (tryEvict(_sizeBytes))
            {
                slotIt = findFreeSlot(_sizeBytes);
            }
        }

        // 4. If still no space, eviction failed.
        if (slotIt == blocks.end())
        {
            throw MemoryAllocationError("Cannot allocate: Not enough contiguous space.", _sizeBytes);
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
            throw std::runtime_error("[DeviceBuffer.getOffset] Node " + std::to_string(nodeId) + " not found in allocation map");
        }
        return it->second->offset;
    }
};

struct MemoryManager
{
    std::unordered_map<Backend, DeviceBuffer> buffers;

    uint64_t allocate(Backend backend, uint32_t nodeId, uint64_t sizeBytes, StorageType storageType, int32_t refCount = 0, float cost = 0.0f)
    {
        auto it = buffers.find(backend);
        if (it == buffers.end())
            throw std::runtime_error("Backend buffer not initialized in MemoryManager");
        return it->second.allocate(nodeId, sizeBytes, storageType, refCount, cost);
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
            throw std::runtime_error("[MemoryManager.transferOwnership] Source ID not found in allocation map");
        }
    }

    TensorView getView(const TensorNode &node) const
    {
        auto it = buffers.find(node.backend);
        if (it == buffers.end())
        {
            throw std::runtime_error("[MemoryManager.getView] Backend buffer not initialized");
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
    TensorView getView(Backend backend, const uint32_t nodeId, std::vector<uint32_t> shape) const
    {
        // 1. Find the device-specific buffer
        auto it = buffers.find(backend);
        if (it == buffers.end())
        {
            std::stringstream ss;
            ss << "[MemoryManager.getView] Backend " << backend << " not initialized.";
            throw std::runtime_error(ss.str());
        }

        const DeviceBuffer &buf = it->second;

        // 2. Find the physical allocation block for this node
        auto allocIt = buf.allocationMap.find(nodeId);
        if (allocIt == buf.allocationMap.end())
        {
            std::stringstream ss;
            ss << "[MemoryManager.getView] Node ID " << nodeId << " is not currently allocated on " << backend;
            throw std::runtime_error(ss.str());
        }

        // 3. Construct the view
        TensorView view;
        // The baseOffset is the start of the block within the physical DeviceBuffer arena
        view.baseOffset = allocIt->second->offset;
        view.shape = std::move(shape);
        // Standard contiguous layout calculation (row-major)
        view.strides = TensorView::calcContiguousStrides(view.shape);

        return view;
    }
};