#pragma once
#include "core/types.hpp"

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

    // Arena not resized at construction so we can do allocation planning without actually making arena huge.
    void init()
    {
        arena.resize(sizeBytes);
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
                {
                    break;
                }

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
            throw std::runtime_error("Cannot allocate: Not enough contiguous space.");
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
};

struct MemoryManager
{
    std::unordered_map<Backend, DeviceBuffer> buffers;

    uint64_t allocate(Backend backend, uint32_t nodeId, uint64_t sizeBytes, StorageType storageType, int32_t refCount = 0, float cost = 0.0f)
    {
        auto it = buffers.find(backend);
        if (it == buffers.end())
        {
            throw std::runtime_error("Backend buffer not initialized in MemoryManager");
        }
        return it->second.allocate(nodeId, sizeBytes, storageType, refCount, cost);
    }
};
