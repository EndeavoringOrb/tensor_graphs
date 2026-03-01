#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <sstream>
#include <vector>
#include <limits>
#include <list>

enum class DType : uint32_t
{
    FLOAT32,
    INT32
};

enum class OpType : uint32_t
{
    INPUT,

    ADD,
    MUL,
    DIVIDE,
    DOT,
    SQRT,
    SIN,
    COS,
    EXP,
    NEGATE,
    POWER,
    SUM,
    MAX,
    RESHAPE,
    PERMUTE,
    SLICE,
    CONCAT,
    CAST,
    REPEAT,
    ARANGE,
    TRIU,
    GATHER,
    FILL,
    WHERE,
    COPY_TO,
    IM2COL,

    CONV2D,
    FMA,
    GELU,
    GROUPNORM,
    RMS_NORM,
    ROPE_2D_CONSECUTIVE,
    ROPE,
    SIGMOID,
    SILU,
    SOFTMAX,
    TANH,
    UPSAMPLE_NEAREST
};

inline constexpr bool isAtomic(OpType type)
{
    return static_cast<uint32_t>(type) < static_cast<uint32_t>(OpType::CONV2D);
}

enum class Backend : uint32_t
{
    CPU
};

enum class StorageType : uint32_t
{
    TRANSIENT,
    PERSISTENT
};

struct Dim
{
    uint32_t start;
    uint32_t stop;
};

struct DirtyRegion
{
    std::vector<Dim> region;
};

struct TensorValue
{
    std::vector<uint8_t> bytes;
};

struct MemRecord
{
    uint64_t offset;
    uint64_t size;
    StorageType storageType;
};

struct TensorNode
{
    uint32_t id;
    OpType opType;
    DType dtype;
    std::vector<uint32_t> parentIds;
    std::vector<uint32_t> shape;
    Backend backend = Backend::CPU;
    std::vector<DirtyRegion> dirtyRegions;
    MemRecord mem;
};

// DType conversion
inline const char *toString(DType dtype)
{
    switch (dtype)
    {
    case DType::FLOAT32:
        return "FLOAT32";
    case DType::INT32:
        return "INT32";
    default:
        return "UNKNOWN_DTYPE";
    }
}

// OpType conversion
inline const char *toString(OpType op)
{
    switch (op)
    {
    case OpType::INPUT:
        return "INPUT";
    case OpType::ADD:
        return "ADD";
    case OpType::MUL:
        return "MUL";
    case OpType::DIVIDE:
        return "DIVIDE";
    case OpType::DOT:
        return "DOT";
    case OpType::SQRT:
        return "SQRT";
    case OpType::SIN:
        return "SIN";
    case OpType::COS:
        return "COS";
    case OpType::EXP:
        return "EXP";
    case OpType::NEGATE:
        return "NEGATE";
    case OpType::POWER:
        return "POWER";
    case OpType::SUM:
        return "SUM";
    case OpType::MAX:
        return "MAX";
    case OpType::RESHAPE:
        return "RESHAPE";
    case OpType::PERMUTE:
        return "PERMUTE";
    case OpType::SLICE:
        return "SLICE";
    case OpType::CONCAT:
        return "CONCAT";
    case OpType::CAST:
        return "CAST";
    case OpType::REPEAT:
        return "REPEAT";
    case OpType::ARANGE:
        return "ARANGE";
    case OpType::TRIU:
        return "TRIU";
    case OpType::GATHER:
        return "GATHER";
    case OpType::FILL:
        return "FILL";
    case OpType::WHERE:
        return "WHERE";
    case OpType::COPY_TO:
        return "COPY_TO";
    case OpType::IM2COL:
        return "IM2COL";
    case OpType::CONV2D:
        return "CONV2D";
    case OpType::FMA:
        return "FMA";
    case OpType::GELU:
        return "GELU";
    case OpType::GROUPNORM:
        return "GROUPNORM";
    case OpType::RMS_NORM:
        return "RMS_NORM";
    case OpType::ROPE_2D_CONSECUTIVE:
        return "ROPE_2D_CONSECUTIVE";
    case OpType::ROPE:
        return "ROPE";
    case OpType::SIGMOID:
        return "SIGMOID";
    case OpType::SILU:
        return "SILU";
    case OpType::SOFTMAX:
        return "SOFTMAX";
    case OpType::TANH:
        return "TANH";
    case OpType::UPSAMPLE_NEAREST:
        return "UPSAMPLE_NEAREST";
    default:
        return "UNKNOWN_OPS";
    }
}

// Backend conversion
inline const char *toString(Backend backend)
{
    switch (backend)
    {
    case Backend::CPU:
        return "CPU";
    default:
        return "UNKNOWN_BACKEND";
    }
}

// StorageType conversion
inline const char *toString(StorageType storage)
{
    switch (storage)
    {
    case StorageType::TRANSIENT:
        return "TRANSIENT";
    case StorageType::PERSISTENT:
        return "PERSISTENT";
    default:
        return "UNKNOWN_STORAGE";
    }
}

inline std::ostream &operator<<(std::ostream &os, DType dtype)
{
    return os << toString(dtype);
}

inline std::ostream &operator<<(std::ostream &os, OpType op)
{
    return os << toString(op);
}

inline std::ostream &operator<<(std::ostream &os, Backend backend)
{
    return os << toString(backend);
}

inline std::ostream &operator<<(std::ostream &os, StorageType storage)
{
    return os << toString(storage);
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
    std::vector<uint8_t> arena;
    uint64_t sizeBytes;

    // Unified physical memory layout representation
    std::list<MemBlock> blocks;
    // O(1) lookup map pointing directly to the list node for fast updates
    std::unordered_map<uint32_t, std::list<MemBlock>::iterator> allocationMap;

    DeviceBuffer(uint64_t _sizeBytes) : arena(_sizeBytes), sizeBytes(_sizeBytes)
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

            // Unregister evicted blocks from O(1) map
            while (it != end_evict)
            {
                if (!it->isFree())
                {
                    allocationMap.erase(it->nodeId);
                }
                it++;
            }

            // Replace the evicted sequence with a single continuous Free block
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
};

struct Graph
{
    uint32_t count = 0;
    std::vector<TensorNode> nodes;

    uint32_t allocateId() noexcept { return count++; }

    uint32_t input(std::vector<uint32_t> shape, DType dtype, MemRecord source)
    {
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::INPUT;
        node.dtype = dtype;
        node.shape = shape;
        node.mem = source;
        nodes.push_back(node);
        return node.id;
    }

    uint32_t add(uint32_t id0, uint32_t id1)
    {
        if (nodes[id0].dtype != nodes[id1].dtype)
        {
            std::stringstream ss;
            ss << "DType mismatch in add: " << nodes[id0].dtype << ", " << nodes[id1].dtype;
            throw std::runtime_error(ss.str());
        }
        TensorNode node = TensorNode();
        node.id = allocateId();
        node.opType = OpType::ADD;
        node.dtype = nodes[id0].dtype;
        node.parentIds = {id0, id1};
        nodes.push_back(node);
        return node.id;
    }
};

int main()
{
    uint32_t maxSeqLen = 128;
}
