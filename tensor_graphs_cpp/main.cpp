#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <sstream>
#include <vector>

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
    uint32_t nodeId;
    StorageType storageType;
    uint32_t refCount;
    bool isLocked;
};

struct FreeSegment
{
    uint64_t start;
    uint64_t size = 0;

    bool operator<(const FreeSegment &other) const noexcept
    {
        return start < other.start;
    }

    bool valid()
    {
        return size != 0;
    }
};

struct DeviceBuffer
{
    std::vector<uint8_t> arena;
    uint64_t sizeBytes;
    std::unordered_map<uint32_t, MemBlock> allocations;
    std::vector<FreeSegment> freeSegments;

    DeviceBuffer(uint64_t _sizeBytes) : arena(_sizeBytes), sizeBytes(_sizeBytes)
    {
        freeSegments.push_back({0, _sizeBytes});
    }

    FreeSegment findFreeSlot(uint64_t _sizeBytes)
    {
        for (const auto &seg : freeSegments)
        {
            if (seg.size >= _sizeBytes)
            {
                return seg;
            }
        }
        return FreeSegment();
    }

    void tryEvict(uint64_t needed) {
        // knapsack-style eviction based on cumulative compute cost
    }

    uint64_t allocate(uint32_t nodeId, uint64_t _sizeBytes, StorageType storageType, int32_t refCount)
    {
        auto it = allocations.find(nodeId);
        if (it != allocations.end())
        {
            MemBlock &block = it->second;
            block.refCount = refCount;
            block.isLocked = true;
            return block.offset;
        }

        auto slot = findFreeSlot(_sizeBytes);
        if (!slot.valid()) {
            // tryEvict
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
