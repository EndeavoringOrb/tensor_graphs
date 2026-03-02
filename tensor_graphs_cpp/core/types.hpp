#pragma once
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <cstdint>
#include <sstream>
#include <fstream>
#include <vector>
#include <limits>
#include <cctype>
#include <list>
#include <json.hpp>
using json = nlohmann::json;

uint64_t countElements(std::vector<uint32_t> shape)
{
    uint64_t count = 1;
    for (uint32_t val : shape)
    {
        count *= val;
    }
    return count;
}

// When you add a new DType, remember to update getDTypeSize and toString(DType dtype)
enum class DType : uint32_t
{
    FLOAT32,
    INT32,
    BF16,
    BOOL,
    _COUNT
};

inline uint64_t getDTypeSize(DType dtype)
{
    switch (dtype)
    {
    case DType::FLOAT32:
        return 4;
    case DType::INT32:
        return 4;
    case DType::BF16:
        return 2;
    case DType::BOOL:
        return 1;
    default:
        throw std::runtime_error("Unknown DType size");
    }
}

enum class OpType : uint32_t
{
    INPUT,

    ADD,
    MUL,
    DIVIDE,
    DOT,
    SIN,
    COS,
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
    COPY_TO,
    IM2COL,

    FUSED
};

inline constexpr bool isAtomic(OpType type)
{
    return type != OpType::FUSED;
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

// ---------------------------------------------------------
// CUSTOM ERROR TYPES
// ---------------------------------------------------------

struct TensorGraphError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

struct ViewOpValidationError : public TensorGraphError
{
    uint32_t nodeId = 0;
    OpType opType;
    std::vector<uint32_t> shape;
    size_t dimIndex;

    ViewOpValidationError(const std::string &msg, uint32_t nid, OpType op,
                          const std::vector<uint32_t> &s, size_t dim = 0)
        : TensorGraphError(msg), nodeId(nid), opType(op), shape(s), dimIndex(dim) {}
};

struct ShapeMismatchError : public TensorGraphError
{
    uint32_t nodeId = 0;
    std::vector<uint32_t> expectedShape;
    std::vector<uint32_t> actualShape;

    ShapeMismatchError(const std::string &msg, uint32_t nid,
                       const std::vector<uint32_t> &expected,
                       const std::vector<uint32_t> &actual)
        : TensorGraphError(msg), nodeId(nid),
          expectedShape(expected), actualShape(actual) {}
};

struct MemoryAllocationError : public TensorGraphError
{
    uint64_t requestedSize = 0;

    MemoryAllocationError(const std::string &msg, uint64_t size)
        : TensorGraphError(msg), requestedSize(size) {}
};

struct Dim
{
    uint32_t start;
    uint32_t stop;
};

struct Region
{
    std::vector<Dim> region;
};

inline bool regionsMatch(const Region &r1, const Region &r2)
{
    if (r1.region.size() != r2.region.size())
        return false;
    for (size_t i = 0; i < r1.region.size(); ++i)
    {
        if (r1.region[i].start != r2.region[i].start ||
            r1.region[i].stop != r2.region[i].stop)
        {
            return false;
        }
    }
    return true;
}

struct TensorView
{
    uint64_t baseOffset; // Offset into the MemoryManager's DeviceBuffer
    std::vector<uint32_t> shape;
    std::vector<int64_t> strides; // Strides in terms of elements, not bytes

    // Check if the physical layout perfectly matches the logical layout
    bool isContiguous() const
    {
        int64_t expectedStride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
        {
            if (shape[i] == 1)
                continue; // Stride doesn't matter for size 1
            if (strides[i] != expectedStride)
                return false;
            expectedStride *= shape[i];
        }
        return true;
    }

    // Helper to generate default contiguous strides for a given shape
    static std::vector<int64_t> calcContiguousStrides(const std::vector<uint32_t> &targetShape)
    {
        std::vector<int64_t> newStrides(targetShape.size());
        int64_t stride = 1;
        for (int i = static_cast<int>(targetShape.size()) - 1; i >= 0; --i)
        {
            newStrides[i] = stride;
            stride *= targetShape[i];
        }
        return newStrides;
    }
};

struct TensorNode
{
    uint32_t id;
    OpType opType;
    std::string opName; // Used if opType == OpType::FUSED
    DType dtype;
    std::vector<uint32_t> parentIds;
    std::vector<uint32_t> shape;
    Backend backend = Backend::CPU;
    TensorView view;
    StorageType storageType = StorageType::TRANSIENT;
};

inline uint64_t getSizeBytes(const std::vector<uint32_t> &shape, DType dtype)
{
    return countElements(shape) * getDTypeSize(dtype);
}

inline const char *toString(DType dtype)
{
    switch (dtype)
    {
    case DType::FLOAT32:
        return "F32";
    case DType::INT32:
        return "I32";
    case DType::BF16:
        return "BF16";
    case DType::BOOL:
        return "BOOL";
    default:
        return "UNKNOWN_DTYPE";
    }
}

inline DType fromString(const std::string &str)
{
    for (uint32_t i = 0; i < static_cast<uint32_t>(DType::_COUNT); ++i)
    {
        DType dtype = static_cast<DType>(i);
        if (toString(dtype) == str)
            return dtype;
    }
    throw std::runtime_error("Unknown dtype: " + str); // TODO: make this throw custom error, and catch for that instead of generic runtime_error
}

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
    case OpType::SIN:
        return "SIN";
    case OpType::COS:
        return "COS";
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
    case OpType::COPY_TO:
        return "COPY_TO";
    case OpType::IM2COL:
        return "IM2COL";
    case OpType::FUSED:
        return "FUSED";
    default:
        return "UNKNOWN_OP";
    }
}

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

inline std::string toString(const std::vector<uint32_t> &shape)
{
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            ss << ", ";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

inline std::ostream &operator<<(std::ostream &os, DType dtype) { return os << toString(dtype); }
inline std::ostream &operator<<(std::ostream &os, OpType op) { return os << toString(op); }
inline std::ostream &operator<<(std::ostream &os, Backend backend) { return os << toString(backend); }
inline std::ostream &operator<<(std::ostream &os, StorageType storage) { return os << toString(storage); }

// ---------------------------------------------------------------------------
// DirtyBucket — one cached propagation result for a specific input region combo
// ---------------------------------------------------------------------------

struct DirtyBucket
{
    /*
    cached forward
    node id -> list of dirty output regions
    */
    std::unordered_map<uint32_t, std::vector<Region>> regions;

    /*
    cached backward
    Outer key is node ID. Inner vector is per-output-region, each entry is a vector of per-parent required regions.
    node id -> input slices
    input slices[output region idx][parent idx][dirty input region idx]
    */
    std::unordered_map<uint32_t, std::vector<std::vector<std::vector<Region>>>> inputSlices;
};