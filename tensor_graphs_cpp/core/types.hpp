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

enum class DType : uint32_t
{
    FLOAT32,
    INT32,
    BF16,
    _COUNT
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
    DType dtype;
    std::vector<uint32_t> parentIds;
    std::vector<uint32_t> shape;
    Backend backend = Backend::CPU;
    TensorView view;
};

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