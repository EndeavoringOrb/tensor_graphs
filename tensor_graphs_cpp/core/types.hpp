#pragma once
#pragma once
#ifdef TG_USE_OPENCL
#include <CL/cl.h>
#else
typedef void *cl_mem;
typedef void *cl_context;
typedef void *cl_command_queue;
typedef void *cl_device_id;
typedef int cl_int;
typedef uint64_t cl_ulong;
typedef uint32_t cl_uint;
#endif

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <json.hpp>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/serialization.hpp"
using json = nlohmann::json;

// TODO: split up into types/tensor_node.hpp, types/...

// TODO: os & architecture detection should be in hardware.hpp and types should
// include hardware.hpp
// --- OS Detection ---
#if defined(_WIN32) || defined(_WIN64)
#define TG_OS_WINDOWS
#elif defined(__APPLE__)
#define TG_OS_MACOS
#elif defined(__linux__)
#define TG_OS_LINUX
#endif

// --- Architecture Detection ---
#if defined(__aarch64__) || defined(_M_ARM64)
#define TG_ARCH_ARM64
#if defined(__ARM_NEON) || defined(TG_OS_WINDOWS) // Windows ARM64 always has NEON
#define TG_HAS_NEON
#endif
#elif defined(__x86_64__) || defined(_M_X64)
#define TG_ARCH_X64
#endif

inline std::string toString(std::source_location loc)
{
    return std::string(loc.file_name()) + ":" + std::to_string(loc.line());
}

namespace Error
{
template <typename T = std::runtime_error, typename... Args>
[[noreturn]] inline void throw_err(const std::string &msg, Args &&...args,
                                   std::source_location loc = std::source_location::current())
{
    std::cerr << "\n[TensorGraph Error] (" << toString(loc) << ") " << msg << std::endl << std::flush;
    throw T(msg, std::forward<Args>(args)...);
}
} // namespace Error

enum class OpType : uint32_t
{
    INPUT,
    CACHE,

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
    UNPACK,
    REPEAT,
    ARANGE,
    TRIU,
    GATHER,
    FILL,
    COPY_TO, // Copy to another mem idx on the same HandleType
    IM2COL,
    CONTIGUOUS,
    SCATTER,
    LOG,
    ARGMAX,
    LT,
    EQ,
    AND,
    OR,
    NOT,

    FUSED
};

// When you add a new DType, remember to update getDTypeSize and toString(DType
// dtype)
enum class DType : uint32_t
{
    FLOAT32,
    INT32,
    INT64,
    BF16,
    BOOL,
    ANY,
    INT8,
    E2M1_PACKED_INT8,
    E2M1,
    F8_E8M0,
    F8_E4M3,
    _COUNT
};

struct BufferId
{
    uint32_t value = UINT32_MAX;

    auto operator<=>(const BufferId &) const = default;

    BufferId operator++(int)
    {
        BufferId temp = *this;
        ++value;
        return temp;
    }
};

struct LogicalId
{
    uint32_t value = UINT32_MAX;

    auto operator<=>(const LogicalId &) const = default;

    LogicalId operator++(int)
    {
        LogicalId temp = *this;
        ++value;
        return temp;
    }
};

class LogicalIdAllocator
{
  public:
    LogicalIdAllocator(const LogicalIdAllocator &) = delete;
    LogicalIdAllocator &operator=(const LogicalIdAllocator &) = delete;
    LogicalIdAllocator(LogicalIdAllocator &&) = delete;
    LogicalIdAllocator &operator=(LogicalIdAllocator &&) = delete;

    static LogicalId allocate()
    {
        return instance()._allocate();
    }

  private:
    LogicalIdAllocator() = default;
    ~LogicalIdAllocator() = default;

    static LogicalIdAllocator &instance()
    {
        static LogicalIdAllocator allocator;
        return allocator;
    }

    LogicalId _allocate()
    {
        return nextId++;
    }

    LogicalId nextId{0};
};

struct EClassId
{
    uint32_t value = UINT32_MAX;
    auto operator<=>(const EClassId &) const = default;
};

struct ENodeId
{
    uint32_t value = UINT32_MAX;
    auto operator<=>(const ENodeId &) const = default;
};

struct KernelId
{
    uint64_t value = UINT32_MAX;
    auto operator<=>(const KernelId &) const = default;
};

enum class HandleType : uint32_t
{
    STORAGE,
    CPP,
    OPENCL,
    CUDA
};

enum class EngineType : uint32_t
{
    CPU,
    QUALCOMM_IGPU,
    CUDA_GPU
};

struct MemSpace
{
    uint32_t idx;
    HandleType type;

    MemSpace() : idx(0), type(HandleType::STORAGE)
    {
    }
    MemSpace(uint32_t idx, HandleType type) : idx(idx), type(type)
    {
    }

    bool operator==(const MemSpace &other) const
    {
        return idx == other.idx && type == other.type;
    }
};

namespace tg_hash
{
inline uint64_t mix64(uint64_t x) noexcept
{
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

inline void hashCombine(uint64_t &h, uint64_t v) noexcept
{
    h ^= mix64(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
}

inline uint64_t computeConstantHash(const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides,
                                    DType dtype, const void *dataPtr, uint64_t sizeBytes) noexcept
{
    uint64_t h = static_cast<uint64_t>(dtype);

    for (uint32_t s : shape)
        hashCombine(h, static_cast<uint64_t>(s));

    for (uint64_t s : strides)
        hashCombine(h, s);

    const uint8_t *ptr = static_cast<const uint8_t *>(dataPtr);
    uint64_t i = 0;

    for (; i + 8 <= sizeBytes; i += 8)
    {
        uint64_t val;
        std::memcpy(&val, ptr + i, 8);
        hashCombine(h, val);
    }

    if (i < sizeBytes)
    {
        uint64_t val = 0;
        std::memcpy(&val, ptr + i, sizeBytes - i);
        hashCombine(h, val);
    }

    return h;
}

inline uint64_t computeConstantHash(const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides,
                                    DType dtype, const std::vector<uint8_t> &data) noexcept
{
    return computeConstantHash(shape, strides, dtype, data.data(), data.size());
}

inline uint64_t computeConstantHash(const std::vector<uint32_t> &shape, DType dtype, const void *dataPtr,
                                    uint64_t sizeBytes) noexcept
{
    static const std::vector<uint64_t> emptyStrides{};
    return computeConstantHash(shape, emptyStrides, dtype, dataPtr, sizeBytes);
}
} // namespace tg_hash

namespace std
{
template <> struct hash<LogicalId>
{
    std::uint64_t operator()(const LogicalId &id) const noexcept
    {
        return std::hash<uint32_t>()(id.value);
    }
};

template <> struct hash<EClassId>
{
    std::uint64_t operator()(const EClassId &id) const noexcept
    {
        return std::hash<uint32_t>()(id.value);
    }
};

template <> struct hash<ENodeId>
{
    std::uint64_t operator()(const ENodeId &id) const noexcept
    {
        return std::hash<uint32_t>()(id.value);
    }
};

template <> struct hash<KernelId>
{
    std::uint64_t operator()(const KernelId &id) const noexcept
    {
        return std::hash<uint64_t>()(id.value);
    }
};

template <> struct hash<BufferId>
{
    std::uint64_t operator()(const BufferId &id) const noexcept
    {
        return std::hash<uint32_t>()(id.value);
    }
};

template <> struct hash<MemSpace>
{
    uint64_t operator()(const MemSpace &ms) const noexcept
    {
        return std::hash<uint32_t>()(ms.idx) ^ (std::hash<uint32_t>()(static_cast<uint32_t>(ms.type)) << 1);
    }
};
} // namespace std

struct Engine
{
    uint32_t idx;
    EngineType type;
    std::unordered_set<MemSpace> supported;

    Engine() : idx(0), type(EngineType::CPU)
    {
    }
    Engine(uint32_t idx, EngineType type, std::unordered_set<MemSpace> supported = {})
        : idx(idx), type(type), supported(std::move(supported))
    {
    }

    bool operator==(const Engine &other) const
    {
        return idx == other.idx && type == other.type;
    }
};

inline uint32_t getDTypeNBits(DType dtype)
{
    switch (dtype)
    {
    case DType::FLOAT32:
        return 32;
    case DType::INT32:
        return 32;
    case DType::INT64:
        return 64;
    case DType::BF16:
        return 16;
    case DType::BOOL:
        return 8;
    case DType::ANY:
        return 0;
    case DType::INT8:
        return 8;
    case DType::E2M1_PACKED_INT8:
        return 8;
    case DType::E2M1:
        return 4;
    case DType::F8_E8M0:
        return 8;
    case DType::F8_E4M3:
        return 8;
    default:
        Error::throw_err("Unknown DType bits");
    }
}

inline uint64_t getDTypeSize(DType dtype)
{
    uint32_t bits = getDTypeNBits(dtype);
    return (bits + 7) / 8;
}

struct ParallelBuffer
{
    BufferId id;         // unique buf id
    MemSpace mem_space;  // which physical memory this buffer lives in
    uint64_t size = 0;   // bytes
    uint32_t start = 0;  // birth time (idx into dispatch order of first eclass that uses this)
    uint32_t end = 0;    // death time (idx into dispatch order of last eclass that uses this)
    int64_t offset = -1; // assigned byte offset, -1 = unallocated
};

struct Dim
{
    uint32_t start;
    uint32_t stop;
};

struct Region
{
    std::vector<Dim> region;

    bool empty() const
    {
        return region.empty();
    }
};

inline uint64_t countElements(const std::vector<uint32_t> &shape)
{
    uint64_t count = 1;
    for (uint32_t val : shape)
    {
        count *= val;
    }
    return count;
}

static std::vector<uint64_t> calcContiguousStrides(const std::vector<uint32_t> &targetShape)
{
    std::vector<uint64_t> newStrides(targetShape.size());
    uint64_t stride = 1;
    for (int i = static_cast<int>(targetShape.size()) - 1; i >= 0; --i)
    {
        newStrides[i] = stride;
        stride *= targetShape[i];
    }
    return newStrides;
}

struct TensorNode
{
  private:
    std::vector<uint32_t> shape;

  public:
    LogicalId id;
    OpType opType;
    std::string opName; // Used if opType == OpType::FUSED
    DType dtype;
    std::vector<LogicalId> child_ids;
    std::vector<uint64_t> strides;
    std::string contentHash;
    std::string debugOrigin;

    TensorNode()
    {
    }

    TensorNode(LogicalId _id, OpType _opType, std::string _opName, DType _dtype, std::vector<LogicalId> _child_ids,
               std::vector<uint32_t> _shape, std::vector<uint64_t> _strides, std::string _contentHash = "",
               std::string _debugOrigin = "")
        : id(_id), opType(_opType), opName(_opName), dtype(_dtype), child_ids(_child_ids), shape(_shape),
          strides(_strides), contentHash(_contentHash), debugOrigin(_debugOrigin)
    {
        if (strides.empty())
        {
            strides = calcContiguousStrides(shape);
        }
    }

    Region fullRegion() const
    {
        Region region = Region();
        for (const uint32_t dimSize : shape)
        {
            region.region.push_back({0, dimSize});
        }
        return region;
    }

    const std::vector<uint32_t> &getShape() const
    {
        return shape;
    }

    void setShape(const std::vector<uint32_t> &_shape)
    {
        shape = _shape;
        strides = calcContiguousStrides(_shape);
    }

    uint64_t getSizeBytes() const
    {
        return countElements(getShape()) * getDTypeSize(dtype);
    }
};

inline uint64_t countElements(const TensorNode &node)
{
    return countElements(node.getShape());
}

struct TensorView
{
  private:
    std::vector<uint32_t> shape;

  public:
    uint64_t offset = 0;           // Offset into the MemoryManager's DeviceBuffer
    std::vector<uint64_t> strides; // Strides in terms of elements, not bytes
    DType dtype;

    TensorView()
    {
    }
    TensorView(const std::vector<uint32_t> &_shape, const uint64_t _offset, const std::vector<uint64_t> &_strides,
               const DType &_dtype)
        : offset(_offset), shape(_shape), strides(_strides), dtype(_dtype)
    {
    }
    TensorView(const TensorNode &node, uint64_t _offset)
        : offset(_offset), shape(node.getShape()), strides(node.strides), dtype(node.dtype)
    {
    }

    const std::vector<uint32_t> &getShape() const
    {
        return shape;
    }

    void setShape(const std::vector<uint32_t> &_shape)
    {
        shape = _shape;
        if (strides.empty() || shape.size() != strides.size())
            strides = calcContiguousStrides(_shape);
    }
};

inline uint64_t countElements(const TensorView &view)
{
    return countElements(view.getShape());
}

struct GraphPatternCacheKey
{
    OpType pOpType;
    std::string pOpName;
    bool reference_only;
    bool ignore_output_mem_space;
    bool ignore_input_mem_spaces;
    bool ignore_engines;
    MemSpace output_mem_space;
    std::vector<MemSpace> input_mem_spaces;
    std::vector<Engine> engines;

    std::vector<TensorNode> inputs;
    TensorNode output;

    bool operator==(const GraphPatternCacheKey &o) const
    {
        if (pOpType != o.pOpType || pOpName != o.pOpName || reference_only != o.reference_only ||
            ignore_output_mem_space != o.ignore_output_mem_space ||
            ignore_input_mem_spaces != o.ignore_input_mem_spaces || ignore_engines != o.ignore_engines)
            return false;
        if (inputs.size() != o.inputs.size())
            return false;
        if (!ignore_output_mem_space && output_mem_space != o.output_mem_space)
            return false;
        if (!ignore_engines && engines != o.engines)
            return false;
        for (uint64_t i = 0; i < inputs.size(); ++i)
        {
            if ((!ignore_input_mem_spaces && i < input_mem_spaces.size() && i < o.input_mem_spaces.size() &&
                 input_mem_spaces[i] != o.input_mem_spaces[i]) ||
                inputs[i].dtype != o.inputs[i].dtype || inputs[i].getShape() != o.inputs[i].getShape() ||
                inputs[i].strides != o.inputs[i].strides)
                return false;
        }
        if (output.dtype != o.output.dtype || output.getShape() != o.output.getShape() ||
            output.strides != o.output.strides)
            return false;
        return true;
    }
};

namespace std
{
template <> struct hash<GraphPatternCacheKey>
{
    uint64_t operator()(const GraphPatternCacheKey &k) const noexcept
    {
        uint64_t h = 0;
        auto combine = [&](uint64_t val) { tg_hash::hashCombine(h, val); };

        combine(static_cast<uint64_t>(k.pOpType));
        if (!k.pOpName.empty())
            combine(std::hash<std::string>()(k.pOpName));
        combine(static_cast<uint64_t>(k.reference_only));
        combine(static_cast<uint64_t>(k.ignore_output_mem_space));
        combine(static_cast<uint64_t>(k.ignore_input_mem_spaces));
        combine(static_cast<uint64_t>(k.ignore_engines));

        if (!k.ignore_output_mem_space)
            combine(std::hash<MemSpace>()(k.output_mem_space));

        if (!k.ignore_engines)
        {
            for (const auto &eng : k.engines)
            {
                combine(static_cast<uint64_t>(eng.idx));
                combine(static_cast<uint64_t>(eng.type));
            }
        }

        for (uint64_t i = 0; i < k.inputs.size(); ++i)
        {
            if (!k.ignore_input_mem_spaces && i < k.input_mem_spaces.size())
                combine(std::hash<MemSpace>()(k.input_mem_spaces[i]));
            auto &in = k.inputs[i];
            combine(static_cast<uint64_t>(in.dtype));
            for (auto s : in.getShape())
                combine(static_cast<uint64_t>(s));
            for (auto s : in.strides)
                combine(static_cast<uint64_t>(s));
        }

        combine(static_cast<uint64_t>(k.output.dtype));
        for (auto s : k.output.getShape())
            combine(static_cast<uint64_t>(s));
        for (auto s : k.output.strides)
            combine(static_cast<uint64_t>(s));

        return h;
    }
};
} // namespace std

inline void tg_serialize(BinaryWriter &bw, const LogicalId &val)
{
    bw.write(val.value);
}
inline void tg_deserialize(BinaryReader &br, LogicalId &val)
{
    br.read(val.value);
}
inline void tg_serialize(BinaryWriter &bw, const KernelId &val)
{
    bw.write(val.value);
}
inline void tg_deserialize(BinaryReader &br, KernelId &val)
{
    br.read(val.value);
}
inline void tg_serialize(BinaryWriter &bw, const BufferId &val)
{
    bw.write(val.value);
}
inline void tg_deserialize(BinaryReader &br, BufferId &val)
{
    br.read(val.value);
}
inline void tg_serialize(BinaryWriter &bw, const EClassId &val)
{
    bw.write(val.value);
}
inline void tg_deserialize(BinaryReader &br, EClassId &val)
{
    br.read(val.value);
}
inline void tg_serialize(BinaryWriter &bw, const MemSpace &val)
{
    bw.write(val.idx);
    bw.write(val.type);
}
inline void tg_deserialize(BinaryReader &br, MemSpace &val)
{
    br.read(val.idx);
    br.read(val.type);
}
inline void tg_serialize(BinaryWriter &bw, const Engine &val)
{
    bw.write(val.idx);
    bw.write(val.type);
    bw.write(val.supported);
}
inline void tg_deserialize(BinaryReader &br, Engine &val)
{
    br.read(val.idx);
    br.read(val.type);
    br.read(val.supported);
}

inline void tg_serialize(BinaryWriter &bw, const ParallelBuffer &val)
{
    bw.write(val.id);
    bw.write(val.mem_space.idx);
    bw.write(static_cast<uint32_t>(val.mem_space.type));
    bw.write(val.size);
    bw.write(val.start);
    bw.write(val.end);
    bw.write(val.offset);
}
inline void tg_deserialize(BinaryReader &br, ParallelBuffer &val)
{
    br.read(val.id);
    br.read(val.mem_space.idx);
    uint32_t type;
    br.read(type);
    val.mem_space.type = static_cast<HandleType>(type);
    br.read(val.size);
    br.read(val.start);
    br.read(val.end);
    br.read(val.offset);
}

inline uint64_t getStridedIndex(uint64_t flatIndex, const std::vector<uint32_t> &shape,
                                const std::vector<uint64_t> &strides)
{
    uint64_t stridedIndex = 0;
    uint64_t temp = flatIndex;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        if (shape[i] > 1)
        {
            uint64_t coord = temp % shape[i];
            stridedIndex += coord * strides[i];
            temp /= shape[i];
        }
    }
    return stridedIndex;
}

inline bool operator==(DType a, DType b)
{
    return static_cast<uint32_t>(a) == static_cast<uint32_t>(b) ||
           static_cast<uint32_t>(a) == static_cast<uint32_t>(DType::ANY) ||
           static_cast<uint32_t>(b) == static_cast<uint32_t>(DType::ANY);
}

inline bool operator!=(DType a, DType b)
{
    return !(a == b);
}

inline constexpr bool isAtomic(OpType type)
{
    return type != OpType::FUSED;
}

struct TensorGraphError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

struct ViewOpValidationError : public TensorGraphError
{
    uint32_t nodeId = 0;
    OpType opType;
    std::vector<uint32_t> shape;
    uint64_t dimIndex;

    ViewOpValidationError(const std::string &msg, uint32_t nid, OpType op, const std::vector<uint32_t> &s,
                          uint64_t dim = 0)
        : TensorGraphError(msg), nodeId(nid), opType(op), shape(s), dimIndex(dim)
    {
    }
};

struct ShapeMismatchError : public TensorGraphError
{
    uint32_t nodeId = 0;
    std::vector<uint32_t> expectedShape;
    std::vector<uint32_t> actualShape;

    ShapeMismatchError(const std::string &msg, uint32_t nid, const std::vector<uint32_t> &expected,
                       const std::vector<uint32_t> &actual)
        : TensorGraphError(msg), nodeId(nid), expectedShape(expected), actualShape(actual)
    {
    }
};

struct MemoryAllocationError : public TensorGraphError
{
    uint64_t requestedSize = 0;

    MemoryAllocationError(const std::string &msg, uint64_t size) : TensorGraphError(msg), requestedSize(size)
    {
    }
};

struct MemoryExhaustedError : public std::runtime_error
{
    uint64_t requestedMemory;
    uint64_t availableMemory;

    MemoryExhaustedError(uint64_t requested, uint64_t available)
        : std::runtime_error("Memory exhausted: requested " + std::to_string(requested) + " bytes, available " +
                             std::to_string(available) + " bytes"),
          requestedMemory(requested), availableMemory(available)
    {
    }
};

inline bool operator==(const Region &a, const Region &b)
{
    if (a.region.size() != b.region.size())
    {
        return false;
    }
    for (int i = 0; i < a.region.size(); i++)
    {
        if (a.region[i].start != b.region[i].start)
        {
            return false;
        }
        if (a.region[i].stop != b.region[i].stop)
        {
            return false;
        }
    }
    return true;
}

inline bool operator<=(const Region &a, const Region &b)
{
    // Does b completely cover a?
    if (a.region.size() != b.region.size())
    {
        Error::throw_err("[Region<Region] cannot compare regions with sizes " + std::to_string(a.region.size()) +
                         " and " + std::to_string(b.region.size()));
    }
    bool covers = true;
    for (int i = 0; i < a.region.size(); i++)
    {
        covers = covers && (a.region[i].start >= b.region[i].start) && (a.region[i].stop <= b.region[i].stop);
    }
    return covers;
}

inline bool regionsMatch(const Region &r1, const Region &r2);

inline void tg_serialize(BinaryWriter &bw, const Dim &val)
{
    bw.write(val.start);
    bw.write(val.stop);
}
inline void tg_deserialize(BinaryReader &br, Dim &val)
{
    br.read(val.start);
    br.read(val.stop);
}

inline void tg_serialize(BinaryWriter &bw, const Region &val)
{
    bw.write(val.region);
}
inline void tg_deserialize(BinaryReader &br, Region &val)
{
    br.read(val.region);
}

inline std::string encodeRegion(const Region &r)
{
    std::stringstream ss;
    ss << "(";
    for (uint64_t i = 0; i < r.region.size(); ++i)
    {
        if (i > 0)
            ss << ",";
        ss << r.region[i].start << "-" << r.region[i].stop;
    }
    ss << ")";
    return ss.str();
}

inline std::vector<Region> normalizeRegions(std::vector<Region> regions)
{
    std::sort(regions.begin(), regions.end(), [](const Region &a, const Region &b) {
        if (a.region.size() != b.region.size())
            return a.region.size() < b.region.size();
        for (uint64_t i = 0; i < a.region.size(); ++i)
        {
            if (a.region[i].start != b.region[i].start)
                return a.region[i].start < b.region[i].start;
            if (a.region[i].stop != b.region[i].stop)
                return a.region[i].stop < b.region[i].stop;
        }
        return false;
    });

    regions.erase(std::unique(regions.begin(), regions.end(),
                              [](const Region &a, const Region &b) { return regionsMatch(a, b); }),
                  regions.end());
    return regions;
}

inline std::string encodeRegionList(const std::vector<Region> &regions)
{
    std::stringstream ss;
    const std::vector<Region> canonical = normalizeRegions(regions);
    for (const auto &r : canonical)
        ss << encodeRegion(r);
    return ss.str();
}

inline bool isContiguous(const std::vector<uint64_t> &strides, const std::vector<uint32_t> &shape)
{
    int64_t expectedStride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        if (strides[i] != expectedStride)
            return false;
        expectedStride *= shape[i];
    }
    return true;
}

inline bool isContiguous(const TensorNode &node)
{
    return isContiguous(node.strides, node.getShape());
}

inline bool isContiguous(const TensorView &view)
{
    return isContiguous(view.strides, view.getShape());
}

inline uint64_t getSizeBytes(const std::vector<uint32_t> &shape,
                             DType dtype) // TODO: redundant with TensorNode::getSizeBytes?
{
    return countElements(shape) * getDTypeSize(dtype);
}

template <typename T> inline std::string toString(const std::vector<T> &vec)
{
    std::stringstream ss;
    ss << "[";
    for (uint64_t i = 0; i < vec.size(); ++i)
    {
        ss << vec[i] << (i == vec.size() - 1 ? "" : ", ");
    }
    ss << "]";
    return ss.str();
}

inline std::string toString(DType dtype)
{
    switch (dtype)
    {
    case DType::FLOAT32:
        return "F32";
    case DType::INT32:
        return "I32";
    case DType::INT64:
        return "I64";
    case DType::BF16:
        return "BF16";
    case DType::BOOL:
        return "BOOL";
    case DType::ANY:
        return "ANY";
    case DType::INT8:
        return "I8";
    case DType::E2M1_PACKED_INT8:
        return "E2M1_PACKED_I8";
    case DType::E2M1:
        return "E2M1";
    case DType::F8_E8M0:
        return "F8_E8M0";
    case DType::F8_E4M3:
        return "F8_E4M3";
    default:
        return "UNKNOWN_DTYPE";
    }
}

inline DType fromString(const std::string &str)
{
    if (str == "F32")
        return DType::FLOAT32;
    if (str == "I32")
        return DType::INT32;
    if (str == "I64")
        return DType::INT64;
    if (str == "BF16")
        return DType::BF16;
    if (str == "BOOL")
        return DType::BOOL;
    if (str == "I8")
        return DType::INT8;
    if (str == "E2M1")
        return DType::E2M1;
    if (str == "E2M1_PACKED_I8")
        return DType::E2M1_PACKED_INT8;
    if (str == "F8_E8M0")
        return DType::F8_E8M0;
    if (str == "F8_E4M3")
        return DType::F8_E4M3;
    Error::throw_err("Unknown dtype: " + str);
}

inline std::string toString(OpType op) // TODO: make build.py check that each op has a case here
{
    switch (op)
    {
    case OpType::INPUT:
        return "INPUT";
    case OpType::CACHE:
        return "CACHE";
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
    case OpType::UNPACK:
        return "UNPACK";
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
    case OpType::CONTIGUOUS:
        return "CONTIGUOUS";
    case OpType::SCATTER:
        return "SCATTER";
    case OpType::LOG:
        return "LOG";
    case OpType::ARGMAX:
        return "ARGMAX";
    case OpType::LT:
        return "LT";
    case OpType::EQ:
        return "EQ";
    case OpType::AND:
        return "AND";
    case OpType::OR:
        return "OR";
    case OpType::NOT:
        return "NOT";
    case OpType::FUSED:
        return "FUSED";
    default:
        return "UNKNOWN_OP";
    }
}

inline std::string toString(HandleType handle)
{
    switch (handle)
    {
    case HandleType::STORAGE:
        return "STORAGE";
    case HandleType::CPP:
        return "CPP";
    case HandleType::CUDA:
        return "CUDA";
    case HandleType::OPENCL:
        return "OPENCL";
    default:
        return "UNKNOWN_HANDLE";
    }
}

inline std::string toString(EngineType engine)
{
    switch (engine)
    {
    case EngineType::CPU:
        return "CPU";
    case EngineType::CUDA_GPU:
        return "CUDA_GPU";
    case EngineType::QUALCOMM_IGPU:
        return "QUALCOMM_IGPU";
    default:
        return "UNKNOWN_ENGINE";
    }
}

inline std::string toString(LogicalId id)
{
    return "LogicalId(" + std::to_string(id.value) + ")";
}

inline std::string toString(KernelId id)
{
    return "KernelId(" + std::to_string(id.value) + ")";
}

inline std::string toString(EClassId id)
{
    return "EClassId(" + std::to_string(id.value) + ")";
}

inline std::string toString(ENodeId id)
{
    return "ENodeId(" + std::to_string(id.value) + ")";
}

inline std::string toString(BufferId id)
{
    return "BufferId(" + std::to_string(id.value) + ")";
}

inline std::ostream &operator<<(std::ostream &os, LogicalId id)
{
    return os << toString(id);
}
inline std::ostream &operator<<(std::ostream &os, KernelId id)
{
    return os << toString(id);
}
inline std::ostream &operator<<(std::ostream &os, EClassId id)
{
    return os << toString(id);
}
inline std::ostream &operator<<(std::ostream &os, ENodeId id)
{
    return os << toString(id);
}
inline std::ostream &operator<<(std::ostream &os, BufferId id)
{
    return os << toString(id);
}
inline std::ostream &operator<<(std::ostream &os, DType dtype)
{
    return os << toString(dtype);
}
inline std::ostream &operator<<(std::ostream &os, OpType op)
{
    return os << toString(op);
}
inline std::ostream &operator<<(std::ostream &os, HandleType handle_type)
{
    return os << toString(handle_type);
}
inline std::ostream &operator<<(std::ostream &os, EngineType engine_type)
{
    return os << toString(engine_type);
}

inline std::string toString(const MemSpace &mem_space)
{
    std::stringstream ss;
    ss << "MemSpace(idx=" << mem_space.idx << ", type=" << mem_space.type << ")";
    return ss.str();
}

inline std::string toString(const Engine &engine)
{
    std::stringstream ss;
    ss << "Engine(idx=" << engine.idx << ", type=" << engine.type << ")";
    return ss.str();
}

inline std::ostream &operator<<(std::ostream &os, MemSpace mem_space)
{
    return os << toString(mem_space);
}
inline std::ostream &operator<<(std::ostream &os, Engine engine)
{
    return os << toString(engine);
}

class SHA256
{
  private:
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t data[64];
    uint32_t datalen;

    static constexpr uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

    static inline uint32_t rotr(uint32_t x, uint32_t n)
    {
        return (x >> n) | (x << (32 - n));
    }
    static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z)
    {
        return (x & y) ^ (~x & z);
    }
    static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z)
    {
        return (x & y) ^ (x & z) ^ (y & z);
    }
    static inline uint32_t ep0(uint32_t x)
    {
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
    }
    static inline uint32_t ep1(uint32_t x)
    {
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
    }
    static inline uint32_t sig0(uint32_t x)
    {
        return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
    }
    static inline uint32_t sig1(uint32_t x)
    {
        return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
    }

    void transform()
    {
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4], f = state[5], g = state[6],
                 h = state[7];
        uint32_t w[64];

        for (int i = 0; i < 16; i++)
            w[i] = (static_cast<uint32_t>(data[i * 4]) << 24) | (static_cast<uint32_t>(data[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(data[i * 4 + 2]) << 8) | (static_cast<uint32_t>(data[i * 4 + 3]));
        for (int i = 16; i < 64; i++)
            w[i] = sig1(w[i - 2]) + w[i - 7] + sig0(w[i - 15]) + w[i - 16];

        for (int i = 0; i < 64; i++)
        {
            uint32_t t1 = h + ep1(e) + ch(e, f, g) + K[i] + w[i];
            uint32_t t2 = ep0(a) + maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

  public:
    SHA256()
    {
        state[0] = 0x6a09e667;
        state[1] = 0xbb67ae85;
        state[2] = 0x3c6ef372;
        state[3] = 0xa54ff53a;
        state[4] = 0x510e527f;
        state[5] = 0x9b05688c;
        state[6] = 0x1f83d9ab;
        state[7] = 0x5be0cd19;
        datalen = 0;
        bitlen = 0;
    }

    void update(const uint8_t *msg, uint64_t length)
    {
        for (uint64_t i = 0; i < length; i++)
        {
            data[datalen++] = msg[i];
            if (datalen == 64)
            {
                transform();
                bitlen += 512;
                datalen = 0;
            }
        }
    }

    void update(const std::string &str)
    {
        update(reinterpret_cast<const uint8_t *>(str.data()), str.length());
    }

    std::string digest()
    {
        uint64_t i = datalen;
        if (datalen < 56)
        {
            data[i++] = 0x80;
            while (i < 56)
                data[i++] = 0x00;
        }
        else
        {
            data[i++] = 0x80;
            while (i < 64)
                data[i++] = 0x00;
            transform();
            std::fill(std::begin(data), std::end(data), 0);
        }

        bitlen += datalen * 8;
        for (int i = 0; i < 8; ++i)
        {
            data[63 - i] = static_cast<uint8_t>(bitlen >> (i * 8));
        }
        transform();

        std::stringstream ss;
        ss << std::hex << std::setfill('0');
        for (int j = 0; j < 8; j++)
        {
            ss << std::setw(8) << state[j];
        }
        return ss.str();
    }
};

struct OpInstruction
{
    EClassId eclass_id;
    LogicalId logical_id;
    KernelId kernel_id;
    std::vector<EClassId> children;
    ParallelBuffer outBuffer;
    std::vector<ParallelBuffer> inBuffers;
    std::string debugOrigin;
};

struct Bucket
{
    std::unordered_map<LogicalId, std::vector<Region>> inputDirtyRegions;
    std::vector<Region> outputNeededRegion;
};

inline void tg_serialize(BinaryWriter &bw, const Bucket &val)
{
    uint32_t constSize = static_cast<uint32_t>(val.inputDirtyRegions.size());
    bw.write(constSize);
    for (const auto &pair : val.inputDirtyRegions)
    {
        bw.write(pair.first);
        // TODO: make a tg_serialize that handles any vector as long as it can
        // handle the type in the vector
        uint32_t regionsSize = static_cast<uint32_t>(pair.second.size());
        bw.write(regionsSize);
        for (const Region &r : pair.second)
        {
            bw.write(r);
        }
    }

    uint32_t outputSize = static_cast<uint32_t>(val.outputNeededRegion.size());
    bw.write(outputSize);
    for (const Region &r : val.outputNeededRegion)
    {
        bw.write(r);
    }
}
inline void tg_deserialize(BinaryReader &br, Bucket &val)
{
    uint32_t constSize;
    br.read(constSize);
    for (int i = 0; i < constSize; i++)
    {
        LogicalId first;
        br.read(first);
        uint32_t size;
        br.read(size);
        std::vector<Region> regs;
        for (int j = 0; j < size; j++)
        {
            Region r;
            br.read(r);
            regs.push_back(r);
        }
        val.inputDirtyRegions[first] = regs;
    }
    uint32_t outSize;
    br.read(outSize);
    std::vector<Region> outRegs;
    for (int j = 0; j < outSize; j++)
    {
        Region r;
        br.read(r);
        outRegs.push_back(r);
    }
    val.outputNeededRegion = outRegs;
}

inline bool operator==(const Bucket &a, const Bucket &b)
{
    bool equal = true;
    if (a.inputDirtyRegions.size() != b.inputDirtyRegions.size())
    {
        return false;
    }
    for (const auto &pair : a.inputDirtyRegions)
    {
        if (b.inputDirtyRegions.count(pair.first) == 0)
        {
            return false;
        }
        if (pair.second.size() != b.inputDirtyRegions.at(pair.first).size())
        {
            return false;
        }
        for (int i = 0; i < pair.second.size(); i++)
        {
            if (pair.second[i] != b.inputDirtyRegions.at(pair.first)[i])
            {
                return false;
            }
        }
    }
    if (a.outputNeededRegion.size() != b.outputNeededRegion.size())
    {
        return false;
    }
    for (int i = 0; i < a.outputNeededRegion.size(); i++)
    {
        if (a.outputNeededRegion[i] != b.outputNeededRegion[i])
        {
            return false;
        }
    }
    return true;
}

struct CompiledGraph
{
    Bucket bucket;
    std::unordered_map<EClassId, TensorView> nodeViews;
    std::vector<OpInstruction> instructions;
    std::unordered_map<EClassId, float> nodeCosts;
    // Canonical direction:
    // compiled physical node id -> original logical node id.
    std::unordered_map<EClassId, LogicalId> eclass_to_logical;
    std::unordered_map<EClassId, std::shared_ptr<std::vector<uint8_t>>> constantStaging;

    float cost() const
    {
        float sum = 0.0f;
        for (const auto &pair : nodeCosts)
        {
            sum += pair.second;
        }
        return sum;
    }

    bool has_logical_id(EClassId eclass_id) const
    {
        return eclass_to_logical.count(eclass_id) != 0;
    }

    LogicalId get_logical_id(EClassId eclass_id) const
    {
        auto it = eclass_to_logical.find(eclass_id);
        if (it != eclass_to_logical.end())
            return it->second;
        Error::throw_err("no logical id found for eclass_id " + toString(eclass_id));
    }
};

inline void tg_serialize(BinaryWriter &bw, const TensorView &val)
{
    bw.write(val.offset);
    bw.write(val.getShape());
    bw.write(val.strides);
    bw.write(val.dtype);
}
inline void tg_deserialize(BinaryReader &br, TensorView &val)
{
    br.read(val.offset);
    std::vector<uint32_t> shape;
    br.read(shape);
    val.setShape(shape);
    br.read(val.strides);
    br.read(val.dtype);
}

inline void tg_serialize(BinaryWriter &bw, const TensorNode &val)
{
    bw.write(val.id);
    bw.write(val.opType);
    bw.write(val.opName);
    bw.write(val.dtype);
    bw.write(val.child_ids);
    bw.write(val.getShape());
    bw.write(val.strides);
    bw.write(val.contentHash);
    bw.write(val.debugOrigin);
}

inline void tg_deserialize(BinaryReader &br, TensorNode &val)
{
    br.read(val.id);
    br.read(val.opType);
    br.read(val.opName);
    br.read(val.dtype);
    br.read(val.child_ids);
    std::vector<uint32_t> shape;
    br.read(shape);
    val.setShape(shape);
    br.read(val.strides);
    br.read(val.contentHash);
    br.read(val.debugOrigin);
}

inline void tg_serialize(BinaryWriter &bw, const OpInstruction &val)
{
    bw.write(val.eclass_id);
    bw.write(val.logical_id);
    bw.write(val.kernel_id);
    bw.write(val.children);
    bw.write(val.outBuffer);
    bw.write(val.inBuffers);
    bw.write(val.debugOrigin);
}

inline void tg_deserialize(BinaryReader &br, OpInstruction &val)
{
    br.read(val.eclass_id);
    br.read(val.logical_id);
    br.read(val.kernel_id);
    br.read(val.children);
    br.read(val.outBuffer);
    br.read(val.inBuffers);
    br.read(val.debugOrigin);
}

inline void tg_serialize(BinaryWriter &bw, const CompiledGraph &val)
{
    bw.write(val.bucket);
    bw.write(val.nodeViews);
    bw.write(val.instructions);
    bw.write(val.nodeCosts);
    bw.write(val.eclass_to_logical);
    uint32_t const_size = 0;
    for (const auto &pair : val.constantStaging)
    {
        if (pair.second)
            const_size++;
    }
    bw.write(const_size);
    for (const auto &pair : val.constantStaging)
    {
        if (pair.second)
        {
            bw.write(pair.first);
            bw.write(*pair.second);
        }
    }
}

inline void tg_deserialize(BinaryReader &br, CompiledGraph &val)
{
    br.read(val.bucket);
    br.read(val.nodeViews);
    br.read(val.instructions);
    br.read(val.nodeCosts);
    br.read(val.eclass_to_logical);
    uint32_t const_size;
    br.read(const_size);
    val.constantStaging.clear();
    for (uint32_t i = 0; i < const_size; ++i)
    {
        EClassId eclass_id;
        br.read(eclass_id);
        std::vector<uint8_t> data;
        br.read(data);
        val.constantStaging[eclass_id] = std::make_shared<std::vector<uint8_t>>(std::move(data));
    }
}

struct KernelContext
{
    std::vector<const void *> inputs;
    std::vector<void *> outputs;
    std::vector<TensorView> inViews;
    std::vector<TensorView> outViews;
    std::vector<int> fd;
    std::vector<cl_mem> cl_inputs;
    std::vector<cl_mem> cl_outputs;

    KernelContext()
    {
    }
    KernelContext(const std::vector<const void *> &_inputs, const std::vector<void *> &_outputs,
                  const std::vector<TensorView> &_inViews, const std::vector<TensorView> &_outViews)
        : inputs(_inputs), outputs(_outputs), inViews(_inViews), outViews(_outViews)
    {
        for (int i = 0; i < inputs.size(); i++)
        {
            fd.push_back(-1);
            cl_inputs.push_back(nullptr);
        }
        for (int i = 0; i < outputs.size(); i++)
        {
            cl_outputs.push_back(nullptr);
        }
    }
};