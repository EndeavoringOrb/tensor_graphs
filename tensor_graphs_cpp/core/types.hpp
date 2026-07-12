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
#include <memory>
#include <list>
#include <algorithm>
#include <map>
#include <iomanip>
#include <json.hpp>
#include <source_location>
#include "core/serialization.hpp"
using json = nlohmann::json;

inline uint32_t GlobalNextPhysId = 0x80000000;

// TODO: os & architecture detection should be in hardware.hpp and types should include hardware.hpp
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

namespace Error
{
    template <typename T = std::runtime_error, typename... Args>
    [[noreturn]] inline void throw_err(const std::string &msg, Args &&...args)
    {
        std::cerr << "\n[TensorGraph Error] " << msg << std::endl
                  << std::flush;
        throw T(msg, std::forward<Args>(args)...);
    }
}

inline uint64_t getStridedIndex(uint64_t flatIndex, const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides)
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

inline uint64_t countElements(const std::vector<uint32_t> &shape)
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
    INT64,
    BF16,
    BOOL,
    ANY,
    _COUNT
};

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

inline uint64_t getDTypeSize(DType dtype)
{
    switch (dtype)
    {
    case DType::FLOAT32:
        return 4;
    case DType::INT32:
        return 4;
    case DType::INT64:
        return 8;
    case DType::BF16:
        return 2;
    case DType::BOOL:
        return 1;
    case DType::ANY:
        return 0;
    default:
        Error::throw_err("Unknown DType size");
    }
}

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
    REPEAT,
    ARANGE,
    TRIU,
    GATHER,
    FILL,
    COPY_TO,
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

inline constexpr bool isAtomic(OpType type)
{
    return type != OpType::FUSED;
}

enum class Backend : uint32_t
{
    STORAGE,
    CPU,
    CUDA,
    OPENCL,
    _COUNT
};

enum class StorageType : uint32_t
{
    TRANSIENT,
    PERSISTENT,
    PINNED // TODO: change to CACHE? or merge with PERSISTENT if OpType::CACHE is enough to differentiate?
};

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

struct MemoryExhaustedError : public std::runtime_error
{
    uint64_t requestedMemory;
    uint64_t availableMemory;

    MemoryExhaustedError(uint64_t requested, uint64_t available)
        : std::runtime_error("Memory exhausted: requested " + std::to_string(requested) +
                             " bytes, available " + std::to_string(available) + " bytes"),
          requestedMemory(requested), availableMemory(available) {}
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
        Error::throw_err("[Region<Region] cannot compare regions with sizes " + std::to_string(a.region.size()) + " and " + std::to_string(b.region.size()));
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
    for (size_t i = 0; i < r.region.size(); ++i)
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
    std::sort(regions.begin(), regions.end(), [](const Region &a, const Region &b)
              {
                  if (a.region.size() != b.region.size())
                      return a.region.size() < b.region.size();
                  for (size_t i = 0; i < a.region.size(); ++i)
                  {
                      if (a.region[i].start != b.region[i].start)
                          return a.region[i].start < b.region[i].start;
                      if (a.region[i].stop != b.region[i].stop)
                          return a.region[i].stop < b.region[i].stop;
                  }
                  return false; });

    regions.erase(std::unique(regions.begin(), regions.end(), [](const Region &a, const Region &b)
                              { return regionsMatch(a, b); }),
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
    uint32_t id;
    OpType opType;
    std::string opName; // Used if opType == OpType::FUSED
    DType dtype;
    std::vector<uint32_t> parentIds;
    std::vector<uint64_t> strides;
    uint64_t viewOffset = 0;
    Backend backend = Backend::CPU;
    StorageType storageType = StorageType::TRANSIENT;
    std::string contentHash;
    std::string debugOrigin;

    TensorNode() {}

    TensorNode(uint32_t _id, OpType _opType, std::string _opName, DType _dtype, std::vector<uint32_t> _parentIds, std::vector<uint32_t> _shape, std::vector<uint64_t> _strides, Backend _backend = Backend::CPU, StorageType _storageType = StorageType::PERSISTENT, std::string _contentHash = "", std::string _debugOrigin = "")
        : id(_id), opType(_opType), opName(_opName), dtype(_dtype), parentIds(_parentIds), shape(_shape), strides(_strides), backend(_backend), storageType(_storageType), contentHash(_contentHash), debugOrigin(_debugOrigin)
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
};

inline bool isContiguous(const TensorNode &node)
{
    return isContiguous(node.strides, node.getShape());
}

inline uint64_t countElements(const TensorNode &node)
{
    return countElements(node.getShape());
}

struct TensorView
{
private:
    std::vector<uint32_t> shape;

public:
    uint64_t baseOffset = 0;       // Offset into the MemoryManager's DeviceBuffer
    std::vector<uint64_t> strides; // Strides in terms of elements, not bytes
    DType dtype;

    TensorView() {}
    TensorView(const TensorNode &node, uint64_t _baseOffset) : baseOffset(_baseOffset), shape(node.getShape()), strides(node.strides), dtype(node.dtype) {}

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

inline bool isContiguous(const TensorView &view)
{
    return isContiguous(view.strides, view.getShape());
}

inline uint64_t countElements(const TensorView &view)
{
    return countElements(view.getShape());
}

inline uint64_t getSizeBytes(const std::vector<uint32_t> &shape, DType dtype)
{
    return countElements(shape) * getDTypeSize(dtype);
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
    Error::throw_err("Unknown dtype: " + str); // TODO: make this throw custom error, and catch for that instead of generic runtime_error
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

inline std::string toString(Backend backend) // TODO: make build.py check that each backend has a case here
{
    switch (backend)
    {
    case Backend::STORAGE:
        return "STORAGE";
    case Backend::CPU:
        return "CPU";
    case Backend::CUDA:
        return "CUDA";
    case Backend::OPENCL:
        return "OPENCL";
    default:
        return "UNKNOWN_BACKEND";
    }
}

inline std::string toString(StorageType storage)
{
    switch (storage)
    {
    case StorageType::TRANSIENT:
        return "TRANSIENT";
    case StorageType::PERSISTENT:
        return "PERSISTENT";
    case StorageType::PINNED:
        return "PINNED";
    default:
        return "UNKNOWN_STORAGE";
    }
}

inline std::ostream &operator<<(std::ostream &os, DType dtype) { return os << toString(dtype); }
inline std::ostream &operator<<(std::ostream &os, OpType op) { return os << toString(op); }
inline std::ostream &operator<<(std::ostream &os, Backend backend) { return os << toString(backend); }
inline std::ostream &operator<<(std::ostream &os, StorageType storage) { return os << toString(storage); }

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

    static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
    static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
    static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
    static inline uint32_t ep0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
    static inline uint32_t ep1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
    static inline uint32_t sig0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
    static inline uint32_t sig1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

    void transform()
    {
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3],
                 e = state[4], f = state[5], g = state[6], h = state[7];
        uint32_t w[64];

        for (int i = 0; i < 16; i++)
            w[i] = (static_cast<uint32_t>(data[i * 4]) << 24) |
                   (static_cast<uint32_t>(data[i * 4 + 1]) << 16) |
                   (static_cast<uint32_t>(data[i * 4 + 2]) << 8) |
                   (static_cast<uint32_t>(data[i * 4 + 3]));
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

    void update(const uint8_t *msg, size_t length)
    {
        for (size_t i = 0; i < length; i++)
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
    uint32_t nodeId;
    uint32_t logicalNodeId = UINT32_MAX;
    uint64_t fullKernelId = 0;
    std::vector<uint64_t> cachedKernelIds;
    std::vector<uint32_t> inputNodeIds;
    int32_t inplaceInputIndex = -1; // -1 if not inplace
    int32_t viewInputIndex = -1;    // -1 if not view
    Backend backend;
    StorageType outputStorageType = StorageType::TRANSIENT;
};

struct Bucket
{
    std::unordered_map<uint32_t, std::vector<Region>> inputDirtyRegions;
    std::vector<Region> outputNeededRegion;
};

inline void tg_serialize(BinaryWriter &bw, const Bucket &val)
{
    uint32_t constSize = static_cast<uint32_t>(val.inputDirtyRegions.size());
    bw.write(constSize);
    for (const auto &pair : val.inputDirtyRegions)
    {
        bw.write(pair.first);
        // TODO: make a tg_serialize that handles any vector as long as it can handle the type in the vector
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
        uint32_t first;
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
    std::vector<OpInstruction> instructions;
    std::unordered_map<uint32_t, uint32_t> refCounts;
    std::unordered_map<uint32_t, TensorNode> nodesMap;
    std::unordered_map<uint32_t, float> nodeCosts;
    // Canonical direction:
    // compiled physical node id -> original logical node id.
    std::unordered_map<uint32_t, uint32_t> physicalToLogicalNodeMap;
    std::unordered_map<uint32_t, std::shared_ptr<std::vector<uint8_t>>> constantStaging;

    float cost() const
    {
        float sum = 0.0f;
        for (const auto &pair : nodeCosts)
        {
            sum += pair.second;
        }
        return sum;
    }

    const uint32_t getLogicalId(uint32_t id) const
    {
        auto it = physicalToLogicalNodeMap.find(id);
        return it != physicalToLogicalNodeMap.end() ? it->second : id;
    }

    void remapPhysIds()
    {
        std::unordered_map<uint32_t, uint32_t> oldToNew;
        for (const auto &pair : nodesMap)
        {
            if (pair.first >= 0x80000000)
            {
                oldToNew[pair.first] = GlobalNextPhysId++;
            }
        }

        if (oldToNew.empty())
            return;

        auto mapId = [&](uint32_t id)
        {
            auto it = oldToNew.find(id);
            return it != oldToNew.end() ? it->second : id;
        };

        for (auto &inst : instructions)
        {
            inst.nodeId = mapId(inst.nodeId);
            for (auto &inId : inst.inputNodeIds)
            {
                inId = mapId(inId);
            }
        }

        std::unordered_map<uint32_t, uint32_t> newRefCounts;
        for (const auto &pair : refCounts)
        {
            newRefCounts[mapId(pair.first)] = pair.second;
        }
        refCounts = std::move(newRefCounts);

        std::unordered_map<uint32_t, TensorNode> newNodesMap;
        for (auto &pair : nodesMap)
        {
            TensorNode &node = pair.second;
            node.id = mapId(node.id);
            for (auto &pId : node.parentIds)
            {
                pId = mapId(pId);
            }
            newNodesMap[node.id] = std::move(node);
        }
        nodesMap = std::move(newNodesMap);

        std::unordered_map<uint32_t, float> newNodeCosts;
        for (const auto &pair : nodeCosts)
        {
            newNodeCosts[mapId(pair.first)] = pair.second;
        }
        nodeCosts = std::move(newNodeCosts);

        std::unordered_map<uint32_t, uint32_t> newPhysToLog;
        for (const auto &pair : physicalToLogicalNodeMap)
        {
            newPhysToLog[mapId(pair.first)] = pair.second;
        }
        physicalToLogicalNodeMap = std::move(newPhysToLog);

        std::unordered_map<uint32_t, std::shared_ptr<std::vector<uint8_t>>> newConst;
        for (auto &pair : constantStaging)
        {
            newConst[mapId(pair.first)] = std::move(pair.second);
        }
        constantStaging = std::move(newConst);
    }
};

inline void tg_serialize(BinaryWriter &bw, const TensorView &val)
{
    bw.write(val.baseOffset);
    bw.write(val.getShape());
    bw.write(val.strides);
    bw.write(val.dtype);
}
inline void tg_deserialize(BinaryReader &br, TensorView &val)
{
    br.read(val.baseOffset);
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
    bw.write(val.parentIds);
    bw.write(val.getShape());
    bw.write(val.strides);
    bw.write(val.viewOffset);
    bw.write(val.backend);
    bw.write(val.storageType);
    bw.write(val.contentHash);
    bw.write(val.debugOrigin);
}

inline void tg_deserialize(BinaryReader &br, TensorNode &val)
{
    br.read(val.id);
    br.read(val.opType);
    br.read(val.opName);
    br.read(val.dtype);
    br.read(val.parentIds);
    std::vector<uint32_t> shape;
    br.read(shape);
    val.setShape(shape);
    br.read(val.strides);
    br.read(val.viewOffset);
    br.read(val.backend);
    br.read(val.storageType);
    br.read(val.contentHash);
    br.read(val.debugOrigin);
}

inline void tg_serialize(BinaryWriter &bw, const OpInstruction &val)
{
    bw.write(val.nodeId);
    bw.write(val.logicalNodeId);
    bw.write(val.fullKernelId);
    bw.write(val.cachedKernelIds);
    bw.write(val.inputNodeIds);
    bw.write(val.inplaceInputIndex);
    bw.write(val.viewInputIndex);
    bw.write(val.backend);
    bw.write(val.outputStorageType);
}
inline void tg_deserialize(BinaryReader &br, OpInstruction &val)
{
    br.read(val.nodeId);
    br.read(val.logicalNodeId);
    br.read(val.fullKernelId);
    br.read(val.cachedKernelIds);
    br.read(val.inputNodeIds);
    br.read(val.inplaceInputIndex);
    br.read(val.viewInputIndex);
    br.read(val.backend);
    br.read(val.outputStorageType);
}

inline void tg_serialize(BinaryWriter &bw, const CompiledGraph &val)
{
    bw.write(val.bucket);
    bw.write(val.instructions);
    bw.write(val.refCounts);
    bw.write(val.nodesMap);
    bw.write(val.nodeCosts);
    bw.write(val.physicalToLogicalNodeMap);

    uint32_t constSize = static_cast<uint32_t>(val.constantStaging.size());
    bw.write(constSize);
    for (const auto &pair : val.constantStaging)
    {
        bw.write(pair.first);
        bw.write(*pair.second);
    }
}
inline void tg_deserialize(BinaryReader &br, CompiledGraph &val)
{
    br.read(val.bucket);
    br.read(val.instructions);
    br.read(val.refCounts);
    br.read(val.nodesMap);
    br.read(val.nodeCosts);
    br.read(val.physicalToLogicalNodeMap);

    uint32_t constSize;
    br.read(constSize);
    val.constantStaging.clear();
    for (uint32_t i = 0; i < constSize; ++i)
    {
        uint32_t k;
        br.read(k);
        std::vector<uint8_t> v;
        br.read(v);
        val.constantStaging[k] = std::make_shared<std::vector<uint8_t>>(std::move(v));
    }
}