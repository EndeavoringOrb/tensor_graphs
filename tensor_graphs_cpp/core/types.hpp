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
#include <json.hpp>
using json = nlohmann::json;

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

inline uint64_t getStridedIndex(uint64_t flatIndex, const std::vector<uint32_t> &shape, const std::vector<int64_t> &strides)
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
        Error::throw_err("Unknown DType size");
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
    CONTIGUOUS,
    SCATTER,

    FUSED
};

inline constexpr bool isAtomic(OpType type)
{
    return type != OpType::FUSED;
}

enum class Backend : uint32_t
{
    CPU,
    CUDA
};

enum class StorageType : uint32_t
{
    TRANSIENT,
    PERSISTENT
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
    uint64_t baseOffset = 0; // Offset into the MemoryManager's DeviceBuffer
    std::vector<uint32_t> shape;
    std::vector<int64_t> strides; // Strides in terms of elements, not bytes
    DType dtype;

    bool isContiguous() const
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
    std::string contentHash;
};

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
    Error::throw_err("Unknown dtype: " + str); // TODO: make this throw custom error, and catch for that instead of generic runtime_error
}

inline std::string toString(OpType op)
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
    case OpType::CONTIGUOUS:
        return "CONTIGUOUS";
    case OpType::SCATTER:
        return "SCATTER";
    case OpType::FUSED:
        return "FUSED";
    default:
        return "UNKNOWN_OP";
    }
}

inline std::string toString(Backend backend)
{
    switch (backend)
    {
    case Backend::CPU:
        return "CPU";
    case Backend::CUDA:
        return "CUDA";
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
    default:
        return "UNKNOWN_STORAGE";
    }
}

inline std::ostream &operator<<(std::ostream &os, DType dtype) { return os << toString(dtype); }
inline std::ostream &operator<<(std::ostream &os, OpType op) { return os << toString(op); }
inline std::ostream &operator<<(std::ostream &os, Backend backend) { return os << toString(backend); }
inline std::ostream &operator<<(std::ostream &os, StorageType storage) { return os << toString(storage); }

struct DirtyBucket
{
    std::unordered_map<uint32_t, std::vector<Region>> regions;
    std::unordered_map<uint32_t, std::vector<std::vector<Region>>> inputSlices;
};

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

NLOHMANN_JSON_SERIALIZE_ENUM(DType, {
                                        {DType::FLOAT32, "FLOAT32"},
                                        {DType::INT32, "INT32"},
                                        {DType::BF16, "BF16"},
                                        {DType::BOOL, "BOOL"},
                                    })

NLOHMANN_JSON_SERIALIZE_ENUM(OpType, {
                                         {OpType::INPUT, "INPUT"},
                                         {OpType::ADD, "ADD"},
                                         {OpType::MUL, "MUL"},
                                         {OpType::DIVIDE, "DIVIDE"},
                                         {OpType::DOT, "DOT"},
                                         {OpType::SIN, "SIN"},
                                         {OpType::COS, "COS"},
                                         {OpType::NEGATE, "NEGATE"},
                                         {OpType::POWER, "POWER"},
                                         {OpType::SUM, "SUM"},
                                         {OpType::MAX, "MAX"},
                                         {OpType::RESHAPE, "RESHAPE"},
                                         {OpType::PERMUTE, "PERMUTE"},
                                         {OpType::SLICE, "SLICE"},
                                         {OpType::CONCAT, "CONCAT"},
                                         {OpType::CAST, "CAST"},
                                         {OpType::REPEAT, "REPEAT"},
                                         {OpType::ARANGE, "ARANGE"},
                                         {OpType::TRIU, "TRIU"},
                                         {OpType::GATHER, "GATHER"},
                                         {OpType::FILL, "FILL"},
                                         {OpType::COPY_TO, "COPY_TO"},
                                         {OpType::IM2COL, "IM2COL"},
                                         {OpType::CONTIGUOUS, "CONTIGUOUS"},
                                         {OpType::SCATTER, "SCATTER"},
                                         {OpType::FUSED, "FUSED"},
                                     })

NLOHMANN_JSON_SERIALIZE_ENUM(Backend, {
                                          {Backend::CPU, "CPU"},
                                          {Backend::CUDA, "CUDA"},
                                      })

NLOHMANN_JSON_SERIALIZE_ENUM(StorageType, {
                                              {StorageType::TRANSIENT, "TRANSIENT"},
                                              {StorageType::PERSISTENT, "PERSISTENT"},
                                          })

struct OpInstruction
{
    uint32_t nodeId;
    uint64_t fullKernelId = 0;
    std::vector<uint64_t> cachedKernelIds;
    std::vector<uint32_t> inputNodeIds;
    int32_t inplaceInputIndex; // -1 if not inplace
    Backend backend;
};

struct CompiledGraph
{
    std::vector<OpInstruction> instructions;
    std::unordered_map<uint32_t, uint32_t> refCounts;
    std::unordered_map<uint32_t, TensorNode> nodesMap;
    std::unordered_map<uint32_t, float> nodeCosts;
    std::unordered_map<uint32_t, uint32_t> logicalNodeMap;
};

inline void to_json(json &j, const Dim &d) { j = json{d.start, d.stop}; }
inline void from_json(const json &j, Dim &d)
{
    d.start = j[0];
    d.stop = j[1];
}

inline void to_json(json &j, const TensorView &v)
{
    j = json{
        {"baseOffset", v.baseOffset},
        {"shape", v.shape},
        {"strides", v.strides},
        {"dtype", v.dtype}};
}
inline void from_json(const json &j, TensorView &v)
{
    v.baseOffset = j.at("baseOffset").get<uint64_t>();
    v.shape = j.at("shape").get<std::vector<uint32_t>>();
    v.strides = j.at("strides").get<std::vector<int64_t>>();
    v.dtype = j.at("dtype").get<DType>();
}

inline void to_json(json &j, const TensorNode &n)
{
    j = json{
        {"id", n.id},
        {"opType", n.opType},
        {"opName", n.opName},
        {"dtype", n.dtype},
        {"parentIds", n.parentIds},
        {"shape", n.shape},
        {"backend", n.backend},
        {"view", n.view},
        {"storageType", n.storageType},
        {"contentHash", n.contentHash}};
}
inline void from_json(const json &j, TensorNode &n)
{
    n.id = j.at("id").get<uint32_t>();
    n.opType = j.at("opType").get<OpType>();
    n.opName = j.at("opName").get<std::string>();
    n.dtype = j.at("dtype").get<DType>();
    n.parentIds = j.at("parentIds").get<std::vector<uint32_t>>();
    n.shape = j.at("shape").get<std::vector<uint32_t>>();
    n.backend = j.at("backend").get<Backend>();
    n.view = j.at("view").get<TensorView>();
    n.storageType = j.at("storageType").get<StorageType>();
    n.contentHash = j.at("contentHash").get<std::string>();
}

inline void to_json(json &j, const OpInstruction &i)
{
    json fullId;
    {
        std::stringstream pss;
        pss << "0x" << std::hex << i.fullKernelId;
        fullId = pss.str();
    }

    json cachedIds = json::array();
    for (uint64_t k : i.cachedKernelIds)
    {
        std::stringstream pss;
        pss << "0x" << std::hex << k;
        cachedIds.push_back(pss.str());
    }

    j = json{
        {"nodeId", i.nodeId},
        {"fullKernelId", fullId},
        {"cachedKernelIds", cachedIds},
        {"inputNodeIds", i.inputNodeIds},
        {"inplaceInputIndex", i.inplaceInputIndex},
        {"backend", i.backend}};
}
inline void from_json(const json &j, OpInstruction &i)
{
    i.nodeId = j.at("nodeId").get<uint32_t>();
    i.fullKernelId = std::stoull(j.at("fullKernelId").get<std::string>(), nullptr, 16);
    i.cachedKernelIds.clear();
    for (const auto &pkStr : j.at("cachedKernelIds"))
    {
        i.cachedKernelIds.push_back(std::stoull(pkStr.get<std::string>(), nullptr, 16));
    }

    i.inputNodeIds = j.at("inputNodeIds").get<std::vector<uint32_t>>();
    i.inplaceInputIndex = j.at("inplaceInputIndex").get<int32_t>();
    i.backend = j.at("backend").get<Backend>();
}
inline void to_json(json &j, const CompiledGraph &cg)
{
    json refCounts = json::object();
    for (const auto &kv : cg.refCounts)
        refCounts[std::to_string(kv.first)] = kv.second;

    json nodesMap = json::object();
    for (const auto &kv : cg.nodesMap)
        nodesMap[std::to_string(kv.first)] = kv.second;

    json nodeCosts = json::object();
    for (const auto &kv : cg.nodeCosts)
        nodeCosts[std::to_string(kv.first)] = kv.second;

    json logicalMap = json::object();
    for (const auto &kv : cg.logicalNodeMap)
        logicalMap[std::to_string(kv.first)] = kv.second;

    j = json{
        {"instructions", cg.instructions},
        {"refCounts", refCounts},
        {"nodesMap", nodesMap},
        {"nodeCosts", nodeCosts},
        {"logicalNodeMap", logicalMap}};
}

inline void from_json(const json &j, CompiledGraph &cg)
{
    cg.instructions = j.at("instructions").get<std::vector<OpInstruction>>();

    cg.refCounts.clear();
    for (const auto &item : j.at("refCounts").items())
        cg.refCounts[std::stoul(item.key())] = item.value().get<uint32_t>();

    cg.nodesMap.clear();
    for (const auto &item : j.at("nodesMap").items())
        cg.nodesMap[std::stoul(item.key())] = item.value().get<TensorNode>();

    cg.nodeCosts.clear();
    for (const auto &item : j.at("nodeCosts").items())
    {
        if (item.value().is_null())
        {
            cg.nodeCosts[std::stoul(item.key())] = std::numeric_limits<float>::infinity();
        }
        else
        {
            cg.nodeCosts[std::stoul(item.key())] = item.value().get<float>();
        }
    }

    cg.logicalNodeMap.clear();
    for (const auto &item : j.at("logicalNodeMap").items())
        cg.logicalNodeMap[std::stoul(item.key())] = item.value().get<uint32_t>();
}
