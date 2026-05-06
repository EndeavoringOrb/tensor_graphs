#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct ENode
{
    uint64_t kernelUid = 0;
    OpType opType = OpType::INPUT;
    std::string opName;
    std::vector<uint32_t> children; // list of child eclass ids
    uint32_t leafId = UINT32_MAX;

    std::vector<uint32_t> shape;
    std::vector<uint64_t> strides;
    uint64_t viewOffset = 0;
    DType dtype = DType::FLOAT32;
    Backend backend = Backend::CPU;

    // Precomputed structural signature used by hashcons buckets.
    uint64_t sig = 0;

    bool operator==(const ENode &other) const
    {
        return kernelUid == other.kernelUid &&
               opType == other.opType &&
               leafId == other.leafId &&
               viewOffset == other.viewOffset &&
               dtype == other.dtype &&
               backend == other.backend &&
               opName == other.opName &&
               children == other.children &&
               shape == other.shape &&
               strides == other.strides;
    }
};

struct EClass
{
    uint32_t id = 0;
    std::vector<uint32_t> enodes;
    std::vector<uint32_t> shape;
    std::vector<uint64_t> strides;
    uint64_t viewOffset = 0;
    DType dtype = DType::FLOAT32;
    Backend backend = Backend::CPU;
};

class EGraph
{
public:
    uint32_t nextLeafId = 0;
    std::unordered_map<uint32_t, std::vector<uint8_t>> constantStaging;

    // Hash map for fast constant lookup: data hash -> list of class ids
    std::unordered_map<uint64_t, std::vector<uint32_t>> constantHashIndex;

    void reserve(size_t classCap, size_t nodeCap)
    {
        classes.reserve(classCap);
        parent.reserve(classCap);
        ufSize.reserve(classCap);

        enodes.reserve(nodeCap);
        nodeToEClass.reserve(nodeCap);
        hashcons.reserve(nodeCap * 2);
    }

    uint32_t getOrAddConstant(const std::vector<uint32_t> &shape,
                              const std::vector<uint64_t> &strides,
                              DType dtype,
                              Backend backend,
                              const std::vector<uint8_t> &data)
    {
        uint64_t dataHash = computeConstantHash(shape, strides, dtype, backend, data);

        auto it = constantHashIndex.find(dataHash);
        if (it != constantHashIndex.end())
        {
            for (uint32_t candidateClsId : it->second)
            {
                uint32_t clsId = find(candidateClsId);
                auto stagingIt = constantStaging.find(clsId);
                if (stagingIt == constantStaging.end())
                    continue;

                const EClass &cls = getEClass(clsId);
                if (cls.dtype == dtype && cls.backend == backend &&
                    cls.shape == shape && cls.strides == strides &&
                    stagingIt->second == data)
                {
                    return clsId;
                }
            }
        }

        uint32_t cls = addEClass(shape, strides, 0, dtype, backend);
        ENode n;
        n.kernelUid = 0;
        n.opType = OpType::INPUT;
        n.dtype = dtype;
        n.shape = shape;
        n.strides = strides;
        n.backend = backend;
        addENode(cls, n);
        constantStaging[cls] = data;
        constantHashIndex[dataHash].push_back(cls);
        return cls;
    }

    template <typename T>
    uint32_t getOrAddConstantData(const std::vector<uint32_t> &shape,
                                  DType dtype,
                                  Backend backend,
                                  const std::vector<T> &vals)
    {
        std::vector<uint64_t> strides = calcContiguousStrides(shape);
        std::vector<uint8_t> bytes(vals.size() * sizeof(T));
        std::memcpy(bytes.data(), vals.data(), bytes.size());
        return getOrAddConstant(shape, strides, dtype, backend, bytes);
    }

    uint32_t addEClass(const std::vector<uint32_t> &shape,
                       const std::vector<uint64_t> &strides,
                       uint64_t viewOffset,
                       DType dtype,
                       Backend backend)
    {
        uint32_t id = static_cast<uint32_t>(classes.size());

        EClass c;
        c.id = id;
        c.shape = shape;
        c.strides = strides;
        c.viewOffset = viewOffset;
        c.dtype = dtype;
        c.backend = backend;

        classes.push_back(std::move(c));
        parent.push_back(id);
        ufSize.push_back(1);
        return id;
    }

    uint32_t addENode(uint32_t eclassId, ENode node)
    {
        uint32_t canonical = find(eclassId);

        for (uint32_t &child : node.children)
        {
            child = find(child);
        }

        if (node.opType == OpType::INPUT && node.leafId == UINT32_MAX)
        {
            node.leafId = nextLeafId++;
        }

        node.sig = computeSignature(node);

        auto it = hashcons.find(node.sig);
        if (it != hashcons.end())
        {
            for (uint32_t otherEnodeId : it->second)
            {
                const ENode &other = enodes[otherEnodeId];
                if (node == other)
                {
                    merge(canonical, nodeToEClass[otherEnodeId]);
                    return find(canonical);
                }
            }
        }

        uint32_t enodeId = static_cast<uint32_t>(enodes.size());
        enodes.push_back(std::move(node));
        classes[canonical].enodes.push_back(enodeId);
        nodeToEClass.push_back(canonical);
        hashcons[enodes[enodeId].sig].push_back(enodeId);
        return canonical;
    }

    uint32_t find(uint32_t id)
    {
        uint32_t root = id;
        while (parent[root] != root)
        {
            root = parent[root];
        }

        while (parent[id] != id)
        {
            uint32_t p = parent[id];
            parent[id] = root;
            id = p;
        }

        return root;
    }

    uint32_t findConst(uint32_t id) const
    {
        while (parent[id] != id)
        {
            id = parent[id];
        }
        return id;
    }

    void merge(uint32_t a, uint32_t b)
    {
        uint32_t ra = find(a);
        uint32_t rb = find(b);
        if (ra == rb)
            return;

        // Union by size.
        if (ufSize[ra] < ufSize[rb])
            std::swap(ra, rb);

#ifdef DEBUG
        if (classes[ra].shape != classes[rb].shape)
        {
            Error::throw_err("EClass merge shape mismatch: " + toString(classes[ra].shape) + ", " + toString(classes[rb].shape));
        }
        if (classes[ra].strides != classes[rb].strides)
        {
            Error::throw_err("EClass merge strides mismatch: " + toString(classes[ra].strides) + ", " + toString(classes[rb].strides));
        }
        if (classes[ra].viewOffset != classes[rb].viewOffset)
        {
            Error::throw_err("EClass merge viewOffset mismatch: " + std::to_string(classes[ra].viewOffset) + ", " + std::to_string(classes[rb].viewOffset));
        }
        if (classes[ra].dtype != classes[rb].dtype)
        {
            Error::throw_err("EClass merge dtype mismatch: " + (std::string)toString(classes[ra].dtype) + ", " + toString(classes[rb].dtype));
        }
        if (classes[ra].backend != classes[rb].backend)
        {
            Error::throw_err("EClass merge backend mismatch: " + (std::string)toString(classes[ra].backend) + ", " + toString(classes[rb].backend));
        }
#endif

        parent[rb] = ra;
        ufSize[ra] += ufSize[rb];

        // Move constant staging from rb to ra to avoid losing constants
        auto itB = constantStaging.find(rb);
        if (itB != constantStaging.end())
        {
            if (constantStaging.find(ra) == constantStaging.end())
            {
                constantStaging[ra] = std::move(itB->second);
            }
            constantStaging.erase(itB);
        }

        classes[ra].enodes.reserve(classes[ra].enodes.size() + classes[rb].enodes.size());
        for (uint32_t enodeId : classes[rb].enodes)
        {
            classes[ra].enodes.push_back(enodeId);
            nodeToEClass[enodeId] = ra;
        }
        classes[rb].enodes.clear();
    }

    void rebuild()
    {
        std::unordered_map<uint64_t, std::vector<uint32_t>> newHash;
        newHash.reserve(enodes.size() * 2);
        uint32_t nDupes = 0;

        for (uint32_t i = 0, n = static_cast<uint32_t>(enodes.size()); i < n; ++i)
        {
            ENode &node = enodes[i];

            bool childrenChanged = false;
            for (uint32_t &child : node.children)
            {
                uint32_t c = find(child);
                if (c != child)
                {
                    child = c;
                    childrenChanged = true;
                }
            }

            uint32_t cls = find(nodeToEClass[i]);
            nodeToEClass[i] = cls;

            if (childrenChanged || node.sig == 0)
            {
                node.sig = computeSignature(node);
            }

            auto &bucket = newHash[node.sig];
            bool merged = false;

            for (uint32_t otherEnodeId : bucket)
            {
                const uint32_t otherCls = find(nodeToEClass[otherEnodeId]);
                if (node == enodes[otherEnodeId])
                {
                    nDupes++;
                    merge(otherCls, cls);
                    nodeToEClass[i] = find(otherCls);
                    merged = true;
                    break;
                }
            }

            if (!merged)
            {
                bucket.push_back(i);
            }
        }
        std::cout << "[EGraph.rebuild] Found " << nDupes << " duplicate enodes" << std::endl;

        hashcons = std::move(newHash);

        // Rebuild constant hash index to remove stale entries from merges
        rebuildConstantHashIndex();
    }

    const std::vector<EClass> &getClasses() const { return classes; }
    const std::vector<ENode> &getENodes() const { return enodes; }

    EClass &getEClass(uint32_t id) { return classes[find(id)]; }
    const EClass &getEClass(uint32_t id) const { return classes[findConst(id)]; }

    uint32_t getENodeEClass(uint32_t enodeId) const
    {
        return nodeToEClass[enodeId];
    }

private:
    static inline uint64_t mix64(uint64_t x) noexcept
    {
        x += 0x9e3779b97f4a7c15ull;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
        return x ^ (x >> 31);
    }

    static inline void hashCombine(uint64_t &h, uint64_t v) noexcept
    {
        h ^= mix64(v) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    }

    static uint64_t hashString(const std::string &s) noexcept
    {
        return std::hash<std::string>{}(s);
    }

    static uint64_t computeSignature(const ENode &node) noexcept
    {
        uint64_t h = mix64(node.kernelUid);

        hashCombine(h, static_cast<uint64_t>(node.opType));
        if (!node.opName.empty())
            hashCombine(h, hashString(node.opName));
        hashCombine(h, static_cast<uint64_t>(node.leafId));

        for (uint32_t c : node.children)
            hashCombine(h, static_cast<uint64_t>(c));

        for (uint32_t s : node.shape)
            hashCombine(h, static_cast<uint64_t>(s));

        for (uint64_t s : node.strides)
            hashCombine(h, s);

        hashCombine(h, node.viewOffset);
        hashCombine(h, static_cast<uint64_t>(node.dtype));
        hashCombine(h, static_cast<uint64_t>(node.backend));

        return h;
    }

    static uint64_t computeConstantHash(const std::vector<uint32_t> &shape,
                                        const std::vector<uint64_t> &strides,
                                        DType dtype,
                                        Backend backend,
                                        const std::vector<uint8_t> &data) noexcept
    {
        uint64_t h = static_cast<uint64_t>(dtype);
        hashCombine(h, static_cast<uint64_t>(backend));

        for (uint32_t s : shape)
            hashCombine(h, static_cast<uint64_t>(s));

        for (uint64_t s : strides)
            hashCombine(h, s);

        // Hash the data bytes efficiently - process 8 bytes at a time
        const uint8_t *ptr = data.data();
        size_t len = data.size();
        size_t i = 0;

        for (; i + 8 <= len; i += 8)
        {
            uint64_t val;
            std::memcpy(&val, ptr + i, 8);
            hashCombine(h, val);
        }

        // Handle remaining bytes
        if (i < len)
        {
            uint64_t val = 0;
            std::memcpy(&val, ptr + i, len - i);
            hashCombine(h, val);
        }

        return h;
    }

    void rebuildConstantHashIndex()
    {
        constantHashIndex.clear();
        for (const auto &kv : constantStaging)
        {
            uint32_t canonicalId = find(kv.first);
            if (canonicalId != kv.first)
                continue; // Skip non-canonical entries (data was moved during merge)

            const EClass &cls = getEClass(canonicalId);
            uint64_t h = computeConstantHash(cls.shape, cls.strides, cls.dtype, cls.backend, kv.second);
            constantHashIndex[h].push_back(canonicalId);
        }
    }

private:
    std::vector<EClass> classes;
    std::vector<ENode> enodes;
    std::vector<uint32_t> parent;
    std::vector<uint32_t> ufSize;

    // signature -> candidate enode ids
    std::unordered_map<uint64_t, std::vector<uint32_t>> hashcons;

    // Dense enodeId -> eclassId mapping.
    std::vector<uint32_t> nodeToEClass;
};

inline bool isContiguous(const EClass &eclass)
{
    return isContiguous(eclass.strides, eclass.shape);
}

inline std::string toString(const ENode &node)
{
    std::stringstream ss;
    ss << "ENode {\n"
       << "  KernelUID:  0x" << std::hex << node.kernelUid << std::dec << "\n"
       << "  OpType:     " << toString(node.opType) << "\n"
       << "  OpName:     " << (node.opName.empty() ? "N/A" : node.opName) << "\n"
       << "  Children:   [";
    for (size_t i = 0; i < node.children.size(); ++i)
    {
        ss << node.children[i] << (i == node.children.size() - 1 ? "" : ", ");
    }
    ss << "]\n"
       << "  LeafID:     " << (node.leafId == UINT32_MAX ? "N/A" : std::to_string(node.leafId)) << "\n"
       << "  Shape:      " << ::toString(node.shape) << "\n"
       << "  Strides:    " << ::toString(node.strides) << "\n"
       << "  ViewOffset: " << node.viewOffset << "\n"
       << "  DType:      " << ::toString(node.dtype) << "\n"
       << "  Backend:    " << ::toString(node.backend) << "\n"
       << "  Signature:  0x" << std::hex << node.sig << std::dec << "\n"
       << "}";
    return ss.str();
}

inline std::string toString(const EClass &cls)
{
    std::stringstream ss;
    ss << "EClass {\n"
       << "  ID:         " << cls.id << "\n"
       << "  Shape:      " << ::toString(cls.shape) << "\n"
       << "  Strides:    " << ::toString(cls.strides) << "\n"
       << "  ViewOffset: " << cls.viewOffset << "\n"
       << "  DType:      " << ::toString(cls.dtype) << "\n"
       << "  Backend:    " << ::toString(cls.backend) << "\n"
       << "  ENodes:     [";

    for (size_t i = 0; i < cls.enodes.size(); ++i)
    {
        ss << cls.enodes[i] << (i == cls.enodes.size() - 1 ? "" : ", ");
    }
    ss << "]\n"
       << "}";
    return ss.str();
}

inline std::ostream &operator<<(std::ostream &os, const EClass &cls) { return os << toString(cls); }