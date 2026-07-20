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

class ENode
{
public:
    ENode(KernelId kernelId,
          OpType opType,
          std::string opName,
          std::vector<EClassId> children,
          std::vector<uint32_t> shape,
          std::vector<uint64_t> strides,
          DType dtype,
          MemSpace mem_space,
          std::vector<Engine> engines,
          uint64_t sig = 0)
        : kernelId(kernelId),
          opType(opType),
          opName(std::move(opName)),
          children(std::move(children)),
          shape(std::move(shape)),
          strides(std::move(strides)),
          dtype(dtype),
          mem_space(mem_space),
          engines(engines),
          sig(sig)
    {
    }

    bool operator==(const ENode &other) const
    {
        return kernelId == other.kernelId &&
               opType == other.opType &&
               opName == other.opName &&
               children == other.children &&
               shape == other.shape &&
               strides == other.strides &&
               dtype == other.dtype &&
               mem_space == other.mem_space &&
               engines == other.engines;
    }

    // Getters for accessing the private fields
    KernelId getKernelId() const { return kernelId; }
    OpType getOpType() const { return opType; }
    const std::string &getOpName() const { return opName; }
    const std::vector<EClassId> &getChildren() const { return children; }
    const std::vector<uint32_t> &getShape() const { return shape; }
    const std::vector<uint64_t> &getStrides() const { return strides; }
    DType getDType() const { return dtype; }
    MemSpace getMemSpace() const { return mem_space; }
    std::vector<Engine> getEngines() const { return engines; }
    uint64_t getSig() const { return sig; }

private:
    KernelId kernelId;
    OpType opType;
    std::string opName;
    std::vector<EClassId> children; // list of child eclass ids
    std::vector<uint32_t> shape;
    std::vector<uint64_t> strides;
    DType dtype;
    MemSpace mem_space;
    std::vector<Engine> engines;

    uint64_t sig; // Precomputed structural signature used by hashcons buckets.
};

struct EClass
{
    EClassId id;
    std::vector<ENodeId> enodes;
    std::vector<uint32_t> shape;
    std::vector<uint64_t> strides;
    DType dtype;
    MemSpace mem_space;
};

struct EGraph
{
    std::vector<EClass> classes;
    std::vector<ENode> enodes;
    std::vector<EClassId> parent;
    std::vector<uint32_t> ufSize;

    // signature -> candidate enode ids
    std::unordered_map<uint64_t, std::vector<uint32_t>> hashcons;

    // Dense enodeId -> e_class_id mapping.
    std::vector<EClassId> nodeToEClass;

    uint32_t nextLeafId = 0;
    std::unordered_map<EClassId, std::shared_ptr<std::vector<uint8_t>>> constantStaging;

    // Hash map for fast constant lookup: data hash -> list of class ids
    std::unordered_map<uint64_t, std::vector<EClassId>> constantHashIndex;

    void reserve(size_t classCap, size_t nodeCap)
    {
        classes.reserve(classCap);
        parent.reserve(classCap);
        ufSize.reserve(classCap);

        enodes.reserve(nodeCap);
        nodeToEClass.reserve(nodeCap);
        hashcons.reserve(nodeCap * 2);
    }

    EClassId getOrAddConstant(const std::vector<uint32_t> &shape,
                              const std::vector<uint64_t> &strides,
                              DType dtype,
                              const std::vector<uint8_t> &data)
    {
        uint64_t dataHash = computeConstantHash(shape, strides, dtype, data);

        auto it = constantHashIndex.find(dataHash);
        if (it != constantHashIndex.end())
        {
            for (EClassId candidateClsId : it->second)
            {
                EClassId clsId = find(candidateClsId);
                auto stagingIt = constantStaging.find(clsId);
                if (stagingIt == constantStaging.end())
                    continue;

                const EClass &cls = getEClass(clsId);
                if (cls.dtype == dtype &&
                    cls.shape == shape && cls.strides == strides &&
                    *stagingIt->second == data)
                {
                    return clsId;
                }
            }
        }

        EClassId cls = addEClass(shape, strides, dtype, MemSpace(1, HandleType::CPP));
        ENode n = ENode(KernelId{0}, OpType::INPUT, "", {}, shape, strides, dtype, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)});
        addENode(cls, n);
        constantStaging[cls] = std::make_shared<std::vector<uint8_t>>(data);
        constantHashIndex[dataHash].push_back(cls);
        return cls;
    }

    template <typename T>
    EClassId getOrAddConstantData(const std::vector<uint32_t> &shape,
                                  DType dtype,
                                  const std::vector<T> &vals)
    {
        std::vector<uint64_t> strides = calcContiguousStrides(shape);
        std::vector<uint8_t> bytes(vals.size() * sizeof(T));
        std::memcpy(bytes.data(), vals.data(), bytes.size());
        return getOrAddConstant(shape, strides, dtype, bytes);
    }

    EClassId addEClass(const std::vector<uint32_t> &shape,
                       const std::vector<uint64_t> &strides,
                       DType dtype,
                       MemSpace mem_space)
    {
        EClassId id{classes.size()};

        EClass c;
        c.id = id;
        c.shape = shape;
        c.strides = strides;
        c.dtype = dtype;
        c.mem_space = mem_space;

        classes.push_back(std::move(c));
        parent.push_back(id);
        ufSize.push_back(1);
        return id;
    }

    uint32_t addENode(EClassId e_class_id, ENode node)
    {
        EClassId canonical = find(e_class_id);

        for (EClassId &child : node.getChildren())
        {
            child = find(child);
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

    EClassId find(EClassId id)
    {
        EClassId root = id;
        while (parent[root.value] != root)
        {
            root = parent[root.value];
        }

        while (parent[id.value] != id)
        {
            EClassId p = parent[id.value];
            parent[id.value] = root;
            id = p;
        }

        return root;
    }

    EClassId findConst(EClassId id) const
    {
        while (parent[id.value] != id)
        {
            id = parent[id.value];
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
    const ENode &getENode(ENodeId id) const {return enodes[id.value];}

    EClass &getEClass(EClassId id) { return classes[find(id)]; }
    const EClass &getEClass(EClassId id) const { return classes[findConst(id)]; }

    EClassId getENodeEClass(ENodeId enodeId) const
    {
        return nodeToEClass[enodeId.value];
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
        uint64_t h = mix64(node.kernelId);

        hashCombine(h, static_cast<uint64_t>(node.opType));
        if (!node.opName.empty())
            hashCombine(h, hashString(node.opName));

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
            uint64_t h = computeConstantHash(cls.shape, cls.strides, cls.dtype, *kv.second);
            constantHashIndex[h].push_back(canonicalId);
        }
    }
};

inline bool isContiguous(const EClass &eclass)
{
    return isContiguous(eclass.strides, eclass.shape);
}

inline std::string toString(const ENode &node)
{
    std::stringstream ss;
    ss << "ENode {\n"
       << "  KernelUID:  0x" << std::hex << node.kernelId << std::dec << "\n"
       << "  OpType:     " << toString(node.opType) << "\n"
       << "  OpName:     " << (node.opName.empty() ? "N/A" : node.opName) << "\n"
       << "  Children:   [";
    for (size_t i = 0; i < node.children.size(); ++i)
    {
        ss << node.children[i] << (i == node.children.size() - 1 ? "" : ", ");
    }
    ss << "]\n"
       << "  Shape:      " << ::toString(node.shape) << "\n"
       << "  Strides:    " << ::toString(node.strides) << "\n"
       << "  ViewOffset: " << node.viewOffset << "\n"
       << "  DType:      " << ::toString(node.dtype) << "\n"
       << "  Backend:    " << ::toString(node.backend) << "\n"
       << "  Signature:  0x" << std::hex << node.sig << std::dec << "\n"
       << "}";
    return ss.str();
}

inline std::string toString(const EClass &cls, const std::string &prefix = "")
{
    std::stringstream ss;
    ss << prefix << "EClass\n"
       << prefix << "  ID:         " << cls.id << "\n"
       << prefix << "  Shape:      " << ::toString(cls.shape) << "\n"
       << prefix << "  Strides:    " << ::toString(cls.strides) << "\n"
       << prefix << "  ViewOffset: " << cls.viewOffset << "\n"
       << prefix << "  DType:      " << ::toString(cls.dtype) << "\n"
       << prefix << "  Backend:    " << ::toString(cls.backend) << "\n"
       << prefix << "  ENodes:     [";

    for (size_t i = 0; i < cls.enodes.size(); ++i)
    {
        ss << cls.enodes[i] << (i == cls.enodes.size() - 1 ? "" : ", ");
    }
    ss << "]";
    return ss.str();
}

inline std::ostream &operator<<(std::ostream &os, const EClass &cls) { return os << toString(cls); }

inline std::string toString(const ENode &node, const EGraph &egraph, const std::string &prefix = "")
{
    std::stringstream ss;
    ss << prefix << "ENode [" << toString(node.opType);
    if (!node.opName.empty())
    {
        ss << " (" << node.opName << ")";
    }
    ss << "]\n"
       << prefix << "  DType:      " << toString(node.dtype) << "\n"
       << prefix << "  Shape:      " << toString(node.shape) << "\n"
       << prefix << "  Strides:    " << toString(node.strides) << "\n"
       << prefix << "  Backend:    " << toString(node.backend) << "\n"
       << prefix << "  ViewOffset: " << node.viewOffset << "\n"
       << prefix << "  Signature:  0x" << std::hex << node.sig << std::dec << "\n";

    if (node.kernelId != 0)
    {
        ss << prefix << "  KernelUID:  0x" << std::hex << node.kernelId << std::dec << "\n";
    }

    ss << prefix << "  Children (" << node.children.size() << "):";

    if (node.children.empty())
    {
        ss << " None";
    }
    else
    {
        for (size_t i = 0; i < node.children.size(); ++i)
        {
            uint32_t childClassId = node.children[i];
            // Resolve the canonical EClass from the graph
            const EClass &childCls = egraph.getEClass(childClassId);

            ss << "\n"
               << prefix << "    [" << i << "] EClass " << childClassId;

            // If the ID we have isn't the canonical one, note the redirect
            uint32_t canonicalId = egraph.findConst(childClassId);
            if (childClassId != canonicalId)
            {
                ss << " -> (Canonical: " << canonicalId << ")";
            }

            ss << "\n"
               << toString(childCls);
        }
    }
    return ss.str();
}