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
          std::string contentHash = "",
          uint64_t sig = 0)
        : kernelId(kernelId),
          opType(opType),
          opName(std::move(opName)),
          children(std::move(children)),
          shape(std::move(shape)),
          strides(std::move(strides)),
          dtype(dtype),
          mem_space(mem_space),
          engines(std::move(engines)),
          contentHash(std::move(contentHash)),
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
               engines == other.engines &&
               contentHash == other.contentHash;
    }

    // Read-only getters
    KernelId getKernelId() const { return kernelId; }
    OpType getOpType() const { return opType; }
    const std::string &getOpName() const { return opName; }
    const std::vector<EClassId> &getChildren() const { return children; }
    const std::vector<uint32_t> &getShape() const { return shape; }
    const std::vector<uint64_t> &getStrides() const { return strides; }
    DType getDType() const { return dtype; }
    MemSpace getMemSpace() const { return mem_space; }
    std::vector<Engine> getEngines() const { return engines; }
    const std::string &getContentHash() const { return contentHash; }
    uint64_t getSig() const { return sig; }

    // Setters
    void setChildren(std::vector<EClassId> newChildren) { children = std::move(newChildren); }
    void setSig(uint64_t newSig) { sig = newSig; }

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
    std::string contentHash;

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
    std::unordered_map<uint64_t, std::vector<ENodeId>> hashcons;

    // Dense enodeId -> e_class_id mapping.
    std::vector<EClassId> nodeToEClass;

    uint32_t nextLeafId = 0;
    std::unordered_map<EClassId, std::shared_ptr<std::vector<uint8_t>>> constantStaging;

    // Hash map for fast constant lookup: data hash -> list of class ids
    std::unordered_map<uint64_t, std::vector<EClassId>> constantHashIndex;

    inline std::vector<int32_t> getConstantInt32(EClassId id) const
    {
        if (constantStaging.count(id))
        {
            const auto &data = *constantStaging.at(id);
            const auto &e_class = getEClass(id);
            uint64_t numElements = countElements(e_class.shape);
            std::vector<int32_t> res(numElements);
            const int32_t *src = reinterpret_cast<const int32_t *>(data.data());
            for (uint64_t i = 0; i < numElements; ++i)
            {
                res[i] = src[getStridedIndex(i, e_class.shape, e_class.strides)]; // TODO: does this need getStridedIndex?
            }
            return res;
        }
        std::stringstream ss;
        ss << "Expected constant for shape inference but not found in staging. Node ID: " << id;
        Error::throw_err(ss.str());
    }

    void reserve(uint64_t classCap, uint64_t nodeCap)
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

        EClassId cls = addEClass(shape, strides, dtype, MemSpace{1, HandleType::CPP});
        std::string contentHash = std::to_string(dataHash);
        ENode n = ENode(KernelId{0}, OpType::INPUT, "", {}, shape, strides, dtype, MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}}, contentHash);
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
        EClassId id{(uint32_t)classes.size()};

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

    EClassId addENode(EClassId e_class_id, ENode node)
    {
        EClassId canonical = find(e_class_id);

        // Retrieve a local copy of children, update them, and apply them back via setter
        std::vector<EClassId> updatedChildren = node.getChildren();
        for (EClassId &child : updatedChildren)
        {
            child = find(child);
        }
        node.setChildren(std::move(updatedChildren));

        node.setSig(computeSignature(node));

        auto it = hashcons.find(node.getSig());
        if (it != hashcons.end())
        {
            for (ENodeId otherEnodeId : it->second)
            {
                const ENode &other = enodes[otherEnodeId.value];
                if (node == other)
                {
                    merge(canonical, nodeToEClass[otherEnodeId.value]);
                    return find(canonical);
                }
            }
        }

        ENodeId enodeId = ENodeId{(uint32_t)enodes.size()};
        enodes.push_back(std::move(node));
        classes[canonical.value].enodes.push_back(enodeId);
        nodeToEClass.push_back(canonical);
        hashcons[enodes[enodeId.value].getSig()].push_back(enodeId);
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

    void merge(EClassId a, EClassId b)
    {
        EClassId ra = find(a);
        EClassId rb = find(b);
        if (ra == rb)
            return;

        // Union by size.
        if (ufSize[ra.value] < ufSize[rb.value])
            std::swap(ra, rb);

#ifdef DEBUG
        if (classes[ra.value].shape != classes[rb.value].shape)
        {
            Error::throw_err("EClass merge shape mismatch: " + toString(classes[ra.value].shape) + ", " + toString(classes[rb.value].shape));
        }
        if (classes[ra.value].strides != classes[rb.value].strides)
        {
            Error::throw_err("EClass merge strides mismatch: " + toString(classes[ra.value].strides) + ", " + toString(classes[rb.value].strides));
        }
        if (classes[ra.value].dtype != classes[rb.value].dtype)
        {
            Error::throw_err("EClass merge dtype mismatch");
        }
        if (!(classes[ra.value].mem_space == classes[rb.value].mem_space))
        {
            Error::throw_err("EClass merge mem_space mismatch");
        }
#endif

        parent[rb.value] = ra;
        ufSize[ra.value] += ufSize[rb.value];

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

        classes[ra.value].enodes.reserve(classes[ra.value].enodes.size() + classes[rb.value].enodes.size());
        for (ENodeId enodeId : classes[rb.value].enodes)
        {
            classes[ra.value].enodes.push_back(enodeId);
            nodeToEClass[enodeId.value] = ra;
        }
        classes[rb.value].enodes.clear();
    }

    void rebuild()
    {
        std::unordered_map<uint64_t, std::vector<ENodeId>> newHash;
        newHash.reserve(enodes.size() * 2);
        uint32_t nDupes = 0;

        for (uint32_t i = 0, n = static_cast<uint32_t>(enodes.size()); i < n; ++i)
        {
            ENode &node = enodes[i];
            ENodeId currentEnodeId{i};

            bool childrenChanged = false;
            std::vector<EClassId> updatedChildren = node.getChildren();
            for (EClassId &child : updatedChildren)
            {
                EClassId c = find(child);
                if (c != child)
                {
                    child = c;
                    childrenChanged = true;
                }
            }
            if (childrenChanged)
            {
                node.setChildren(std::move(updatedChildren));
            }

            EClassId cls = find(nodeToEClass[i]);
            nodeToEClass[i] = cls;

            if (childrenChanged || node.getSig() == 0)
            {
                node.setSig(computeSignature(node));
            }

            auto &bucket = newHash[node.getSig()];
            bool merged = false;

            for (ENodeId otherEnodeId : bucket)
            {
                const EClassId otherCls = find(nodeToEClass[otherEnodeId.value]);
                if (node == enodes[otherEnodeId.value])
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
                bucket.push_back(currentEnodeId);
            }
        }
        std::cout << "[EGraph.rebuild] Found " << nDupes << " duplicate enodes" << std::endl;

        hashcons = std::move(newHash);

        // Rebuild constant hash index to remove stale entries from merges
        rebuildConstantHashIndex();
    }

    const std::vector<EClass> &getClasses() const { return classes; }
    const std::vector<ENode> &getENodes() const { return enodes; }
    const ENode &getENode(ENodeId id) const { return enodes[id.value]; }

    EClass &getEClass(EClassId id) { return classes[find(id).value]; }
    const EClass &getEClass(EClassId id) const { return classes[findConst(id).value]; }

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
        uint64_t h = mix64(node.getKernelId().value);

        hashCombine(h, static_cast<uint64_t>(node.getOpType()));
        if (!node.getOpName().empty())
            hashCombine(h, hashString(node.getOpName()));
        if (!node.getContentHash().empty())
            hashCombine(h, hashString(node.getContentHash()));

        for (EClassId c : node.getChildren())
            hashCombine(h, static_cast<uint64_t>(c.value));

        for (uint32_t s : node.getShape())
            hashCombine(h, static_cast<uint64_t>(s));

        for (uint64_t s : node.getStrides())
            hashCombine(h, s);

        hashCombine(h, static_cast<uint64_t>(node.getDType()));

        hashCombine(h, static_cast<uint64_t>(node.getMemSpace().idx));
        hashCombine(h, static_cast<uint64_t>(node.getMemSpace().type));

        for (const Engine &e : node.getEngines())
        {
            hashCombine(h, static_cast<uint64_t>(e.idx));
            hashCombine(h, static_cast<uint64_t>(e.type));
        }

        return h;
    }

    static uint64_t computeConstantHash(const std::vector<uint32_t> &shape,
                                        const std::vector<uint64_t> &strides,
                                        DType dtype,
                                        const std::vector<uint8_t> &data) noexcept
    {
        uint64_t h = static_cast<uint64_t>(dtype);

        for (uint32_t s : shape)
            hashCombine(h, static_cast<uint64_t>(s));

        for (uint64_t s : strides)
            hashCombine(h, s);

        // Hash the data bytes efficiently - process 8 bytes at a time
        const uint8_t *ptr = data.data();
        uint64_t len = data.size();
        uint64_t i = 0;

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
            EClassId canonicalId = find(kv.first);
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
       << "  KernelUID:  0x" << std::hex << node.getKernelId().value << std::dec << "\n"
       << "  OpType:     " << toString(node.getOpType()) << "\n"
       << "  OpName:     " << (node.getOpName().empty() ? "N/A" : node.getOpName()) << "\n"
       << "  Children:   [";
    const auto &children = node.getChildren();
    for (uint64_t i = 0; i < children.size(); ++i)
    {
        ss << children[i].value << (i == children.size() - 1 ? "" : ", ");
    }
    ss << "]\n"
       << "  Shape:      " << ::toString(node.getShape()) << "\n"
       << "  Strides:    " << ::toString(node.getStrides()) << "\n"
       << "  DType:      " << ::toString(node.getDType()) << "\n"
       << "  MemSpace:   " << node.getMemSpace().idx << "\n"
       << "  Signature:  0x" << std::hex << node.getSig() << std::dec << "\n"
       << "}";
    return ss.str();
}

inline std::string toString(const EClass &cls, const std::string &prefix = "")
{
    std::stringstream ss;
    ss << prefix << "EClass\n"
       << prefix << "  ID:         " << cls.id.value << "\n"
       << prefix << "  Shape:      " << ::toString(cls.shape) << "\n"
       << prefix << "  Strides:    " << ::toString(cls.strides) << "\n"
       << prefix << "  DType:      " << ::toString(cls.dtype) << "\n"
       << prefix << "  MemSpace:   " << cls.mem_space.idx << "\n"
       << prefix << "  ENodes:     [";

    for (uint64_t i = 0; i < cls.enodes.size(); ++i)
    {
        ss << cls.enodes[i].value << (i == cls.enodes.size() - 1 ? "" : ", ");
    }
    ss << "]";
    return ss.str();
}

inline std::ostream &operator<<(std::ostream &os, const EClass &cls) { return os << toString(cls); }

inline std::string toString(const ENode &node, const EGraph &egraph, const std::string &prefix = "")
{
    std::stringstream ss;
    ss << prefix << "ENode [" << toString(node.getOpType());
    if (!node.getOpName().empty())
    {
        ss << " (" << node.getOpName() << ")";
    }
    ss << "]\n"
       << prefix << "  DType:      " << toString(node.getDType()) << "\n"
       << prefix << "  Shape:      " << toString(node.getShape()) << "\n"
       << prefix << "  Strides:    " << toString(node.getStrides()) << "\n"
       << prefix << "  MemSpace:   " << node.getMemSpace().idx << "\n"
       << prefix << "  Signature:  0x" << std::hex << node.getSig() << std::dec << "\n";

    if (node.getKernelId().value != 0 && node.getKernelId().value != UINT32_MAX)
    {
        ss << prefix << "  KernelUID:  0x" << std::hex << node.getKernelId().value << std::dec << "\n";
    }

    const auto &children = node.getChildren();
    ss << prefix << "  Children (" << children.size() << "):";

    if (children.empty())
    {
        ss << " None";
    }
    else
    {
        for (uint64_t i = 0; i < children.size(); ++i)
        {
            uint32_t childClassId = children[i].value;
            // Resolve the canonical EClass from the graph
            const EClass &childCls = egraph.getEClass(EClassId{childClassId});

            ss << "\n"
               << prefix << "    [" << i << "] EClass " << childClassId;

            // If the ID we have isn't the canonical one, note the redirect
            uint32_t canonicalId = egraph.findConst(EClassId{childClassId}).value;
            if (childClassId != canonicalId)
            {
                ss << " -> (Canonical: " << canonicalId << ")";
            }

            ss << "\n"
               << toString(childCls, prefix + "    ");
        }
    }
    return ss.str();
}