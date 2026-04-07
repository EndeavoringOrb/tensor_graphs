#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <limits>
#include <algorithm>

struct ENodeKey
{
    OpType opType;
    uint64_t kernelUid = 0;
    uint32_t leafId = UINT32_MAX; // Used only for INPUT nodes to prevent bad merges
    Backend backend = Backend::CPU;
    std::vector<uint32_t> children;

    bool operator==(const ENodeKey &other) const
    {
        return opType == other.opType &&
               kernelUid == other.kernelUid &&
               leafId == other.leafId &&
               backend == other.backend &&
               children == other.children;
    }
};

// TODO: make generic hashing thing and use it instead of this custom hash
struct ENodeKeyHash
{
    size_t operator()(const ENodeKey &key) const noexcept
    {
        size_t h = std::hash<uint64_t>{}(key.kernelUid);
        h ^= std::hash<uint32_t>{}(static_cast<uint32_t>(key.opType)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(key.leafId) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>{}(static_cast<uint32_t>(key.backend)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (uint32_t c : key.children)
        {
            h ^= std::hash<uint32_t>{}(c) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct ENode
{
    uint32_t nodeId = 0; // Graph node id associated with this enode
    uint64_t kernelUid = 0;
    OpType opType = OpType::INPUT;
    std::string opName;
    Backend backend = Backend::CPU;
    std::vector<uint32_t> children;
};

struct EClass
{
    uint32_t id = 0;
    std::vector<uint32_t> enodes;
    std::vector<uint32_t> shape;
    DType dtype = DType::FLOAT32;
    bool contiguous = true;
    Backend backend = Backend::CPU;
    uint32_t refCount = 0;
};

class EGraph
{
public:
    uint32_t addEClass(const std::vector<uint32_t> &shape, DType dtype, uint32_t refCount, bool contiguous, Backend backend)
    {
        uint32_t id = static_cast<uint32_t>(classes.size());
        EClass c;
        c.id = id;
        c.shape = shape;
        c.dtype = dtype;
        c.refCount = refCount;
        c.contiguous = contiguous;
        c.backend = backend;
        classes.push_back(std::move(c));
        parent.push_back(id);
        return id;
    }

    uint32_t addENode(uint32_t eclassId, const ENode &enode)
    {
        uint32_t canonical = find(eclassId);
        ENode node = enode;
        for (uint32_t &child : node.children)
        {
            child = find(child);
        }

        ENodeKey key{
            node.opType,
            node.kernelUid,
            node.opType == OpType::INPUT ? node.nodeId : UINT32_MAX,
            node.backend,
            node.children};

        auto it = hashcons.find(key);
        if (it != hashcons.end())
        {
            merge(canonical, it->second);
            return find(canonical);
        }

        uint32_t enodeId = static_cast<uint32_t>(enodes.size());
        enodes.push_back(std::move(node));
        classes[canonical].enodes.push_back(enodeId);
        nodeToEClass[enodeId] = canonical;
        hashcons[key] = canonical;
        return canonical;
    }

    uint32_t find(uint32_t id)
    {
        if (parent[id] == id)
            return id;
        parent[id] = find(parent[id]);
        return parent[id];
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

        // Strict physical equality check to prevent polluting downstream expectations
        if (classes[ra].contiguous != classes[rb].contiguous ||
            classes[ra].backend != classes[rb].backend)
        {
            return;
        }

        if (classes[ra].enodes.size() < classes[rb].enodes.size())
            std::swap(ra, rb);

        parent[rb] = ra;

        for (uint32_t enodeId : classes[rb].enodes)
        {
            classes[ra].enodes.push_back(enodeId);
            nodeToEClass[enodeId] = ra;
        }
        classes[rb].enodes.clear();

        if (classes[ra].shape != classes[rb].shape)
        {
            Error::throw_err("EClass merge shape mismatch: " + toString(classes[ra].shape) + ", " + toString(classes[rb].shape));
        }
        if (classes[ra].dtype != classes[rb].dtype)
        {
            Error::throw_err("EClass merge dtype mismatch: " + (std::string)toString(classes[ra].dtype) + ", " + toString(classes[rb].dtype));
        }

        classes[ra].refCount = std::max(classes[ra].refCount, classes[rb].refCount);
    }

    void rebuild()
    {
        std::unordered_map<ENodeKey, uint32_t, ENodeKeyHash> newHash;
        for (uint32_t i = 0; i < enodes.size(); ++i)
        {
            ENode &node = enodes[i];
            for (uint32_t &child : node.children)
            {
                child = find(child);
            }

            ENodeKey key{
                node.opType,
                node.kernelUid,
                node.opType == OpType::INPUT ? node.nodeId : UINT32_MAX,
                node.backend,
                node.children};

            auto it = newHash.find(key);
            if (it != newHash.end())
            {
                merge(it->second, nodeToEClass.at(i));
                nodeToEClass[i] = find(it->second);
            }
            else
            {
                newHash[key] = find(nodeToEClass.at(i));
            }
        }
        hashcons = std::move(newHash);
    }

    void setNodeEClass(uint32_t enodeId, uint32_t eclassId)
    {
        nodeToEClass[enodeId] = eclassId;
    }

    const std::vector<EClass> &getClasses() const { return classes; }
    const std::vector<ENode> &getENodes() const { return enodes; }
    EClass &getEClass(uint32_t id) { return classes[find(id)]; }
    const EClass &getEClass(uint32_t id) const { return classes[findConst(id)]; }

    uint32_t getENodeEClass(uint32_t enodeId) const
    {
        auto it = nodeToEClass.find(enodeId);
        if (it == nodeToEClass.end())
            Error::throw_err("Missing enode -> eclass mapping.");
        return it->second;
    }

private:
    std::vector<EClass> classes;
    std::vector<ENode> enodes;
    std::vector<uint32_t> parent;
    std::unordered_map<ENodeKey, uint32_t, ENodeKeyHash> hashcons;
    std::unordered_map<uint32_t, uint32_t> nodeToEClass;
};