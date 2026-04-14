#pragma once
#include "core/types.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <limits>
#include <algorithm>

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

    bool operator==(const ENode &other) const
    {
        return kernelUid == other.kernelUid &&
               opType == other.opType &&
               opName == other.opName &&
               children == other.children &&
               leafId == other.leafId &&
               shape == other.shape &&
               strides == other.strides &&
               viewOffset == other.viewOffset &&
               dtype == other.dtype &&
               backend == other.backend;
    }
};

struct ENodeHash
{
    size_t operator()(const ENode &key) const noexcept
    {
        size_t h = std::hash<uint64_t>{}(key.kernelUid);

        auto hash_combine = [](size_t &h, size_t v)
        {
            h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
        };

        hash_combine(h, static_cast<size_t>(key.opType));
        if (!key.opName.empty())
            hash_combine(h, std::hash<std::string>{}(key.opName));
        hash_combine(h, key.leafId);

        for (uint32_t c : key.children)
            hash_combine(h, c);

        for (uint32_t s : key.shape)
            hash_combine(h, s);

        for (uint64_t s : key.strides)
            hash_combine(h, static_cast<size_t>(s));

        hash_combine(h, static_cast<size_t>(key.viewOffset));
        hash_combine(h, static_cast<size_t>(key.dtype));
        hash_combine(h, static_cast<size_t>(key.backend));

        return h;
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

    uint32_t addEClass(const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides, uint64_t viewOffset, DType dtype, Backend backend)
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

        uint32_t leafId = UINT32_MAX;
        if (node.opType == OpType::INPUT)
        {
            leafId = nextLeafId;
            nextLeafId++;
        }
        node.leafId = leafId;

        auto it = hashcons.find(node);
        if (it != hashcons.end())
        {
            merge(canonical, it->second);
            return find(canonical);
        }

        uint32_t enodeId = static_cast<uint32_t>(enodes.size());
        enodes.push_back(node);
        classes[canonical].enodes.push_back(enodeId);
        nodeToEClass[enodeId] = canonical;
        hashcons[node] = canonical;
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
    }

    void rebuild()
    {
        std::unordered_map<ENode, uint32_t, ENodeHash> newHash;
        newHash.reserve(enodes.size());
        for (uint32_t i = 0; i < enodes.size(); ++i)
        {
            ENode &node = enodes[i];
            for (uint32_t &child : node.children)
            {
                child = find(child);
            }

            auto it = newHash.find(node);
            if (it != newHash.end())
            {
                merge(it->second, nodeToEClass.at(i));
                nodeToEClass[i] = find(it->second);
            }
            else
            {
                newHash[node] = find(nodeToEClass.at(i));
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
    std::unordered_map<ENode, uint32_t, ENodeHash> hashcons;
    std::unordered_map<uint32_t, uint32_t> nodeToEClass;
};

bool isContiguous(const EClass &eclass)
{
    return isContiguous(eclass.strides, eclass.shape);
}