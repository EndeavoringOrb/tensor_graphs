#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

struct ENode
{
  public:
    ENode(KernelId kernelId, OpType opType, std::string opName, std::vector<EClassId> children,
          std::vector<uint32_t> shape, std::vector<uint64_t> strides, DType dtype, MemSpace mem_space,
          std::vector<Engine> engines, std::string contentHash = "", uint64_t sig = 0, std::string debugOrigin = "")
        : kernelId(kernelId), opType(opType), opName(std::move(opName)), children(std::move(children)),
          shape(std::move(shape)), strides(std::move(strides)), dtype(dtype), mem_space(mem_space),
          engines(std::move(engines)), contentHash(std::move(contentHash)), sig(sig),
          debugOrigin(std::move(debugOrigin))
    {
    }

    bool operator==(const ENode &other) const
    {
        return kernelId == other.kernelId && opType == other.opType && opName == other.opName &&
               children == other.children && shape == other.shape && strides == other.strides && dtype == other.dtype &&
               mem_space == other.mem_space && engines == other.engines && contentHash == other.contentHash;
    }

    bool operator!=(const ENode &other) const
    {
        return !(*this == other);
    }

    // Read-only getters
    KernelId getKernelId() const
    {
        return kernelId;
    }
    OpType getOpType() const
    {
        return opType;
    }
    const std::string &getOpName() const
    {
        return opName;
    }
    const std::vector<EClassId> &getChildren() const
    {
        return children;
    }
    const std::vector<uint32_t> &getShape() const
    {
        return shape;
    }
    const std::vector<uint64_t> &getStrides() const
    {
        return strides;
    }
    DType getDType() const
    {
        return dtype;
    }
    MemSpace getMemSpace() const
    {
        return mem_space;
    }
    const std::vector<Engine> &getEngines() const
    {
        return engines;
    }
    const std::string &getContentHash() const
    {
        return contentHash;
    }
    uint64_t getSig() const
    {
        return sig;
    }
    const std::string &getDebugOrigin() const
    {
        return debugOrigin;
    }

    // Setters
    void setChildren(std::vector<EClassId> newChildren)
    {
        children = std::move(newChildren);
    }
    void setSig(uint64_t newSig)
    {
        sig = newSig;
    }
    void setDebugOrigin(std::string origin)
    {
        debugOrigin = std::move(origin);
    }

  private:
    KernelId kernelId;
    OpType opType;
    std::string opName;
    std::vector<EClassId> children;
    std::vector<uint32_t> shape;
    std::vector<uint64_t> strides;
    DType dtype;
    MemSpace mem_space;
    std::vector<Engine> engines;
    std::string contentHash;
    uint64_t sig;
    std::string debugOrigin;
};

struct EClass
{
    EClassId id;
    BaseEClassId base_eclass_id;
    std::vector<ENodeId> enodes;
    std::vector<uint32_t> shape;
    std::vector<uint64_t> strides;
    DType dtype;
    MemSpace mem_space;
    LogicalId logical_id;
    bool is_clean = false;
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

    // Base eclass ID -> current canonical eclass ID.
    mutable std::unordered_map<BaseEClassId, EClassId> baseEClassToEClass;
    mutable bool baseEClassIndexInitialized = false;

    // Logical node ID -> current canonical eclass ID.
    mutable std::unordered_map<LogicalId, EClassId> logicalToEClass;
    mutable bool logicalIndexInitialized = false;

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
                res[i] = src[getStridedIndex(i, e_class.shape,
                                             e_class.strides)]; // TODO: does this need getStridedIndex?
            }
            return res;
        }
        std::stringstream ss;
        ss << "Expected constant for shape inference but not found in staging. "
              "Node ID: "
           << id;
        Error::throw_err(ss.str());
    }

    void reserve(uint64_t classCap, uint64_t nodeCap)
    {
        classes.reserve(classCap);
        parent.reserve(classCap);
        ufSize.reserve(classCap);
        baseEClassToEClass.reserve(classCap);
        logicalToEClass.reserve(classCap);

        enodes.reserve(nodeCap);
        nodeToEClass.reserve(nodeCap);
        hashcons.reserve(nodeCap * 2);
    }

    EClassId getOrAddConstant(const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides, DType dtype,
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
                if (cls.dtype == dtype && cls.shape == shape && cls.strides == strides && *stagingIt->second == data)
                {
                    return clsId;
                }
            }
        }

        EClassId cls = addEClass(shape, strides, dtype, MemSpace{1, HandleType::CPP});
        std::string contentHash = std::to_string(dataHash);
        ENode n = ENode(KernelId{0}, OpType::INPUT, "", {}, shape, strides, dtype, MemSpace{1, HandleType::CPP},
                        {Engine{0, EngineType::CPU}}, contentHash);
        addENode(cls, n);
        constantStaging[cls] = std::make_shared<std::vector<uint8_t>>(data);
        constantHashIndex[dataHash].push_back(cls);
        return cls;
    }

    template <typename T>
    EClassId getOrAddConstantData(const std::vector<uint32_t> &shape, DType dtype, const std::vector<T> &vals)
    {
        std::vector<uint64_t> strides = calcContiguousStrides(shape);
        std::vector<uint8_t> bytes(vals.size() * sizeof(T));
        std::memcpy(bytes.data(), vals.data(), bytes.size());
        return getOrAddConstant(shape, strides, dtype, bytes);
    }

    EClassId addIntConst(const std::vector<int32_t> &vals)
    {
        return getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, vals);
    }

    EClassId addEClass(const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides, DType dtype,
                       MemSpace mem_space, LogicalId logical_id = LogicalId{}, bool is_clean = false)
    {
        EClassId id{(uint32_t)classes.size()};

        EClass c;
        c.id = id;
        c.shape = shape;
        c.strides = strides;
        c.dtype = dtype;
        c.mem_space = mem_space;
        c.logical_id = logical_id;
        c.is_clean = is_clean;

        classes.push_back(std::move(c));
        parent.push_back(id);
        ufSize.push_back(1);
        if (logicalIndexInitialized && logical_id != LogicalId{})
            logicalToEClass.insert_or_assign(logical_id, id);
        return id;
    }

    EClassId addENode(EClassId e_class_id, ENode node)
    {
        EClassId canonical = find(e_class_id);

        // Retrieve a local copy of children, update them, and apply them back via
        // setter
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

#ifdef TG_DEBUG
        if (classes[ra.value].shape != classes[rb.value].shape)
        {
            Error::throw_err("EClass merge shape mismatch: " + toString(classes[ra.value].shape) + ", " +
                             toString(classes[rb.value].shape));
        }
        if (classes[ra.value].strides != classes[rb.value].strides)
        {
            Error::throw_err("EClass merge strides mismatch: " + toString(classes[ra.value].strides) + ", " +
                             toString(classes[rb.value].strides));
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

        const BaseEClassId baseA = classes[ra.value].base_eclass_id;
        const BaseEClassId baseB = classes[rb.value].base_eclass_id;
        if (baseA != BaseEClassId{} && baseB != BaseEClassId{})
        {
            auto describeEClass = [this](EClassId id) {
                const EClass &e_class = classes[id.value];
                std::stringstream details;
                details << "EClass " << toString(id) << " (base=" << toString(e_class.base_eclass_id)
                        << ", shape=" << toString(e_class.shape) << ", strides=" << toString(e_class.strides)
                        << ", dtype=" << toString(e_class.dtype) << ", mem_space=" << toString(e_class.mem_space)
                        << ", enodes=" << e_class.enodes.size() << ")";
                for (ENodeId enode_id : e_class.enodes)
                {
                    const ENode &enode = enodes[enode_id.value];
                    details << "\n  ENode " << toString(enode_id) << ": op=" << toString(enode.getOpType())
                            << ", name=" << (enode.getOpName().empty() ? "N/A" : enode.getOpName())
                            << ", kernel=" << toString(enode.getKernelId()) << ", debugOrigin="
                            << (enode.getDebugOrigin().empty() ? "N/A" : enode.getDebugOrigin());
                }
                return details.str();
            };
            Error::throw_err("EClass merge would merge two base eclasses:\n  " + describeEClass(ra) +
                             "\n  " + describeEClass(rb));
        }
        if (baseA == BaseEClassId{})
        {
            classes[ra.value].base_eclass_id = baseB;
            if (baseEClassIndexInitialized && baseB != BaseEClassId{})
                baseEClassToEClass.insert_or_assign(baseB, ra);
        }

        if (classes[ra.value].logical_id == LogicalId{})
        {
            classes[ra.value].logical_id = classes[rb.value].logical_id;
        }
        if (logicalIndexInitialized)
        {
            if (classes[ra.value].logical_id != LogicalId{})
                logicalToEClass.insert_or_assign(classes[ra.value].logical_id, ra);
            if (classes[rb.value].logical_id != LogicalId{})
                logicalToEClass.insert_or_assign(classes[rb.value].logical_id, ra);
        }
        classes[ra.value].is_clean = classes[ra.value].is_clean || classes[rb.value].is_clean;

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

    uint32_t getNumUniqueENodes() const
    {
        uint32_t count = 0;
        // hashcons only stores canonical, deduplicated nodes
        for (const auto &kv : hashcons)
        {
            count += kv.second.size();
        }
        return count;
    }

    void rebuild(bool compact = false)
    {
        while (true)
        {
            std::unordered_map<uint64_t, std::vector<ENodeId>> new_hash;
            new_hash.reserve(enodes.size() * 2);
            uint32_t n_dupes = 0;
            uint32_t n_merges = 0;

            for (uint32_t i = 0, n = static_cast<uint32_t>(enodes.size()); i < n; ++i)
            {
                ENode &node = enodes[i];
                ENodeId current_enode_id{i};

                bool children_changed = false;
                std::vector<EClassId> updated_children = node.getChildren();
                for (EClassId &child : updated_children)
                {
                    EClassId c = find(child);
                    if (c != child)
                    {
                        child = c;
                        children_changed = true;
                    }
                }
                if (children_changed)
                {
                    node.setChildren(std::move(updated_children));
                }

                EClassId cls = find(nodeToEClass[i]);
                nodeToEClass[i] = cls;

                if (children_changed || node.getSig() == 0)
                {
                    node.setSig(computeSignature(node));
                }

                auto &bucket = new_hash[node.getSig()];
                bool merged = false;

                for (ENodeId other_enode_id : bucket)
                {
                    const EClassId other_cls = find(nodeToEClass[other_enode_id.value]);
                    if (node == enodes[other_enode_id.value])
                    {
                        n_dupes++;
                        if (other_cls != cls)
                        {
                            merge(other_cls, cls);
                            n_merges++;
                        }
                        nodeToEClass[i] = find(other_cls);
                        merged = true;
                        break;
                    }
                }

                if (!merged)
                {
                    bucket.push_back(current_enode_id);
                }
            }

            hashcons = std::move(new_hash);
            rebuildConstantHashIndex();

            if (!compact || n_merges == 0)
                break;
        }

        if (!compact)
            return;

        // Compaction phase: prune dead/merged classes and duplicate enodes,
        // and renumber both classes and enodes densely starting from 0.
        const uint32_t num_old_classes = static_cast<uint32_t>(classes.size());
        std::vector<EClassId> old_to_new_class(num_old_classes, EClassId{UINT32_MAX});
        uint32_t new_class_count = 0;

        for (uint32_t i = 0; i < num_old_classes; ++i)
        {
            EClassId cid{i};
            if (find(cid) == cid)
            {
                old_to_new_class[i] = EClassId{new_class_count++};
            }
        }
        for (uint32_t i = 0; i < num_old_classes; ++i)
        {
            EClassId canon = find(EClassId{i});
            old_to_new_class[i] = old_to_new_class[canon.value];
        }

        // Mark unique enodes kept in hashcons as alive
        std::vector<bool> is_alive_node(enodes.size(), false);
        for (const auto &kv : hashcons)
        {
            for (ENodeId nid : kv.second)
            {
                is_alive_node[nid.value] = true;
            }
        }

        std::vector<ENodeId> old_to_new_node(enodes.size(), ENodeId{UINT32_MAX});
        std::vector<ENode> new_enodes;
        new_enodes.reserve(getNumUniqueENodes());
        std::vector<EClassId> new_node_to_eclass;
        new_node_to_eclass.reserve(getNumUniqueENodes());

        std::vector<EClass> new_classes;
        new_classes.reserve(new_class_count);

        for (uint32_t old_cls_idx = 0; old_cls_idx < num_old_classes; ++old_cls_idx)
        {
            EClassId old_cls_id{old_cls_idx};
            if (find(old_cls_id) != old_cls_id)
                continue;

            EClass &old_cls = classes[old_cls_idx];
            EClassId new_cls_id = old_to_new_class[old_cls_idx];

            EClass new_cls;
            new_cls.id = new_cls_id;
            new_cls.base_eclass_id = old_cls.base_eclass_id;
            new_cls.shape = std::move(old_cls.shape);
            new_cls.strides = std::move(old_cls.strides);
            new_cls.dtype = old_cls.dtype;
            new_cls.mem_space = old_cls.mem_space;
            new_cls.logical_id = old_cls.logical_id;
            new_cls.is_clean = old_cls.is_clean;

            for (ENodeId old_nid : old_cls.enodes)
            {
                if (!is_alive_node[old_nid.value])
                    continue;

                ENodeId new_nid{static_cast<uint32_t>(new_enodes.size())};
                old_to_new_node[old_nid.value] = new_nid;

                ENode node = std::move(enodes[old_nid.value]);
                std::vector<EClassId> children = node.getChildren();
                for (EClassId &child : children)
                {
                    child = old_to_new_class[child.value];
                }
                node.setChildren(std::move(children));
                node.setSig(computeSignature(node));

                new_enodes.push_back(std::move(node));
                new_node_to_eclass.push_back(new_cls_id);
                new_cls.enodes.push_back(new_nid);
            }

            new_classes.push_back(std::move(new_cls));
        }

        // Remap constant staging keys
        std::unordered_map<EClassId, std::shared_ptr<std::vector<uint8_t>>> new_constant_staging;
        for (auto &kv : constantStaging)
        {
            if (kv.first.value < num_old_classes)
            {
                EClassId new_cls_id = old_to_new_class[kv.first.value];
                if (new_cls_id.value != UINT32_MAX && new_constant_staging.find(new_cls_id) == new_constant_staging.end())
                {
                    new_constant_staging.emplace(new_cls_id, std::move(kv.second));
                }
            }
        }
        constantStaging = std::move(new_constant_staging);

        // Reset union-find data structures
        std::vector<EClassId> new_parent(new_class_count);
        for (uint32_t i = 0; i < new_class_count; ++i)
        {
            new_parent[i] = EClassId{i};
        }
        std::vector<uint32_t> new_uf_size(new_class_count, 1);

        // Rebuild baseEClassToEClass map if initialized
        if (baseEClassIndexInitialized)
        {
            baseEClassToEClass.clear();
            for (const auto &cls : new_classes)
            {
                if (cls.base_eclass_id != BaseEClassId{})
                    baseEClassToEClass.emplace(cls.base_eclass_id, cls.id);
            }
        }

        // Rebuild logicalToEClass map if initialized
        if (logicalIndexInitialized)
        {
            std::unordered_map<LogicalId, EClassId> new_logical_map;
            new_logical_map.reserve(logicalToEClass.size());
            for (const auto &pair : logicalToEClass)
            {
                if (pair.second.value < num_old_classes)
                {
                    EClassId canon = find(pair.second);
                    if (canon.value < num_old_classes)
                    {
                        EClassId new_id = old_to_new_class[canon.value];
                        if (new_id.value != UINT32_MAX)
                            new_logical_map.emplace(pair.first, new_id);
                    }
                }
            }
            for (const auto &cls : new_classes)
            {
                if (cls.logical_id != LogicalId{})
                    new_logical_map.insert_or_assign(cls.logical_id, cls.id);
            }
            logicalToEClass = std::move(new_logical_map);
        }

        // Rebuild hashcons
        std::unordered_map<uint64_t, std::vector<ENodeId>> new_hashcons;
        new_hashcons.reserve(new_enodes.size() * 2);
        for (uint32_t i = 0; i < new_enodes.size(); ++i)
        {
            new_hashcons[new_enodes[i].getSig()].push_back(ENodeId{i});
        }

        LOG(DEBUG) << "[EGraph.rebuild] Compacted egraph: " << num_old_classes << " -> " << new_classes.size()
                  << " classes, " << enodes.size() << " -> " << new_enodes.size() << " enodes" << std::endl;

        classes = std::move(new_classes);
        enodes = std::move(new_enodes);
        parent = std::move(new_parent);
        ufSize = std::move(new_uf_size);
        nodeToEClass = std::move(new_node_to_eclass);
        hashcons = std::move(new_hashcons);

        rebuildConstantHashIndex();
    }

    const std::vector<EClass> &getClasses() const
    {
        return classes;
    }
    const std::vector<ENode> &getENodes() const
    {
        return enodes;
    }
    const ENode &getENode(ENodeId id) const
    {
        return enodes[id.value];
    }

    EClass &getEClass(EClassId id)
    {
        return classes[find(id).value];
    }
    const EClass &getEClass(EClassId id) const
    {
        return classes[findConst(id).value];
    }

    void populateBaseEClassIds()
    {
        baseEClassToEClass.clear();
        baseEClassToEClass.reserve(classes.size());
        for (EClass &cls : classes)
        {
            if (findConst(cls.id) == cls.id)
            {
                cls.base_eclass_id = BaseEClassId{cls.id.value};
                if (cls.base_eclass_id != BaseEClassId{})
                    baseEClassToEClass.emplace(cls.base_eclass_id, cls.id);
            }
        }
        baseEClassIndexInitialized = true;
    }

    EClassId findEClassByBaseId(BaseEClassId base_id) const
    {
        if (base_id == BaseEClassId{})
            return EClassId{};
        // Some callers construct an EGraph and assign base_eclass_id directly
        // through getEClass(). Build the index on the first lookup so those
        // graphs retain the same lookup behavior without penalizing indexed
        // lookups thereafter.
        if (!baseEClassIndexInitialized)
        {
            baseEClassToEClass.clear();
            baseEClassToEClass.reserve(classes.size());
            for (const EClass &cls : classes)
            {
                if (findConst(cls.id) == cls.id && cls.base_eclass_id != BaseEClassId{})
                    baseEClassToEClass.emplace(cls.base_eclass_id, cls.id);
            }
            baseEClassIndexInitialized = true;
        }
        auto it = baseEClassToEClass.find(base_id);
        return it == baseEClassToEClass.end() ? EClassId{} : it->second;
    }

    EClassId getENodeEClass(ENodeId enodeId) const
    {
        return nodeToEClass[enodeId.value];
    }

    EClassId findEClassByLogicalId(LogicalId logical_id) const
    {
        if (logical_id == LogicalId{})
            return EClassId{};
        if (!logicalIndexInitialized)
        {
            logicalToEClass.clear();
            logicalToEClass.reserve(classes.size());
            for (const EClass &cls : classes)
            {
                if (cls.logical_id != LogicalId{})
                    logicalToEClass.insert_or_assign(cls.logical_id, findConst(cls.id));
            }
            logicalIndexInitialized = true;
        }
        auto it = logicalToEClass.find(logical_id);
        return it == logicalToEClass.end() ? EClassId{} : findConst(it->second);
    }

    LogicalId getLogicalId(EClassId id) const
    {
        return getEClass(id).logical_id;
    }

    void setLogicalId(EClassId id, LogicalId logical_id)
    {
        EClassId cid = find(id);
        classes[cid.value].logical_id = logical_id;
        if (logicalIndexInitialized && logical_id != LogicalId{})
            logicalToEClass.insert_or_assign(logical_id, cid);
    }

    bool isClean(EClassId id) const
    {
        return getEClass(id).is_clean;
    }

    void setClean(EClassId id, bool clean = true)
    {
        classes[find(id).value].is_clean = clean;
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

    static uint64_t computeConstantHash(const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides,
                                        DType dtype, const std::vector<uint8_t> &data) noexcept
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
       << prefix << "  ID:         " << cls.id.value << "\n";
    if (cls.logical_id != LogicalId{})
    {
        ss << prefix << "  LogicalId:  " << cls.logical_id.value << "\n";
    }
    if (cls.base_eclass_id != BaseEClassId{})
    {
        ss << prefix << "  BaseId:     " << cls.base_eclass_id.value << "\n";
    }
    if (cls.is_clean)
    {
        ss << prefix << "  Clean:      true\n";
    }
    ss << prefix << "  Shape:      " << ::toString(cls.shape) << "\n"
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

inline std::ostream &operator<<(std::ostream &os, const EClass &cls)
{
    return os << toString(cls);
}

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

            ss << "\n" << prefix << "    [" << i << "] EClass " << childClassId;

            // If the ID we have isn't the canonical one, note the redirect
            uint32_t canonicalId = egraph.findConst(EClassId{childClassId}).value;
            if (childClassId != canonicalId)
            {
                ss << " -> (Canonical: " << canonicalId << ")";
            }

            ss << "\n" << toString(childCls, prefix + "    ");
        }
    }
    return ss.str();
}
