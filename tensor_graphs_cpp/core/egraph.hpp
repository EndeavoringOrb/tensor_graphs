#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <deque>

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
    void setContentHash(std::string value)
    {
        contentHash = std::move(value);
        sig = 0;
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
       << "  DebugOrigin: " << (node.getDebugOrigin().empty() ? "N/A" : node.getDebugOrigin()) << "\n"
       << "  Signature:  0x" << std::hex << node.getSig() << std::dec << "\n"
       << "}";
    return ss.str();
}

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

// Diagnostics: name of the rewrite rule currently applying on this thread
// (set by Planner::saturate, read by the merge trace).
inline thread_local const char *tg_active_rule = nullptr;

// Diagnostics: TG_TRACE_MERGE_SHAPE=4,2,16 logs every merge of classes with
// that shape; TG_TRACE_BASE_ID=<id> logs every merge involving that base id.
struct MergeTraceConfig
{
    bool enabled = false;
    bool has_shape = false;
    std::vector<uint32_t> shape;
    bool has_base_id = false;
    uint32_t base_id = 0;
};

inline const MergeTraceConfig &mergeTraceConfig()
{
    static const MergeTraceConfig config = [] {
        MergeTraceConfig c;
        if (const char *shape_env = std::getenv("TG_TRACE_MERGE_SHAPE"))
        {
            std::stringstream ss(shape_env);
            std::string item;
            while (std::getline(ss, item, ','))
                c.shape.push_back(static_cast<uint32_t>(std::stoul(item)));
            if (c.shape.empty())
                Error::throw_err("[mergeTraceConfig] TG_TRACE_MERGE_SHAPE is empty");
            c.has_shape = true;
        }
        if (const char *base_env = std::getenv("TG_TRACE_BASE_ID"))
        {
            c.base_id = static_cast<uint32_t>(std::stoul(base_env));
            c.has_base_id = true;
        }
        c.enabled = c.has_shape || c.has_base_id;
        return c;
    }();
    return config;
}

struct EGraph
{
    std::shared_ptr<const std::vector<EClass>> shared_classes;
    std::vector<EClass> classes;
    std::vector<uint32_t> modified_base_class;
    std::deque<EClass> local_modified_classes;
    std::vector<uint8_t> clean_classes;

    std::shared_ptr<const std::vector<ENode>> shared_enodes;
    std::vector<ENode> enodes;
    std::vector<EClassId> parent;
    std::vector<uint32_t> ufSize;

    // signature -> candidate enode ids
    std::shared_ptr<const std::unordered_map<uint64_t, std::vector<ENodeId>>> shared_hashcons;
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
    // Bucket saturation may prove additional equalities after full saturation
    // assigned base IDs. The joint planner reconciles these before search.
    bool allow_base_equalities = false;
    std::unordered_map<BaseEClassId, BaseEClassId> base_equalities;
    EClassId root_class_id = EClassId{};

    EGraph() = default;
    EGraph(EGraph &&) = default;
    EGraph &operator=(EGraph &&) = default;

    EGraph(const EGraph &other)
    {
        *this = other;
    }

    EGraph &operator=(const EGraph &other)
    {
        if (this != &other)
        {
            const bool profile = std::getenv("TG_PROFILE_COPY") != nullptr;
            auto t0 = std::chrono::high_resolution_clock::now();
            if (other.shared_classes)
            {
                shared_classes = other.shared_classes;
                classes.reserve(other.classes.size() + 2048);
                classes = other.classes;
                local_modified_classes = other.local_modified_classes;
                modified_base_class = other.modified_base_class;
            }
            else
            {
                shared_classes.reset();
                classes.reserve(other.classes.size() + 2048);
                classes = other.classes;
                local_modified_classes.clear();
                modified_base_class.clear();
            }
            clean_classes = other.clean_classes;
            auto t1 = std::chrono::high_resolution_clock::now();
            if (other.shared_enodes)
            {
                shared_enodes = other.shared_enodes;
                enodes.reserve(other.enodes.size() + 2048);
                enodes = other.enodes;
            }
            else
            {
                shared_enodes.reset();
                enodes.reserve(other.enodes.size() + 4096);
                enodes = other.enodes;
            }
            auto t2 = std::chrono::high_resolution_clock::now();
            parent.reserve(other.parent.size() + 2048);
            parent = other.parent;
            ufSize.reserve(other.ufSize.size() + 2048);
            ufSize = other.ufSize;
            if (other.shared_hashcons && other.hashcons.empty())
            {
                shared_hashcons = other.shared_hashcons;
                hashcons.clear();
            }
            else if (other.shared_hashcons)
            {
                shared_hashcons = other.shared_hashcons;
                hashcons = other.hashcons;
            }
            else
            {
                shared_hashcons.reset();
                hashcons = other.hashcons;
            }
            nodeToEClass.reserve(other.nodeToEClass.size() + 4096);
            nodeToEClass = other.nodeToEClass;
            auto t3 = std::chrono::high_resolution_clock::now();
            nextLeafId = other.nextLeafId;
            constantStaging = other.constantStaging;
            constantHashIndex = other.constantHashIndex;
            allow_base_equalities = other.allow_base_equalities;
            base_equalities = other.base_equalities;
            root_class_id = other.root_class_id;
            baseEClassToEClass.clear();
            baseEClassIndexInitialized = false;
            if (other.shared_logical_map && other.logicalToEClass.empty())
            {
                shared_logical_map = other.shared_logical_map;
                logicalToEClass.clear();
            }
            else if (other.shared_logical_map)
            {
                shared_logical_map = other.shared_logical_map;
                logicalToEClass = other.logicalToEClass;
            }
            else
            {
                shared_logical_map.reset();
                logicalToEClass = other.logicalToEClass;
            }
            logicalIndexInitialized = other.logicalIndexInitialized;
            auto t4 = std::chrono::high_resolution_clock::now();
            if (profile && (other.getNumClasses() > 100000))
            {
                double d_cls = std::chrono::duration<double, std::milli>(t1 - t0).count();
                double d_enodes = std::chrono::duration<double, std::milli>(t2 - t1).count();
                double d_arrays = std::chrono::duration<double, std::milli>(t3 - t2).count();
                double d_maps = std::chrono::duration<double, std::milli>(t4 - t3).count();
                std::cout << "[profile copy] classes=" << d_cls << " ms (" << getNumClasses()
                          << "), enodes=" << d_enodes << " ms (" << enodes.size()
                          << "), arrays=" << d_arrays << " ms"
                          << ", maps=" << d_maps << " ms" << std::endl;
            }
        }
        return *this;
    }

    BaseEClassId resolveBaseId(BaseEClassId id) const
    {
        auto it = base_equalities.find(id);
        while (it != base_equalities.end())
        {
            id = it->second;
            it = base_equalities.find(id);
        }
        return id;
    }

    // Logical node ID -> current canonical eclass ID.
    std::shared_ptr<const std::unordered_map<LogicalId, EClassId>> shared_logical_map;
    mutable std::unordered_map<LogicalId, EClassId> logicalToEClass;
    mutable bool logicalIndexInitialized = false;

    inline std::vector<int32_t> getConstantInt32(EClassId id, SourceLocation loc = SourceLocation::current()) const
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
        Error::throw_err(ss.str(), loc);
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
        EClassId id{static_cast<uint32_t>(getNumClasses())};

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

    void freezeClasses()
    {
        if (!shared_classes && !classes.empty())
        {
            shared_classes = std::make_shared<std::vector<EClass>>(std::move(classes));
            classes.clear();
            modified_base_class.clear();
            local_modified_classes.clear();
        }
    }

    void materializeClasses() const
    {
        if (shared_classes)
        {
            auto *self = const_cast<EGraph *>(this);
            std::vector<EClass> combined;
            combined.reserve(shared_classes->size() + self->classes.size());
            combined.insert(combined.end(), shared_classes->begin(), shared_classes->end());
            for (size_t i = 0; i < self->modified_base_class.size(); ++i)
            {
                uint32_t mod_idx = self->modified_base_class[i];
                if (mod_idx != UINT32_MAX && i < combined.size())
                {
                    combined[i] = self->local_modified_classes[mod_idx];
                }
            }
            combined.insert(combined.end(),
                            self->classes.begin(),
                            self->classes.end());
            if (!self->clean_classes.empty())
            {
                for (size_t i = 0; i < combined.size() && i < self->clean_classes.size(); ++i)
                {
                    if (self->clean_classes[i])
                        combined[i].is_clean = true;
                }
            }
            self->classes = std::move(combined);
            self->shared_classes.reset();
            self->modified_base_class.clear();
            self->local_modified_classes.clear();
        }
    }

    size_t getNumClasses() const
    {
        return (shared_classes ? shared_classes->size() : 0) + classes.size();
    }

    void freezeHashcons()
    {
        if (!shared_hashcons && !hashcons.empty())
        {
            shared_hashcons = std::make_shared<std::unordered_map<uint64_t, std::vector<ENodeId>>>(
                std::move(hashcons));
            hashcons.clear();
        }
    }

    void freezeLogicalIndex()
    {
        if (!logicalIndexInitialized && !shared_logical_map)
        {
            const size_t num_classes = getNumClasses();
            logicalToEClass.reserve(std::min<size_t>(num_classes, 256));
            for (size_t i = 0; i < num_classes; ++i)
            {
                EClassId cid{static_cast<uint32_t>(i)};
                const EClass &cls = getEClass(cid);
                if (cls.logical_id != LogicalId{})
                    logicalToEClass.insert_or_assign(cls.logical_id, findConst(cls.id));
            }
            logicalIndexInitialized = true;
        }
        if (!shared_logical_map && !logicalToEClass.empty())
        {
            shared_logical_map = std::make_shared<std::unordered_map<LogicalId, EClassId>>(
                std::move(logicalToEClass));
            logicalToEClass.clear();
        }
    }

    void clearHashcons()
    {
        hashcons.clear();
        shared_hashcons.reset();
    }

    void materializeENodes() const
    {
        if (shared_enodes)
        {
            if (enodes.empty())
            {
                const_cast<EGraph *>(this)->enodes = *shared_enodes;
            }
            else
            {
                std::vector<ENode> combined;
                combined.reserve(shared_enodes->size() + enodes.size() + 2048);
                combined.insert(combined.end(), shared_enodes->begin(), shared_enodes->end());
                combined.insert(combined.end(),
                                std::make_move_iterator(const_cast<EGraph *>(this)->enodes.begin()),
                                std::make_move_iterator(const_cast<EGraph *>(this)->enodes.end()));
                const_cast<EGraph *>(this)->enodes = std::move(combined);
            }
            const_cast<EGraph *>(this)->shared_enodes.reset();
        }
    }

    size_t getNumENodes() const
    {
        return shared_enodes ? (shared_enodes->size() + enodes.size()) : enodes.size();
    }

    void freezeENodes()
    {
        if (!shared_enodes && !enodes.empty())
        {
            shared_enodes = std::make_shared<std::vector<ENode>>(std::move(enodes));
            enodes.clear();
        }
    }

    void clearENodes()
    {
        enodes.clear();
        shared_enodes.reset();
    }

    // Class already holding an enode equal to `node` (children compared
    // canonically), or an invalid id.
    EClassId lookupENode(const ENode &node) const
    {
        uint64_t sig = node.getSig();
        bool needsCanon = false;
        for (EClassId child : node.getChildren())
        {
            if (child != findConst(child))
            {
                needsCanon = true;
                break;
            }
        }
        if (needsCanon)
        {
            ENode copy = node;
            std::vector<EClassId> children = copy.getChildren();
            for (EClassId &child : children)
                child = findConst(child);
            copy.setChildren(std::move(children));
            copy.setSig(computeSignature(copy));
            return lookupENode(copy);
        }
        if (sig == 0)
            sig = computeSignature(node);
        auto it = hashcons.find(sig);
        if (it != hashcons.end())
        {
            for (ENodeId other : it->second)
                if (node == getENode(other))
                    return findConst(nodeToEClass[other.value]);
        }
        if (shared_hashcons)
        {
            auto it2 = shared_hashcons->find(sig);
            if (it2 != shared_hashcons->end())
            {
                for (ENodeId other : it2->second)
                    if (node == getENode(other))
                        return findConst(nodeToEClass[other.value]);
            }
        }
        return EClassId{};
    }

    EClassId addENode(EClassId e_class_id, ENode node)
    {
        EClassId canonical = find(e_class_id);

        bool needsCanon = false;
        for (EClassId child : node.getChildren())
        {
            if (child != findConst(child))
            {
                needsCanon = true;
                break;
            }
        }
        if (needsCanon)
        {
            std::vector<EClassId> updatedChildren = node.getChildren();
            for (EClassId &child : updatedChildren)
            {
                child = find(child);
            }
            node.setChildren(std::move(updatedChildren));
            node.setSig(computeSignature(node));
        }
        else if (node.getSig() == 0)
        {
            node.setSig(computeSignature(node));
        }

        auto it = hashcons.find(node.getSig());
        if (it != hashcons.end())
        {
            for (ENodeId otherEnodeId : it->second)
            {
                const ENode &other = getENode(otherEnodeId);
                if (node == other)
                {
                    merge(canonical, nodeToEClass[otherEnodeId.value]);
                    return find(canonical);
                }
            }
        }
        if (shared_hashcons)
        {
            auto it2 = shared_hashcons->find(node.getSig());
            if (it2 != shared_hashcons->end())
            {
                for (ENodeId otherEnodeId : it2->second)
                {
                    const ENode &other = getENode(otherEnodeId);
                    if (node == other)
                    {
                        merge(canonical, nodeToEClass[otherEnodeId.value]);
                        return find(canonical);
                    }
                }
            }
        }

        ENodeId enodeId = ENodeId{static_cast<uint32_t>(getNumENodes())};
        const uint64_t sig = node.getSig();
        enodes.push_back(std::move(node));
        getMutableEClass(canonical).enodes.push_back(enodeId);
        nodeToEClass.push_back(canonical);
        hashcons[sig].push_back(enodeId);
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

    void traceMerge(EClassId ra, EClassId rb) const
    {
        const MergeTraceConfig &config = mergeTraceConfig();
        const EClass &cls_ra = getRawEClass(ra.value);
        const EClass &cls_rb = getRawEClass(rb.value);
        bool hit = config.has_shape && cls_ra.shape == config.shape;
        if (config.has_base_id)
        {
            const BaseEClassId wanted{config.base_id};
            hit = hit || (cls_ra.base_eclass_id != BaseEClassId{} && resolveBaseId(cls_ra.base_eclass_id) == wanted) ||
                  (cls_rb.base_eclass_id != BaseEClassId{} && resolveBaseId(cls_rb.base_eclass_id) == wanted);
        }
        if (!hit)
            return;
        auto describe = [this](const EClass &cls) {
            std::stringstream ss;
            ss << "class=" << cls.id.value << " base=" << cls.base_eclass_id.value << " logical=" << cls.logical_id.value
               << " enodes=" << cls.enodes.size();
            if (!cls.enodes.empty())
            {
                const ENode &first = getENode(cls.enodes.front());
                ss << " first=" << toString(first.getOpType()) << ":" << first.getOpName() << " children=[";
                for (size_t i = 0; i < first.getChildren().size(); ++i)
                    ss << (i ? "," : "") << first.getChildren()[i].value;
                ss << "] origin=" << (first.getDebugOrigin().empty() ? "N/A" : first.getDebugOrigin());
            }
            return ss.str();
        };
        std::stringstream out;
        out << "[merge trace] rule=" << (tg_active_rule ? tg_active_rule : "-") << " shape=" << toString(cls_ra.shape)
            << " strides=" << toString(cls_ra.strides) << "\n  A " << describe(cls_ra) << "\n  B " << describe(cls_rb)
            << "\n";
        std::cout << out.str() << std::flush;
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

        const EClass &cls_ra = getRawEClass(ra.value);
        const EClass &cls_rb = getRawEClass(rb.value);
        if (cls_ra.shape != cls_rb.shape ||
            cls_ra.dtype != cls_rb.dtype ||
            cls_ra.mem_space != cls_rb.mem_space ||
            cls_ra.strides != cls_rb.strides)
        {
            std::stringstream ss;
            ss << "[ILLEGAL EClass MERGE] ra=" << ra.value << " (shape=" << toString(cls_ra.shape)
               << ", strides=" << toString(cls_ra.strides)
               << ", dtype=" << static_cast<int>(cls_ra.dtype)
               << ", mem_space=" << toString(cls_ra.mem_space) << ") vs rb=" << rb.value
               << " (shape=" << toString(cls_rb.shape)
               << ", strides=" << toString(cls_rb.strides)
               << ", dtype=" << static_cast<int>(cls_rb.dtype)
               << ", mem_space=" << toString(cls_rb.mem_space) << ")";
            Error::throw_err(ss.str());
        }
        if (mergeTraceConfig().enabled)
            traceMerge(ra, rb);

        auto itConstA = constantStaging.find(ra);
        auto itConstB = constantStaging.find(rb);
        if (itConstA != constantStaging.end() && itConstB != constantStaging.end())
        {
            if (*itConstA->second != *itConstB->second)
            {
                Error::throw_err("[CRITICAL CONSTANT MERGE] Cannot merge different constants: ra=" +
                                 std::to_string(ra.value) + " vs rb=" + std::to_string(rb.value) +
                                 " (shape=" + toString(cls_ra.shape) + ", strides=" + toString(cls_ra.strides) + ")");
            }
        }

        const BaseEClassId baseA = cls_ra.base_eclass_id;
        const BaseEClassId baseB = cls_rb.base_eclass_id;
        if (baseA != BaseEClassId{} && baseB != BaseEClassId{} && !allow_base_equalities)
        {
            auto describeEClass = [this](EClassId id) {
                const EClass &e_class = getRawEClass(id.value);
                std::stringstream details;
                details << "EClass " << toString(id) << " (base=" << toString(e_class.base_eclass_id)
                        << ", shape=" << toString(e_class.shape) << ", strides=" << toString(e_class.strides)
                        << ", dtype=" << toString(e_class.dtype) << ", mem_space=" << toString(e_class.mem_space)
                        << ", enodes=" << e_class.enodes.size() << ")";
                for (ENodeId enode_id : e_class.enodes)
                {
                    const ENode &enode = getENode(enode_id);
                    details << "\n  ENode " << toString(enode_id) << ":\n" << toString(enode);
                }
                return details.str();
            };
            Error::throw_err("EClass merge would merge two base eclasses:\n  " + describeEClass(ra) +
                             "\n  " + describeEClass(rb));
        }
        if (baseA != BaseEClassId{} && baseB != BaseEClassId{} && allow_base_equalities)
        {
            const BaseEClassId left = resolveBaseId(baseA), right = resolveBaseId(baseB);
            const BaseEClassId canonical = left.value < right.value ? left : right;
            const BaseEClassId alias = left.value < right.value ? right : left;
            if (alias != canonical)
                base_equalities[alias] = canonical;
            getMutableEClass(ra).base_eclass_id = canonical;
            baseEClassIndexInitialized = false;
        }
        if (baseA == BaseEClassId{})
        {
            getMutableEClass(ra).base_eclass_id = baseB;
            if (baseEClassIndexInitialized && baseB != BaseEClassId{})
                baseEClassToEClass.insert_or_assign(baseB, ra);
        }

        LogicalId logA = cls_ra.logical_id;
        LogicalId logB = cls_rb.logical_id;
        if (logA == LogicalId{})
        {
            getMutableEClass(ra).logical_id = logB;
            logA = logB;
        }
        if (logicalIndexInitialized || shared_logical_map)
        {
            if (logA != LogicalId{})
                logicalToEClass.insert_or_assign(logA, ra);
            if (logB != LogicalId{})
                logicalToEClass.insert_or_assign(logB, ra);
        }
        getMutableEClass(ra).is_clean = isClean(ra) || isClean(rb);

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

        std::vector<ENodeId> rb_enodes = std::move(getMutableEClass(rb).enodes);
        getMutableEClass(rb).enodes.clear();
        EClass &mut_ra = getMutableEClass(ra);
        mut_ra.enodes.reserve(mut_ra.enodes.size() + rb_enodes.size());
        for (ENodeId enodeId : rb_enodes)
        {
            mut_ra.enodes.push_back(enodeId);
            nodeToEClass[enodeId.value] = ra;
        }

        parent[rb.value] = ra;
        ufSize[ra.value] += ufSize[rb.value];
    }

    uint32_t getNumUniqueENodes() const
    {
        uint32_t count = 0;
        // hashcons only stores canonical, deduplicated nodes
        for (const auto &kv : hashcons)
        {
            count += static_cast<uint32_t>(kv.second.size());
        }
        if (shared_hashcons)
        {
            for (const auto &kv : *shared_hashcons)
            {
                count += static_cast<uint32_t>(kv.second.size());
            }
        }
        return count;
    }

    void rebuild(bool compact = false)
    {
        uint32_t total_merges = 0;
        const size_t base_count = shared_enodes ? shared_enodes->size() : 0;
        while (true)
        {
            std::unordered_map<uint64_t, std::vector<ENodeId>> new_hash;
            new_hash.reserve(enodes.size() * 2);
            uint32_t n_dupes = 0;
            uint32_t n_merges = 0;

            for (uint32_t i = 0, n = static_cast<uint32_t>(enodes.size()); i < n; ++i)
            {
                ENode &node = enodes[i];
                ENodeId current_enode_id{static_cast<uint32_t>(base_count + i)};

                bool children_changed = false;
                for (EClassId child : node.getChildren())
                {
                    if (find(child) != child)
                    {
                        children_changed = true;
                        break;
                    }
                }
                if (children_changed)
                {
                    std::vector<EClassId> updated_children = node.getChildren();
                    for (EClassId &child : updated_children)
                    {
                        child = find(child);
                    }
                    node.setChildren(std::move(updated_children));
                    node.setSig(computeSignature(node));
                }

                EClassId cls = find(nodeToEClass[current_enode_id.value]);
                nodeToEClass[current_enode_id.value] = cls;

                if (node.getSig() == 0)
                {
                    node.setSig(computeSignature(node));
                }

                auto &bucket = new_hash[node.getSig()];
                bool merged = false;

                for (ENodeId other_enode_id : bucket)
                {
                    const EClassId other_cls = find(nodeToEClass[other_enode_id.value]);
                    if (node == getENode(other_enode_id))
                    {
                        n_dupes++;
                        if (other_cls != cls)
                        {
                            merge(other_cls, cls);
                            n_merges++;
                        }
                        nodeToEClass[current_enode_id.value] = find(other_cls);
                        merged = true;
                        break;
                    }
                }

                if (!merged && shared_hashcons)
                {
                    auto it = shared_hashcons->find(node.getSig());
                    if (it != shared_hashcons->end())
                    {
                        for (ENodeId other_enode_id : it->second)
                        {
                            const EClassId other_cls = find(nodeToEClass[other_enode_id.value]);
                            if (node == getENode(other_enode_id))
                            {
                                n_dupes++;
                                if (other_cls != cls)
                                {
                                    merge(other_cls, cls);
                                    n_merges++;
                                }
                                nodeToEClass[current_enode_id.value] = find(other_cls);
                                merged = true;
                                break;
                            }
                        }
                    }
                }

                if (!merged)
                {
                    bucket.push_back(current_enode_id);
                }
            }

            hashcons = std::move(new_hash);
            total_merges += n_merges;

            if (n_merges == 0)
                break;
        }

        if (total_merges > 0)
        {
            rebuildConstantHashIndex();
        }

        if (!compact)
            return;

        // Compaction phase: prune dead/merged classes and duplicate enodes,
        // and renumber both classes and enodes densely starting from 0.
        const uint32_t num_old_classes = static_cast<uint32_t>(getNumClasses());
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
        std::vector<bool> is_alive_node(getNumENodes(), false);
        for (const auto &kv : hashcons)
        {
            for (ENodeId nid : kv.second)
            {
                is_alive_node[nid.value] = true;
            }
        }
        if (shared_hashcons)
        {
            for (const auto &kv : *shared_hashcons)
            {
                for (ENodeId nid : kv.second)
                {
                    is_alive_node[nid.value] = true;
                }
            }
        }

        std::vector<ENodeId> old_to_new_node(getNumENodes(), ENodeId{UINT32_MAX});
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

            const EClass &old_cls = getEClass(old_cls_id);
            EClassId new_cls_id = old_to_new_class[old_cls_idx];

            EClass new_cls;
            new_cls.id = new_cls_id;
            new_cls.base_eclass_id = old_cls.base_eclass_id;
            new_cls.shape = old_cls.shape;
            new_cls.strides = old_cls.strides;
            new_cls.dtype = old_cls.dtype;
            new_cls.mem_space = old_cls.mem_space;
            new_cls.logical_id = old_cls.logical_id;
            new_cls.is_clean = isClean(old_cls_id);

            for (ENodeId old_nid : old_cls.enodes)
            {
                if (!is_alive_node[old_nid.value])
                    continue;

                ENodeId new_nid{static_cast<uint32_t>(new_enodes.size())};
                old_to_new_node[old_nid.value] = new_nid;

                ENode node = getENode(old_nid);
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

        if (root_class_id != EClassId{} && root_class_id.value < num_old_classes)
        {
            root_class_id = old_to_new_class[root_class_id.value];
        }

        // Rebuild logicalToEClass map if initialized
        if (logicalIndexInitialized || shared_logical_map)
        {
            std::unordered_map<LogicalId, EClassId> new_logical_map;
            if (shared_logical_map)
            {
                new_logical_map.reserve(shared_logical_map->size() + logicalToEClass.size());
                for (const auto &pair : *shared_logical_map)
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
            }
            for (const auto &pair : logicalToEClass)
            {
                if (pair.second.value < num_old_classes)
                {
                    EClassId canon = find(pair.second);
                    if (canon.value < num_old_classes)
                    {
                        EClassId new_id = old_to_new_class[canon.value];
                        if (new_id.value != UINT32_MAX)
                            new_logical_map.insert_or_assign(pair.first, new_id);
                    }
                }
            }
            for (const auto &cls : new_classes)
            {
                if (cls.logical_id != LogicalId{})
                    new_logical_map.insert_or_assign(cls.logical_id, cls.id);
            }
            shared_logical_map.reset();
            logicalToEClass = std::move(new_logical_map);
            logicalIndexInitialized = true;
        }

        // Rebuild hashcons
        std::unordered_map<uint64_t, std::vector<ENodeId>> new_hashcons;
        new_hashcons.reserve(new_enodes.size() * 2);
        for (uint32_t i = 0; i < new_enodes.size(); ++i)
        {
            new_hashcons[new_enodes[i].getSig()].push_back(ENodeId{i});
        }

        LOG(DEBUG) << "[EGraph.rebuild] Compacted egraph: " << num_old_classes << " -> " << new_classes.size()
                  << " classes, " << getNumENodes() << " -> " << new_enodes.size() << " enodes" << std::endl;

        classes = std::move(new_classes);
        shared_classes.reset();
        modified_base_class.clear();
        local_modified_classes.clear();
        clean_classes.clear();
        shared_enodes.reset();
        enodes = std::move(new_enodes);
        parent = std::move(new_parent);
        ufSize = std::move(new_uf_size);
        nodeToEClass = std::move(new_node_to_eclass);
        shared_hashcons.reset();
        hashcons = std::move(new_hashcons);

        rebuildConstantHashIndex();
    }

    const std::vector<EClass> &getClasses() const
    {
        materializeClasses();
        return classes;
    }
    const std::vector<ENode> &getENodes() const
    {
        if (shared_enodes && enodes.empty())
            return *shared_enodes;
        materializeENodes();
        return enodes;
    }
    const ENode &getENode(ENodeId id) const
    {
        if (shared_enodes)
        {
            if (id.value < shared_enodes->size())
                return (*shared_enodes)[id.value];
            return enodes[id.value - shared_enodes->size()];
        }
        return enodes[id.value];
    }

    EClass &getMutableEClass(EClassId canon)
    {
        const uint32_t val = canon.value;
        if (shared_classes && val < shared_classes->size())
        {
            if (modified_base_class.size() <= val)
                modified_base_class.resize(shared_classes->size(), UINT32_MAX);
            uint32_t mod_idx = modified_base_class[val];
            if (mod_idx == UINT32_MAX)
            {
                mod_idx = static_cast<uint32_t>(local_modified_classes.size());
                modified_base_class[val] = mod_idx;
                local_modified_classes.push_back((*shared_classes)[val]);
            }
            return local_modified_classes[mod_idx];
        }
        const uint32_t base_count = shared_classes ? static_cast<uint32_t>(shared_classes->size()) : 0;
        return classes[val - base_count];
    }

    EClass &getEClass(EClassId id)
    {
        return getMutableEClass(find(id));
    }
    const EClass &getRawEClass(uint32_t val) const
    {
        if (shared_classes && val < shared_classes->size())
        {
            if (val < modified_base_class.size())
            {
                uint32_t mod_idx = modified_base_class[val];
                if (mod_idx != UINT32_MAX)
                    return local_modified_classes[mod_idx];
            }
            return (*shared_classes)[val];
        }
        const uint32_t base_count = shared_classes ? static_cast<uint32_t>(shared_classes->size()) : 0;
        return classes[val - base_count];
    }
    const EClass &getEClass(EClassId id) const
    {
        return getRawEClass(findConst(id).value);
    }

    void populateBaseEClassIds()
    {
        baseEClassToEClass.clear();
        baseEClassIndexInitialized = false;
        const size_t num_classes = getNumClasses();
        for (size_t i = 0; i < num_classes; ++i)
        {
            EClassId cid{static_cast<uint32_t>(i)};
            if (findConst(cid) == cid)
            {
                getMutableEClass(cid).base_eclass_id = BaseEClassId{cid.value};
            }
        }
    }

    EClassId findEClassByBaseId(BaseEClassId base_id) const
    {
        base_id = resolveBaseId(base_id);
        if (base_id == BaseEClassId{})
            return EClassId{};
        // Some callers construct an EGraph and assign base_eclass_id directly
        // through getEClass(). Build the index on the first lookup so those
        // graphs retain the same lookup behavior without penalizing indexed
        // lookups thereafter.
        if (!baseEClassIndexInitialized)
        {
            baseEClassToEClass.clear();
            const size_t num_classes = getNumClasses();
            baseEClassToEClass.reserve(num_classes);
            for (size_t i = 0; i < num_classes; ++i)
            {
                EClassId cid{static_cast<uint32_t>(i)};
                const EClass &cls = getEClass(cid);
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
        auto it = logicalToEClass.find(logical_id);
        if (it != logicalToEClass.end())
            return findConst(it->second);
        if (shared_logical_map)
        {
            auto it2 = shared_logical_map->find(logical_id);
            if (it2 != shared_logical_map->end())
                return findConst(it2->second);
        }
        if (!logicalIndexInitialized)
        {
            logicalToEClass.clear();
            const size_t num_classes = getNumClasses();
            logicalToEClass.reserve(std::min<size_t>(num_classes, 256));
            for (size_t i = 0; i < num_classes; ++i)
            {
                EClassId cid{static_cast<uint32_t>(i)};
                const EClass &cls = getEClass(cid);
                if (cls.logical_id != LogicalId{})
                    logicalToEClass.insert_or_assign(cls.logical_id, findConst(cls.id));
            }
            logicalIndexInitialized = true;
            auto it3 = logicalToEClass.find(logical_id);
            if (it3 != logicalToEClass.end())
                return findConst(it3->second);
        }
        return EClassId{};
    }

    LogicalId getLogicalId(EClassId id) const
    {
        return getEClass(id).logical_id;
    }

    void setLogicalId(EClassId id, LogicalId logical_id)
    {
        EClassId cid = find(id);
        getMutableEClass(cid).logical_id = logical_id;
        if ((logicalIndexInitialized || shared_logical_map) && logical_id != LogicalId{})
            logicalToEClass.insert_or_assign(logical_id, cid);
    }

    bool isClean(EClassId id) const
    {
        uint32_t canon = findConst(id).value;
        if (canon < clean_classes.size() && clean_classes[canon])
            return true;
        return getEClass(id).is_clean;
    }

    void setClean(EClassId id, bool clean = true)
    {
        uint32_t canon = find(id).value;
        if (canon < clean_classes.size())
            clean_classes[canon] = clean ? 1 : 0;
        getMutableEClass(EClassId{canon}).is_clean = clean;
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

    uint64_t computeConstantHash(const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides,
                                 DType dtype, const std::vector<uint8_t> &data) const noexcept
    {
        return tg_hash::computeConstantHash(shape, strides, dtype, data);
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
