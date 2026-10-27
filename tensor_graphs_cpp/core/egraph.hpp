#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "core/common/thread_pool.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

// Interned immutable strings: equal values share one never-freed copy, so
// e-nodes copy names and debug origins without allocating. The table only
// grows; the values come from a small set (kernel names, rule origins).
inline const std::string *internString(const std::string &value)
{
    static const std::string *const empty = new std::string();
    if (value.empty())
        return empty;
    // Per-thread lookups first: rewrite threads intern the same few names
    // millions of times. Keys view the interned (stable) strings.
    thread_local std::unordered_map<std::string_view, const std::string *> local;
    const auto local_it = local.find(std::string_view(value));
    if (local_it != local.end())
        return local_it->second;
    struct Shard
    {
        std::mutex mutex;
        std::unordered_set<std::string> values;
    };
    constexpr size_t kShards = 64;
    static Shard *const shards = new Shard[kShards];
    Shard &shard = shards[std::hash<std::string>{}(value) % kShards];
    const std::string *interned;
    {
        std::lock_guard<std::mutex> lock(shard.mutex);
        interned = &*shard.values.insert(value).first;
    }
    local.emplace(std::string_view(*interned), interned);
    return interned;
}

// Interned immutable vectors (shapes, strides, engine lists), as for
// internString. Equality is exact element equality (engines include their
// supported memory spaces), so an interned vector equals its input.
inline bool internedElementEqual(uint32_t a, uint32_t b) { return a == b; }
inline bool internedElementEqual(uint64_t a, uint64_t b) { return a == b; }
inline bool internedElementEqual(const Engine &a, const Engine &b)
{
    return a.idx == b.idx && a.type == b.type && a.supported == b.supported;
}
inline uint64_t internedElementHash(uint32_t v) { return v; }
inline uint64_t internedElementHash(uint64_t v) { return v; }
inline uint64_t internedElementHash(const Engine &e)
{
    return (static_cast<uint64_t>(e.idx) << 8) ^ static_cast<uint64_t>(e.type) ^ (e.supported.size() << 40);
}

template <typename T>
struct InternedVectorOps
{
    size_t operator()(const std::vector<T> *v) const noexcept
    {
        uint64_t h = 0xcbf29ce484222325ull ^ v->size();
        for (const T &x : *v)
            h = (h ^ internedElementHash(x)) * 0x100000001b3ull;
        return static_cast<size_t>(h);
    }
    bool operator()(const std::vector<T> *a, const std::vector<T> *b) const noexcept
    {
        if (a->size() != b->size())
            return false;
        for (size_t i = 0; i < a->size(); ++i)
            if (!internedElementEqual((*a)[i], (*b)[i]))
                return false;
        return true;
    }
};

template <typename T>
inline const std::vector<T> *internVector(const std::vector<T> &value)
{
    static const std::vector<T> *const empty = new std::vector<T>();
    if (value.empty())
        return empty;
    using Set = std::unordered_set<const std::vector<T> *, InternedVectorOps<T>, InternedVectorOps<T>>;
    thread_local Set local;
    const auto local_it = local.find(&value);
    if (local_it != local.end())
        return *local_it;
    struct Shard
    {
        std::mutex mutex;
        Set values;
    };
    constexpr size_t kShards = 64;
    static Shard *const shards = new Shard[kShards];
    Shard &shard = shards[InternedVectorOps<T>{}(&value) % kShards];
    const std::vector<T> *interned;
    {
        std::lock_guard<std::mutex> lock(shard.mutex);
        const auto it = shard.values.find(&value);
        if (it != shard.values.end())
            interned = *it;
        else
        {
            interned = new std::vector<T>(value);
            shard.values.insert(interned);
        }
    }
    local.insert(interned);
    return interned;
}

// An interned vector held by value: copies share the interned storage.
// Reads go through the vector interface; assignment interns the new value.
template <typename T>
class InternedVector
{
  public:
    InternedVector() : value_(internVector(std::vector<T>())) {}
    InternedVector(const std::vector<T> &value) : value_(internVector(value)) {}
    InternedVector &operator=(const std::vector<T> &value)
    {
        value_ = internVector(value);
        return *this;
    }
    operator const std::vector<T> &() const { return *value_; }
    const std::vector<T> &get() const { return *value_; }
    size_t size() const { return value_->size(); }
    bool empty() const { return value_->empty(); }
    const T &operator[](size_t i) const { return (*value_)[i]; }
    const T &at(size_t i) const { return value_->at(i); }
    const T &front() const { return value_->front(); }
    const T &back() const { return value_->back(); }
    const T *data() const { return value_->data(); }
    typename std::vector<T>::const_iterator begin() const { return value_->begin(); }
    typename std::vector<T>::const_iterator end() const { return value_->end(); }
    bool operator==(const InternedVector &other) const { return value_ == other.value_; }
    bool operator!=(const InternedVector &other) const { return value_ != other.value_; }
    friend bool operator==(const InternedVector &a, const std::vector<T> &b) { return *a.value_ == b; }
    friend bool operator==(const std::vector<T> &a, const InternedVector &b) { return a == *b.value_; }
    friend bool operator!=(const InternedVector &a, const std::vector<T> &b) { return *a.value_ != b; }
    friend bool operator!=(const std::vector<T> &a, const InternedVector &b) { return a != *b.value_; }

  private:
    const std::vector<T> *value_;
};

template <typename T>
inline std::string toString(const InternedVector<T> &value)
{
    return toString(value.get());
}

struct ENode
{
  public:
    // Names, shapes, strides and engines are interned, so they are taken by
    // reference (an interned value is copied only the first time it occurs).
    ENode(KernelId kernelId, OpType opType, const std::string &opName, std::vector<EClassId> children,
          const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides, DType dtype, MemSpace mem_space,
          const std::vector<Engine> &engines, std::string contentHash = "", uint64_t sig = 0,
          const std::string &debugOrigin = "")
        : kernelId(kernelId), opType(opType), opName(internString(opName)), children(std::move(children)),
          shape(internVector(shape)), strides(internVector(strides)), dtype(dtype), mem_space(mem_space),
          engines(internVector(engines)), contentHash(std::move(contentHash)), sig(sig),
          debugOrigin(internString(debugOrigin))
    {
    }

    // Interned names, shapes and strides are equal exactly when their
    // pointers are; engine lists compare as before (by index and type).
    bool operator==(const ENode &other) const
    {
        return kernelId == other.kernelId && opType == other.opType && opName == other.opName &&
               children == other.children && shape == other.shape && strides == other.strides && dtype == other.dtype &&
               mem_space == other.mem_space && (engines == other.engines || *engines == *other.engines) &&
               contentHash == other.contentHash;
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
        return *opName;
    }
    const std::vector<EClassId> &getChildren() const
    {
        return children;
    }
    const std::vector<uint32_t> &getShape() const
    {
        return *shape;
    }
    const std::vector<uint64_t> &getStrides() const
    {
        return *strides;
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
        return *engines;
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
        return *debugOrigin;
    }

    // Overwrites every field, reusing this node's storage (for probing the
    // hashcons without allocating); the signature and debug origin are reset.
    void assignFields(KernelId newKernelId, OpType newOpType, const std::string &newOpName,
                      const std::vector<EClassId> &newChildren, const std::vector<uint32_t> &newShape,
                      const std::vector<uint64_t> &newStrides, DType newDType, MemSpace newMemSpace,
                      const std::vector<Engine> &newEngines, const std::string &newContentHash)
    {
        kernelId = newKernelId;
        opType = newOpType;
        opName = internString(newOpName);
        children.assign(newChildren.begin(), newChildren.end());
        // A reused probe often keeps its shape, strides and engines: compare
        // before hashing them into the intern table.
        if (!InternedVectorOps<uint32_t>{}(shape, &newShape))
            shape = internVector(newShape);
        if (!InternedVectorOps<uint64_t>{}(strides, &newStrides))
            strides = internVector(newStrides);
        dtype = newDType;
        mem_space = newMemSpace;
        if (!InternedVectorOps<Engine>{}(engines, &newEngines))
            engines = internVector(newEngines);
        contentHash.assign(newContentHash);
        sig = 0;
        debugOrigin = internString(std::string());
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
    void setDebugOrigin(const std::string &origin)
    {
        debugOrigin = internString(origin);
    }

  private:
    KernelId kernelId;
    OpType opType;
    const std::string *opName;
    std::vector<EClassId> children;
    const std::vector<uint32_t> *shape;
    const std::vector<uint64_t> *strides;
    DType dtype;
    MemSpace mem_space;
    const std::vector<Engine> *engines;
    std::string contentHash;
    uint64_t sig;
    const std::string *debugOrigin;
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
    // Interned: copying a class (copy-on-write, compaction) copies only its
    // e-node list.
    InternedVector<uint32_t> shape;
    InternedVector<uint64_t> strides;
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
    // True while `hashcons` holds exactly the owned e-nodes that are not
    // duplicates, each under its stored signature, and every other owned
    // e-node equals one of them or a shared e-node (addENode and rebuild keep
    // this). rebuild then re-canonicalizes only e-nodes whose children
    // changed instead of re-hashing every owned e-node.
    bool hashcons_tracks_owned = true;
    // Owned e-nodes by child class (the child id as stored when indexed),
    // kept between congruence closures while valid; a class merged away
    // since (merged_since_closure) has its list appended to its root's when
    // the next closure visits its parents.
    struct OwnedParentIndex
    {
        bool valid = false;
        uint32_t indexed = 0;
        std::vector<uint32_t> head, tail, edge_next, edge_node;
    };
    OwnedParentIndex owned_parents;
    std::vector<uint32_t> merged_since_closure;
    // With no owned e-nodes the empty index is complete (shared e-nodes are
    // never re-canonicalized); otherwise the next closure rebuilds it.
    void resetOwnedParents()
    {
        owned_parents = OwnedParentIndex{};
        owned_parents.valid = enodes.empty();
    }

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
            hashcons_tracks_owned = other.hashcons_tracks_owned;
            resetOwnedParents();
            merged_since_closure.clear();
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
            hashcons_tracks_owned = enodes.empty();
            resetOwnedParents();
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
        hashcons_tracks_owned = enodes.empty() && !shared_enodes;
        owned_parents = OwnedParentIndex{};
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
            // Owned positions shift: the parent index refers to the old ones.
            const_cast<EGraph *>(this)->owned_parents = OwnedParentIndex{};
        }
    }

    size_t getNumOwnedENodes() const { return enodes.size(); }

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
            hashcons_tracks_owned = hashcons.empty();
            resetOwnedParents();
        }
    }

    void clearENodes()
    {
        enodes.clear();
        shared_enodes.reset();
        hashcons_tracks_owned = hashcons.empty();
        owned_parents = OwnedParentIndex{};
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

    bool isCongruent(const ENode &a, const ENode &b) const
    {
        if (a == b)
            return true;
        if (a.getKernelId() != b.getKernelId() || a.getOpType() != b.getOpType() ||
            a.getOpName() != b.getOpName() || a.getDType() != b.getDType() ||
            a.getMemSpace() != b.getMemSpace() || a.getContentHash() != b.getContentHash() ||
            a.getShape() != b.getShape() || a.getStrides() != b.getStrides() ||
            a.getEngines() != b.getEngines())
            return false;
        const auto &ch_a = a.getChildren();
        const auto &ch_b = b.getChildren();
        if (ch_a.size() != ch_b.size())
            return false;
        for (size_t k = 0; k < ch_a.size(); ++k)
        {
            if (findConst(ch_a[k]) != findConst(ch_b[k]))
                return false;
        }
        return true;
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
        merged_since_closure.push_back(rb.value);
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

    // `keep_hashcons` false leaves the compacted graph without a hashcons
    // (as clearHashcons does) for callers that never add or look up enodes
    // afterwards.
    // Congruence closure when `hashcons` tracks the owned e-nodes: only owned
    // e-nodes with a non-canonical child (their stored signature is stale)
    // leave the hashcons, are re-canonicalized and are looked up again, in
    // increasing id order; a match merges the two classes, which can make
    // further e-nodes stale (the owned parents of merged-away classes, the
    // only e-nodes a merge can make stale). Ends with the same classes and the same set of
    // distinct owned e-nodes as re-hashing every owned e-node; only which of
    // two congruent e-nodes stays in the hashcons can differ. Shared e-nodes
    // are never re-canonicalized, as in the full pass.
    uint32_t closeCongruenceIncrementally(size_t base_count, uint32_t thread_count)
    {
        uint32_t total_merges = 0;
        const uint32_t num_owned = static_cast<uint32_t>(enodes.size());
        constexpr uint32_t kNone = UINT32_MAX;
        OwnedParentIndex &index = owned_parents;
        auto addEdges = [&](uint32_t from, uint32_t to) {
            const size_t num_classes = getNumClasses();
            if (index.head.size() < num_classes)
            {
                index.head.resize(num_classes, kNone);
                index.tail.resize(num_classes, kNone);
            }
            for (uint32_t i = from; i < to; ++i)
                for (EClassId child : enodes[i].getChildren())
                {
                    const uint32_t c = child.value;
                    const uint32_t edge = static_cast<uint32_t>(index.edge_node.size());
                    index.edge_node.push_back(i);
                    index.edge_next.push_back(kNone);
                    if (index.tail[c] == kNone)
                        index.head[c] = edge;
                    else
                        index.edge_next[index.tail[c]] = edge;
                    index.tail[c] = edge;
                }
        };
        std::vector<uint32_t> visited(num_owned, 0);
        uint32_t visit_epoch = 0;
        std::vector<uint32_t> stale_ids;
        // Appends a merged-away class's list to its root's, collecting its
        // e-nodes (once per epoch) as stale.
        auto foldMergedAway = [&](uint32_t loser) {
            const uint32_t root = findConst(EClassId{loser}).value;
            if (loser >= index.head.size() || index.head[loser] == kNone || root == loser)
                return;
            for (uint32_t edge = index.head[loser]; edge != kNone; edge = index.edge_next[edge])
            {
                const uint32_t node = index.edge_node[edge];
                if (visited[node] != visit_epoch)
                {
                    visited[node] = visit_epoch;
                    stale_ids.push_back(node);
                }
            }
            if (index.tail[root] == kNone)
                index.head[root] = index.head[loser];
            else
                index.edge_next[index.tail[root]] = index.head[loser];
            index.tail[root] = index.tail[loser];
            index.head[loser] = index.tail[loser] = kNone;
        };

        ++visit_epoch;
        if (!index.valid)
        {
            // First closure of this graph: every owned e-node with a
            // non-canonical child (or no signature yet); the index starts here.
            std::vector<uint8_t> stale(num_owned, 0);
            ThreadPool::get().parallel_for(thread_count, [&](uint32_t thread_idx) {
                for (uint32_t i = thread_idx; i < num_owned; i += thread_count)
                {
                    const ENode &node = enodes[i];
                    uint8_t is_stale = node.getSig() == 0 ? 1 : 0;
                    for (EClassId child : node.getChildren())
                        if (findConst(child) != child)
                        {
                            is_stale = 1;
                            break;
                        }
                    stale[i] = is_stale;
                    EClassId &owner = nodeToEClass[base_count + i];
                    owner = findConst(owner);
                }
            });
            for (uint32_t i = 0; i < num_owned; ++i)
                if (stale[i])
                    stale_ids.push_back(i);
            addEdges(0, num_owned);
            index.indexed = num_owned;
            index.valid = true;
        }
        else
        {
            // Since the last closure only merges made e-nodes stale: the
            // parents of the classes merged away (e-nodes added since are
            // indexed first, under the children they were added with).
            addEdges(index.indexed, num_owned);
            index.indexed = num_owned;
            for (uint32_t loser : merged_since_closure)
                foldMergedAway(loser);
            for (uint32_t i : stale_ids)
            {
                EClassId &owner = nodeToEClass[base_count + i];
                owner = findConst(owner);
            }
            std::sort(stale_ids.begin(), stale_ids.end());
        }
        merged_since_closure.clear();

        std::vector<std::pair<EClassId, EClassId>> merges;
        while (!stale_ids.empty())
        {
            for (uint32_t i : stale_ids)
            {
                auto it = hashcons.find(enodes[i].getSig());
                if (it == hashcons.end())
                    continue;
                const ENodeId id{static_cast<uint32_t>(base_count + i)};
                auto pos = std::find(it->second.begin(), it->second.end(), id);
                if (pos == it->second.end())
                    continue;
                it->second.erase(pos);
                if (it->second.empty())
                    hashcons.erase(it);
            }
            const uint32_t num_stale = static_cast<uint32_t>(stale_ids.size());
            ThreadPool::get().parallel_for(thread_count, [&](uint32_t thread_idx) {
                for (uint32_t k = thread_idx; k < num_stale; k += thread_count)
                {
                    ENode &node = enodes[stale_ids[k]];
                    std::vector<EClassId> children = node.getChildren();
                    for (EClassId &child : children)
                        child = findConst(child);
                    node.setChildren(std::move(children));
                    node.setSig(computeSignature(node));
                }
            });
            merges.clear();
            for (uint32_t i : stale_ids)
            {
                const ENode &node = enodes[i];
                const ENodeId id{static_cast<uint32_t>(base_count + i)};
                const EClassId cls = findConst(nodeToEClass[id.value]);
                EClassId duplicate_class{};
                bool duplicate = false;
                auto it = hashcons.find(node.getSig());
                if (it != hashcons.end())
                    for (ENodeId other : it->second)
                        if (node == getENode(other))
                        {
                            duplicate_class = findConst(nodeToEClass[other.value]);
                            duplicate = true;
                            break;
                        }
                if (!duplicate && shared_hashcons)
                {
                    auto shared_it = shared_hashcons->find(node.getSig());
                    if (shared_it != shared_hashcons->end())
                        for (ENodeId other : shared_it->second)
                            if (node == getENode(other))
                            {
                                duplicate_class = findConst(nodeToEClass[other.value]);
                                duplicate = true;
                                break;
                            }
                }
                if (!duplicate)
                    hashcons[node.getSig()].push_back(id);
                else if (duplicate_class != cls)
                    merges.push_back({duplicate_class, cls});
            }
            uint32_t n_merges = 0;
            for (const auto &[a, b] : merges)
                if (findConst(a) != findConst(b))
                {
                    merge(a, b);
                    ++n_merges;
                }
            total_merges += n_merges;
            if (n_merges == 0)
                break;
            ++visit_epoch;
            stale_ids.clear();
            std::vector<uint32_t> losers;
            losers.swap(merged_since_closure);
            for (uint32_t loser : losers)
                foldMergedAway(loser);
            std::sort(stale_ids.begin(), stale_ids.end());
        }
        merged_since_closure.clear();
        if (total_merges > 0)
            ThreadPool::get().parallel_for(thread_count, [&](uint32_t thread_idx) {
                for (uint32_t i = thread_idx; i < num_owned; i += thread_count)
                {
                    EClassId &owner = nodeToEClass[base_count + i];
                    owner = findConst(owner);
                }
            });
        return total_merges;
    }

    void rebuild(bool compact = false, bool keep_hashcons = true)
    {
        if (!keep_hashcons && !compact)
            Error::throw_err("[EGraph.rebuild] dropping the hashcons requires compaction");
        uint32_t total_merges = 0;
        const size_t base_count = shared_enodes ? shared_enodes->size() : 0;
        const uint32_t thread_count = std::max(1u, ThreadPool::get().get_num_threads());
        struct RebuildShard
        {
            std::unordered_map<uint64_t, std::vector<ENodeId>> hash;
            std::vector<std::pair<EClassId, EClassId>> merges;
        };
        std::vector<RebuildShard> shards(thread_count);
        if (hashcons_tracks_owned)
            total_merges = closeCongruenceIncrementally(base_count, thread_count);
        while (!hashcons_tracks_owned)
        {
            const uint32_t num_enodes = static_cast<uint32_t>(enodes.size());
            std::vector<std::vector<std::vector<uint32_t>>> local_indices(
                thread_count, std::vector<std::vector<uint32_t>>(thread_count));
            ThreadPool::get().parallel_for(thread_count, [&](uint32_t thread_idx) {
                for (uint32_t i = thread_idx; i < num_enodes; i += thread_count)
                {
                    ENode &node = enodes[i];
                    const ENodeId enode_id{static_cast<uint32_t>(base_count + i)};
                    bool children_changed = false;
                    for (EClassId child : node.getChildren())
                    {
                        if (findConst(child) != child)
                        {
                            children_changed = true;
                            break;
                        }
                    }
                    if (children_changed)
                    {
                        std::vector<EClassId> updated_children = node.getChildren();
                        for (EClassId &child : updated_children)
                            child = findConst(child);
                        node.setChildren(std::move(updated_children));
                        node.setSig(computeSignature(node));
                    }
                    else if (node.getSig() == 0)
                        node.setSig(computeSignature(node));

                    nodeToEClass[enode_id.value] = findConst(nodeToEClass[enode_id.value]);
                    const uint32_t shard_idx = static_cast<uint32_t>(node.getSig() % thread_count);
                    local_indices[thread_idx][shard_idx].push_back(i);
                }
            });

            std::vector<std::vector<uint32_t>> shard_indices(thread_count);
            for (uint32_t shard_idx = 0; shard_idx < thread_count; ++shard_idx)
            {
                size_t shard_size = 0;
                for (uint32_t thread_idx = 0; thread_idx < thread_count; ++thread_idx)
                    shard_size += local_indices[thread_idx][shard_idx].size();
                shard_indices[shard_idx].reserve(shard_size);
                for (uint32_t thread_idx = 0; thread_idx < thread_count; ++thread_idx)
                {
                    auto &indices = local_indices[thread_idx][shard_idx];
                    shard_indices[shard_idx].insert(shard_indices[shard_idx].end(), indices.begin(), indices.end());
                }
            }

            ThreadPool::get().parallel_for(thread_count, [&](uint32_t shard_idx) {
                auto &shard = shards[shard_idx];
                shard.hash.clear();
                shard.hash.reserve(shard_indices[shard_idx].size() * 2);
                shard.merges.clear();
                for (uint32_t i : shard_indices[shard_idx])
                {
                    ENode &node = enodes[i];
                    const ENodeId current_enode_id{static_cast<uint32_t>(base_count + i)};
                    const EClassId cls = nodeToEClass[current_enode_id.value];
                    auto &bucket = shard.hash[node.getSig()];
                    EClassId duplicate_class{};
                    bool duplicate = false;
                    for (ENodeId other_enode_id : bucket)
                    {
                        if (node == getENode(other_enode_id))
                        {
                            duplicate_class = nodeToEClass[other_enode_id.value];
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate && shared_hashcons)
                    {
                        auto it = shared_hashcons->find(node.getSig());
                        if (it != shared_hashcons->end())
                        {
                            for (ENodeId other_enode_id : it->second)
                            {
                                if (node == getENode(other_enode_id))
                                {
                                    duplicate_class = nodeToEClass[other_enode_id.value];
                                    duplicate = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (duplicate)
                    {
                        if (duplicate_class != cls)
                            shard.merges.push_back({duplicate_class, cls});
                    }
                    else
                        bucket.push_back(current_enode_id);
                }
            });

            uint32_t n_merges = 0;
            for (const RebuildShard &shard : shards)
                for (const auto &[a, b] : shard.merges)
                    if (findConst(a) != findConst(b))
                    {
                        merge(a, b);
                        ++n_merges;
                    }
            total_merges += n_merges;

            if (n_merges == 0)
            {
                hashcons.clear();
                hashcons.reserve(static_cast<size_t>(num_enodes) * 2);
                for (RebuildShard &shard : shards)
                    for (auto &[signature, enode_ids] : shard.hash)
                        hashcons.emplace(signature, std::move(enode_ids));
                hashcons_tracks_owned = true;
                owned_parents = OwnedParentIndex{};
                merged_since_closure.clear();
                break;
            }
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
        // The hashcons is built over every e-node, including ones pruned from
        // their class, so a congruence class's representative can be an
        // e-node no class lists. Its first listed congruent e-node stands in
        // for it; otherwise compaction would drop that listed e-node too and
        // could leave its class empty while parents still reference it.
        // Decided before any e-node is moved below.
        {
            std::vector<uint8_t> listed(getNumENodes(), 0);
            for (uint32_t i = 0; i < num_old_classes; ++i)
                if (find(EClassId{i}) == EClassId{i})
                    for (ENodeId nid : getEClass(EClassId{i}).enodes)
                        listed[nid.value] = 1;
            auto findRepresentative = [&](const ENode &node) -> ENodeId {
                auto it = hashcons.find(node.getSig());
                if (it != hashcons.end())
                    for (ENodeId other : it->second)
                        if (isCongruent(node, getENode(other)))
                            return other;
                if (shared_hashcons)
                {
                    auto shared_it = shared_hashcons->find(node.getSig());
                    if (shared_it != shared_hashcons->end())
                        for (ENodeId other : shared_it->second)
                            if (isCongruent(node, getENode(other)))
                                return other;
                }
                bool needs_canon = false;
                for (EClassId c : node.getChildren())
                {
                    if (findConst(c) != c)
                    {
                        needs_canon = true;
                        break;
                    }
                }
                if (needs_canon)
                {
                    ENode canon_node = node;
                    std::vector<EClassId> ch = canon_node.getChildren();
                    for (EClassId &c : ch)
                        c = findConst(c);
                    canon_node.setChildren(std::move(ch));
                    canon_node.setSig(computeSignature(canon_node));

                    auto it_c = hashcons.find(canon_node.getSig());
                    if (it_c != hashcons.end())
                        for (ENodeId other : it_c->second)
                            if (isCongruent(canon_node, getENode(other)))
                                return other;
                    if (shared_hashcons)
                    {
                        auto shared_it_c = shared_hashcons->find(canon_node.getSig());
                        if (shared_it_c != shared_hashcons->end())
                            for (ENodeId other : shared_it_c->second)
                                if (isCongruent(canon_node, getENode(other)))
                                    return other;
                    }
                }
                return ENodeId{UINT32_MAX};
            };
            // Congruent e-nodes share a class, and the hashcons holds one
            // e-node per congruence class: a listed alive e-node of the same
            // class equal to this one is its representative, found without
            // a hashcons lookup.
            std::unordered_map<uint32_t, uint32_t> stand_in;
            for (uint32_t i = 0; i < num_old_classes; ++i)
            {
                if (find(EClassId{i}) != EClassId{i})
                    continue;
                const std::vector<ENodeId> &class_enodes = getEClass(EClassId{i}).enodes;
                for (ENodeId nid : class_enodes)
                {
                    if (is_alive_node[nid.value])
                        continue;
                    const ENode &node = getENode(nid);
                    bool listed_representative = false;
                    for (ENodeId other : class_enodes)
                    {
                        if (!is_alive_node[other.value] || other == nid)
                            continue;
                        const ENode &candidate = getENode(other);
                        if (isCongruent(candidate, node))
                        {
                            listed_representative = true;
                            break;
                        }
                    }
                    if (listed_representative)
                        continue;
                    const ENodeId rep = findRepresentative(node);
                    if (rep.value == UINT32_MAX)
                    {
                        bool class_has_alive = false;
                        for (ENodeId other : class_enodes)
                        {
                            if (is_alive_node[other.value])
                            {
                                class_has_alive = true;
                                break;
                            }
                        }
                        if (class_has_alive)
                            continue;
                        is_alive_node[nid.value] = true;
                        continue;
                    }
                    if (listed[rep.value])
                        continue;
                    if (stand_in.emplace(rep.value, nid.value).second)
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

        // The old classes and enodes are discarded below, so the ones this
        // graph owns (not in a shared copy-on-write base) are moved, not
        // copied; each live class and alive enode is visited once.
        const size_t shared_enode_count = shared_enodes ? shared_enodes->size() : 0;
        for (uint32_t old_cls_idx = 0; old_cls_idx < num_old_classes; ++old_cls_idx)
        {
            EClassId old_cls_id{old_cls_idx};
            if (find(old_cls_id) != old_cls_id)
                continue;

            const bool owned_class = !shared_classes;
            const EClass &old_cls = owned_class ? classes[old_cls_idx] : getEClass(old_cls_id);
            EClassId new_cls_id = old_to_new_class[old_cls_idx];

            EClass new_cls;
            new_cls.id = new_cls_id;
            new_cls.base_eclass_id = old_cls.base_eclass_id;
            if (owned_class)
            {
                new_cls.shape = std::move(classes[old_cls_idx].shape);
                new_cls.strides = std::move(classes[old_cls_idx].strides);
            }
            else
            {
                new_cls.shape = old_cls.shape;
                new_cls.strides = old_cls.strides;
            }
            new_cls.dtype = old_cls.dtype;
            new_cls.mem_space = old_cls.mem_space;
            new_cls.logical_id = old_cls.logical_id;
            new_cls.is_clean = isClean(old_cls_id);
            new_cls.enodes.reserve(old_cls.enodes.size());

            for (ENodeId old_nid : old_cls.enodes)
            {
                if (!is_alive_node[old_nid.value])
                    continue;

                const ENodeId moved_to = old_to_new_node[old_nid.value];
                ENodeId new_nid{static_cast<uint32_t>(new_enodes.size())};
                old_to_new_node[old_nid.value] = new_nid;

                // Listed again: the earlier copy is already renumbered.
                const bool renumbered = moved_to.value != UINT32_MAX;
                ENode node = [&]() -> ENode {
                    if (renumbered)
                        return new_enodes[moved_to.value];
                    if (old_nid.value >= shared_enode_count)
                        return std::move(enodes[old_nid.value - shared_enode_count]);
                    return getENode(old_nid);
                }();
                if (!renumbered)
                {
                    std::vector<EClassId> children = node.getChildren();
                    for (EClassId &child : children)
                    {
                        child = old_to_new_class[child.value];
                    }
                    node.setChildren(std::move(children));
                    node.setSig(computeSignature(node));
                }

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
        if (keep_hashcons)
        {
            new_hashcons.reserve(new_enodes.size() * 2);
            for (uint32_t i = 0; i < new_enodes.size(); ++i)
            {
                new_hashcons[new_enodes[i].getSig()].push_back(ENodeId{i});
            }
        }

        LOG(DEBUG) << "[EGraph.rebuild] Compacted egraph: " << num_old_classes << " -> " << new_classes.size()
                  << " classes, " << getNumENodes() << " -> " << new_enodes.size() << " enodes" << std::endl;

        // Freeing the replaced containers takes most of a second on large
        // graphs; nothing references them, so free them off this thread.
        std::thread([old_classes = std::move(classes), old_enodes = std::move(enodes),
                     old_hashcons = std::move(hashcons)]() mutable {}).detach();
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
        hashcons_tracks_owned = keep_hashcons;
        owned_parents = OwnedParentIndex{};
        merged_since_closure.clear();

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
        ensureBaseEClassIndex();
        auto it = baseEClassToEClass.find(base_id);
        return it == baseEClassToEClass.end() ? EClassId{} : it->second;
    }

    // Builds the base id -> class index findEClassByBaseId uses (lazily
    // built otherwise; graphs can build theirs concurrently).
    void ensureBaseEClassIndex() const
    {
        if (baseEClassIndexInitialized)
            return;
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
