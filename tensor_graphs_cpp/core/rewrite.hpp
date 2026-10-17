#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#ifdef TG_PROFILE
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#endif
#include <cstring>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/plan/bucket_regions.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/kernel_pattern_cache.hpp"
#include "core/ops/ops.hpp"
#include "core/loaders/tg_store.hpp"
#include "core/shape_propagator.hpp"

// Diagnostics switch, read once (getenv takes the CRT environment lock).
inline bool debugEnvTgDebugRoots()
{
    static const bool enabled = std::getenv("TG_DEBUG_ROOTS") != nullptr;
    return enabled;
}

inline std::vector<std::vector<MemSpace>> findMemSpacePaths(MemSpace src, MemSpace dst, const TensorNode &node,
                                                            const std::vector<Engine> &engines)
{
    if (src == dst)
        return {{src}};

    const auto &all_spaces = System::get().getAvailableMemSpaces();
    std::unordered_map<MemSpace, std::vector<MemSpace>> adj;

    for (const auto &s1 : all_spaces)
    {
        for (const auto &s2 : all_spaces)
        {
            if (s1 == s2)
                continue;
            TensorNode dummyIn = node;
            TensorNode dummyOut = node;
            auto refs = KernelRegistry::get().findMatchingKernels(OpType::COPY_TO, "", {dummyIn}, dummyOut, false, s2,
                                                                  {s1}, engines, false, false, true, true);
            if (!refs.empty())
            {
                adj[s1].push_back(s2);
            }
        }
    }

    std::vector<std::vector<MemSpace>> all_paths;
    std::vector<MemSpace> current_path = {src};
    std::unordered_set<MemSpace> visited = {src};

    std::function<void(MemSpace)> dfs = [&](MemSpace curr) {
        if (curr == dst)
        {
            all_paths.push_back(current_path);
            return;
        }

        auto it = adj.find(curr);
        if (it == adj.end())
            return;

        for (MemSpace next : it->second)
        {
            if (visited.find(next) == visited.end())
            {
                visited.insert(next);
                current_path.push_back(next);
                dfs(next);
                current_path.pop_back();
                visited.erase(next);
            }
        }
    };

    dfs(src);
    return all_paths;
}

inline bool isEClassProtected(EClassId e_class_id, const std::unordered_set<EClassId> &protectedEClasses,
                              const EGraph &egraph)
{
    EClassId canon = egraph.findConst(e_class_id);
    if (protectedEClasses.count(canon))
        return true;
    for (EClassId id : protectedEClasses)
    {
        if (egraph.findConst(id) == canon)
            return true;
    }
    return false;
}

struct SliceMemoKey
{
    uint32_t cid;
    Region box;
    bool operator==(const SliceMemoKey &o) const
    {
        return cid == o.cid && box == o.box;
    }
};
struct SliceMemoKeyHash
{
    size_t operator()(const SliceMemoKey &k) const
    {
        size_t h = std::hash<uint32_t>{}(k.cid);
        for (const auto &d : k.box.region)
        {
            h = h * 31 + d.start;
            h = h * 31 + d.stop;
        }
        return h;
    }
};
using SliceMemoTable = std::unordered_map<SliceMemoKey, EClassId, SliceMemoKeyHash>;

struct RuleCtx
{
    EGraph &egraph;
    const std::unordered_set<EClassId> &protectedEClasses;
    TGStore *repo;
    CostModel *costModel = nullptr;
    // Dirty/needed regions of the bucket being saturated (partial buckets).
    const BucketRegions *regions = nullptr;
    uint32_t base_enode_watermark = 0;
    uint32_t base_class_watermark = 0;
    const std::vector<uint8_t> *sweep_filter = nullptr;
    const std::vector<uint8_t> *base_staged = nullptr;
    SliceMemoTable *slice_memo = nullptr;
};

// "Tensor = its previous (cached) value with `box` recomputed" (a SCATTER of
// the box's computation into the tensor's buffer) only holds if the box
// contains every element that changed and still matters: all of the
// tensor's dirty & needed elements in this bucket. Without bucket regions
// (no partial bucket) the slice push-down rules keep their original form.
inline bool scatterFormValid(const RuleCtx &ctx, const EClass &cls, const Region &box)
{
    if (!ctx.regions || cls.base_eclass_id == BaseEClassId{})
        return false;
    if (cls.logical_id == LogicalId{})
        return false;
    if (!isContiguous(cls))
        return false;
    return ctx.regions->coversLive(cls.logical_id, box);
}

inline bool hasScatter(const EGraph &egraph, EClassId cid)
{
    const auto &enodes = egraph.getEClass(egraph.findConst(cid)).enodes;
    for (ENodeId id : enodes)
    {
        if (egraph.getENode(id).getOpType() == OpType::SCATTER)
            return true;
    }
    return false;
}

inline const int32_t *constIntsData(const EGraph &egraph, EClassId cid, size_t &count)
{
    cid = egraph.findConst(cid);
    auto it = egraph.constantStaging.find(cid);
    if (it == egraph.constantStaging.end() || it->second == nullptr)
        return nullptr;
    count = it->second->size() / sizeof(int32_t);
    return reinterpret_cast<const int32_t *>(it->second->data());
}

inline bool constInts(const EGraph &egraph, EClassId cid, std::vector<int32_t> &out)
{
    size_t count = 0;
    const int32_t *data = constIntsData(egraph, cid, count);
    if (!data)
        return false;
    out.assign(data, data + count);
    return true;
}

// Box of a slice node in its source's coordinates; false unless it is a
// proper (non-full) unit-step slice.
inline bool sliceBox(const EGraph &egraph, const ENode &sliceNode, const std::vector<uint32_t> &srcShape,
                     Region &box)
{
    size_t st_len = 0, en_len = 0, sp_len = 0;
    const int32_t *starts = constIntsData(egraph, sliceNode.getChildren()[1], st_len);
    const int32_t *ends = constIntsData(egraph, sliceNode.getChildren()[2], en_len);
    const int32_t *steps = constIntsData(egraph, sliceNode.getChildren()[3], sp_len);
    if (!starts || !ends || !steps)
        return false;
    if (st_len != srcShape.size() || en_len != srcShape.size() || sp_len != srcShape.size())
        return false;
    box.region.clear();
    bool full = true;
    for (size_t d = 0; d < srcShape.size(); ++d)
    {
        int32_t st = starts[d] < 0 ? starts[d] + (int32_t)srcShape[d] : starts[d];
        int32_t en = ends[d] < 0 ? ends[d] + (int32_t)srcShape[d] : ends[d];
        if (steps[d] != 1 || st < 0 || en > (int32_t)srcShape[d] || st >= en)
            return false;
        box.region.push_back({(uint32_t)st, (uint32_t)en});
        full = full && st == 0 && en == (int32_t)srcShape[d];
    }
    return !full;
}

inline std::vector<uint32_t> boxShape(const Region &box)
{
    std::vector<uint32_t> shape;
    for (const Dim &d : box.region)
        shape.push_back(d.stop - d.start);
    return shape;
}

inline Region fullBox(const std::vector<uint32_t> &shape)
{
    Region r;
    for (uint32_t s : shape)
        r.region.push_back({0, s});
    return r;
}

inline Region regionFromBounds(const std::vector<int32_t> &starts, const std::vector<int32_t> &ends,
                               const std::vector<uint32_t> &shape)
{
    Region box;
    for (size_t d = 0; d < shape.size(); ++d)
    {
        int32_t st = d < starts.size() ? starts[d] : 0;
        int32_t en = d < ends.size() ? ends[d] : (int32_t)shape[d];
        if (st < 0)
            st += (int32_t)shape[d];
        if (en < 0)
            en += (int32_t)shape[d];
        st = std::max(0, std::min(st, (int32_t)shape[d]));
        en = std::max(st, std::min(en, (int32_t)shape[d]));
        box.region.push_back({(uint32_t)st, (uint32_t)en});
    }
    return box;
}

// contiguous(slice(src, box)) equals src's op restricted to box. Record that
// equality (always valid) when both classes have the same layout; e-classes
// carry their layout, so differing layouts must not be merged.
inline void mergeRestricted(EGraph &egraph, EClassId contigClass, EClassId restricted)
{
    contigClass = egraph.findConst(contigClass);
    restricted = egraph.findConst(restricted);
    if (contigClass == restricted)
        return;
    {
        const EClass &a = egraph.getEClass(contigClass);
        const EClass &b = egraph.getEClass(restricted);
        if (a.shape != b.shape || a.dtype != b.dtype)
            Error::throw_err("[mergeRestricted] restricted computation does not match its slice: " + toString(a.shape) +
                             " vs " + toString(b.shape));
        if (a.strides != b.strides || a.mem_space != b.mem_space)
            return;
    }
    egraph.merge(contigClass, restricted);
}

// A stateful update's identity includes the persistent value whose untouched
// cells it retains. This prevents hash-consing updates to different owners.
inline std::string scatterFormTag(const EGraph &egraph, EClassId owner)
{
    const EClass &cls = egraph.getEClass(egraph.findConst(owner));
    if (cls.base_eclass_id == BaseEClassId{})
        Error::throw_err("[scatterFormTag] cached update has no persistent identity");
    return "scatter-into:b" + std::to_string(cls.base_eclass_id.value);
}

struct Rule
{
    virtual ~Rule() = default;
    virtual std::string name() const = 0;
    virtual bool match(uint32_t eNodeIdx, RuleCtx &ctx) = 0;
    virtual void apply(uint32_t eNodeIdx, RuleCtx &ctx) = 0;
    // Copy including match state (visited sets, completed applications).
    // A clone continues saturation on a copy of the e-graph this rule has
    // already saturated, whose enode and class IDs are unchanged.
    virtual std::unique_ptr<Rule> clone() const = 0;
    uint32_t base_enode_watermark = 0;
    virtual void clearVisited() {}
    virtual bool canMatchOp(OpType op) const { return true; }
    virtual bool prepareSweep(RuleCtx &ctx) { return true; }
};

// Enabled only in --profile builds. Keeping the instrumentation at the Rule
// dispatch boundary measures every rewrite implementation without requiring
// boilerplate in each individual rule.
#ifdef TG_PROFILE
struct RewriteRuleTimingStats
{
    uint64_t constructor_ns = 0;
    uint64_t constructor_calls = 0;
    uint64_t match_ns = 0;
    uint64_t match_calls = 0;
    uint64_t match_hits = 0;
    uint64_t apply_ns = 0;
    uint64_t apply_calls = 0;
};

class RewriteRuleProfiler
{
  public:
    static RewriteRuleProfiler &get()
    {
        static RewriteRuleProfiler instance;
        return instance;
    }

    void recordConstructor(const std::string &name, uint64_t ns)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto &stats = stats_by_rule[name];
        stats.constructor_ns += ns;
        stats.constructor_calls++;
    }

    void recordMatch(const std::string &name, uint64_t ns, bool matched)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto &stats = stats_by_rule[name];
        stats.match_ns += ns;
        stats.match_calls++;
        if (matched)
            stats.match_hits++;
    }

    void recordApply(const std::string &name, uint64_t ns)
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto &stats = stats_by_rule[name];
        stats.apply_ns += ns;
        stats.apply_calls++;
    }

    void printSummary() const
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (stats_by_rule.empty())
            return;

        std::cout << "\n========================================================================================================\n";
        std::cout << " [Rewrite Rule Profiling Report]\n";
        std::cout << "========================================================================================================\n";
        std::cout << std::left << std::setw(34) << "Rule Name" << std::right << std::setw(12) << "Ctor (ms)"
                  << std::setw(12) << "Ctor Calls" << std::setw(12) << "Match (ms)" << std::setw(14)
                  << "Match Calls" << std::setw(10) << "Hits" << std::setw(12) << "Apply (ms)" << std::setw(12)
                  << "Total (ms)\n";
        std::cout << std::string(118, '-') << "\n";

        for (const auto &[name, stats] : stats_by_rule)
        {
            double constructor_ms = stats.constructor_ns / 1e6;
            double match_ms = stats.match_ns / 1e6;
            double apply_ms = stats.apply_ns / 1e6;
            double total_ms = constructor_ms + match_ms + apply_ms;
            std::cout << std::left << std::setw(34) << name.substr(0, 33) << std::right << std::fixed
                      << std::setprecision(2) << std::setw(12) << constructor_ms << std::setw(12)
                      << stats.constructor_calls << std::setw(12) << match_ms << std::setw(14) << stats.match_calls
                      << std::setw(10) << stats.match_hits << std::setw(12) << apply_ms << std::setw(12) << total_ms
                      << "\n";
        }
        std::cout << "========================================================================================================\n\n"
                  << std::flush;
    }

  private:
    mutable std::mutex mutex;
    std::unordered_map<std::string, RewriteRuleTimingStats> stats_by_rule;
};

template <typename RuleT, typename... Args> std::unique_ptr<Rule> makeProfiledRewriteRule(Args &&...args)
{
    auto start_time = std::chrono::steady_clock::now();
    auto rule = std::make_unique<RuleT>(std::forward<Args>(args)...);
    auto end_time = std::chrono::steady_clock::now();
    RewriteRuleProfiler::get().recordConstructor(
        rule->name(), std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    return rule;
}

inline void printRewriteProfileSummary()
{
    RewriteRuleProfiler::get().printSummary();
}
#else
template <typename RuleT, typename... Args> std::unique_ptr<Rule> makeProfiledRewriteRule(Args &&...args)
{
    return std::make_unique<RuleT>(std::forward<Args>(args)...);
}

inline void printRewriteProfileSummary()
{
}
#endif

inline EClassId addOpToEGraph(EGraph &egraph, OpType op, const std::vector<EClassId> &children,
                              const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides, DType dtype,
                              MemSpace mem_space, EClassId targetEClass = EClassId(),
                              SourceLocation loc = SourceLocation::current(), const std::string &debugOrigin = "",
                              const std::string &contentHash = "")
{
    EClassId cls = targetEClass;
    std::string origin = debugOrigin.empty() ? toString(loc) : debugOrigin;

    struct MatchedKernel
    {
        KernelId uid;
        std::vector<Engine> engines;
    };
    struct OpMatches
    {
        bool hasPattern = false;
        std::vector<MatchedKernel> kernels;
    };
    struct OpMatchKey
    {
        uint64_t h1;
        uint64_t h2;
        bool operator==(const OpMatchKey &o) const noexcept { return h1 == o.h1 && h2 == o.h2; }
    };
    struct OpMatchKeyHash
    {
        size_t operator()(const OpMatchKey &k) const noexcept {
            return static_cast<size_t>(k.h1 ^ (k.h2 * 1099511628211ULL));
        }
    };
    thread_local std::unordered_map<OpMatchKey, OpMatches, OpMatchKeyHash> opMatchMemo;

    uint64_t h1 = 14695981039346656037ULL;
    uint64_t h2 = 1099511628211ULL;
    auto addHash = [&](const void *data, size_t size) {
        const uint8_t *p = static_cast<const uint8_t *>(data);
        const size_t words = size / 8;
        for (size_t i = 0; i < words; ++i) {
            uint64_t w;
            std::memcpy(&w, p + i * 8, 8);
            h1 = (h1 ^ w) * 1099511628211ULL;
            h2 = (h2 ^ (w * 14695981039346656037ULL)) * 1099511628211ULL;
        }
        for (size_t i = words * 8; i < size; ++i) {
            h1 = (h1 ^ p[i]) * 1099511628211ULL;
            h2 = (h2 ^ p[i]) * 14695981039346656037ULL;
        }
    };

    const uint32_t opValue = static_cast<uint32_t>(op), dtypeValue = static_cast<uint32_t>(dtype);
    addHash(&opValue, sizeof(opValue));
    addHash(&dtypeValue, sizeof(dtypeValue));
    addHash(shape.data(), shape.size() * sizeof(uint32_t));
    addHash(strides.data(), strides.size() * sizeof(uint64_t));
    addHash(&mem_space, sizeof(mem_space));
    for (EClassId c : children)
    {
        const EClass &childCls = egraph.getEClass(egraph.findConst(c));
        const uint32_t cd = static_cast<uint32_t>(childCls.dtype);
        addHash(&cd, sizeof(cd));
        addHash(childCls.shape.data(), childCls.shape.size() * sizeof(uint32_t));
        addHash(childCls.strides.data(), childCls.strides.size() * sizeof(uint64_t));
        addHash(&childCls.mem_space, sizeof(MemSpace));
    }
    OpMatchKey key{h1, h2};
    auto memo = opMatchMemo.find(key);
    if (memo == opMatchMemo.end())
    {
        TensorNode outNode;
        outNode.opType = op;
        outNode.dtype = dtype;
        outNode.setShape(shape);
        outNode.strides = strides;

        std::vector<TensorNode> inNodes;
        std::vector<MemSpace> input_mem_spaces;
        for (EClassId c : children)
        {
            const EClass &childCls = egraph.getEClass(egraph.findConst(c));
            TensorNode in;
            in.opType = OpType::INPUT;
            in.dtype = childCls.dtype;
            in.setShape(childCls.shape);
            in.strides = childCls.strides;
            inNodes.push_back(std::move(in));
            input_mem_spaces.push_back(childCls.mem_space);
        }

        OpMatches result;
        Graph pGraph;
        std::vector<LogicalId> pInputs;
        for (auto &in : inNodes)
            pInputs.push_back(pGraph.input(in.getShape(), in.dtype));
        const auto &traits = getOpTraits(op);
        LogicalId pRoot = traits.buildPattern ? traits.buildPattern(pGraph, pInputs, dtype) : LogicalId();
        if (pRoot != LogicalId())
        {
            result.hasPattern = true;
            auto matches = KernelPatternCache::get().find(pGraph, pRoot, inNodes,
                                                          outNode, false, mem_space, input_mem_spaces, {}, false,
                                                          false, true, false);
            for (KernelId uid : matches)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                std::vector<Engine> actual_engines;
                kernel.matches(inNodes, outNode, mem_space, input_mem_spaces, {}, false, false, true, true,
                               &actual_engines);
                result.kernels.push_back({uid, std::move(actual_engines)});
            }
        }
        memo = opMatchMemo.emplace(key, std::move(result)).first;
    }

    if (memo->second.hasPattern)
    {
        if (memo->second.kernels.empty())
        {
            std::stringstream ss;
            ss << "\n[addOpToEGraph] No matching kernel found for the given configuration at " << toString(loc) << "\n"
               << "  Operation:       " << toString(op) << "\n"
               << "  Target MemSpace: " << toString(mem_space) << "\n"
               << "  Expected Output: "
               << "dtype=" << toString(dtype) << ", shape=" << toString(shape) << ", strides=" << toString(strides)
               << "\n"
               << "  Inputs (" << children.size() << ")\n";
            Error::throw_err(ss.str());
        }
        for (const MatchedKernel &matched : memo->second.kernels)
        {
            const auto &kernel = KernelRegistry::get().getKernel(matched.uid);

            std::vector<EClassId> adapted_children = children;
            for (size_t input_idx = 0; input_idx < adapted_children.size(); ++input_idx)
            {
                adapted_children[input_idx] = egraph.findConst(adapted_children[input_idx]);
                if (kernel.requiresContiguous.empty())
                    continue;

                const size_t rule_idx = std::min(input_idx, kernel.requiresContiguous.size() - 1);
                if (!kernel.requiresContiguous[rule_idx])
                    continue;

                const EClassId source_class = adapted_children[input_idx];
                std::vector<uint32_t> srcShape;
                DType srcDType;
                MemSpace srcMem;
                bool needContig = false;
                {
                    const EClass &source = egraph.getEClass(source_class);
                    if (!isContiguous(source))
                    {
                        needContig = true;
                        srcShape = source.shape;
                        srcDType = source.dtype;
                        srcMem = source.mem_space;
                    }
                }
                if (needContig)
                {
                    EClassId contiguous_class = addOpToEGraph(
                        egraph, OpType::CONTIGUOUS, {source_class}, srcShape,
                        calcContiguousStrides(srcShape), srcDType, srcMem, EClassId(), loc,
                        origin.empty() ? "addOpToEGraph.make_contiguous" : (origin + " -> make_contiguous"));
                    adapted_children[input_idx] = contiguous_class;
                }
            }

            ENode n(matched.uid, op, kernel.opName, adapted_children, shape,
                    kernel.is_view ? strides : calcContiguousStrides(shape), dtype, mem_space, matched.engines,
                    contentHash, 0, origin);
            n.setSig(EGraph::computeSignature(n));
            if (cls == EClassId())
                cls = egraph.lookupENode(n);
            if (cls == EClassId())
            {
                cls = egraph.addEClass(shape, strides, dtype, mem_space);
                cls = egraph.addENode(cls, std::move(n));
            }
            else if (targetEClass != EClassId())
            {
                cls = egraph.addENode(cls, std::move(n));
            }
        }
    }
    if (cls == EClassId())
        cls = egraph.addEClass(shape, strides, dtype, mem_space);
    return cls;
}

inline EClassId copyTo(EGraph &egraph, EClassId class_id, MemSpace target_mem_space,
                       const std::string &debugOrigin = "")
{
    EClassId canon = egraph.find(class_id);
    {
        const EClass &cls = egraph.getEClass(canon);
        if (cls.mem_space == target_mem_space)
            return canon;
    }

    std::vector<uint32_t> clsShape;
    std::vector<uint64_t> clsStrides;
    DType clsDType;
    MemSpace clsMem;
    bool needsContig = false;
    {
        const EClass &cls = egraph.getEClass(canon);
        if (!isContiguous(cls))
        {
            needsContig = true;
            clsShape = cls.shape;
            clsDType = cls.dtype;
            clsMem = cls.mem_space;
        }
    }
    if (needsContig)
    {
        canon = addOpToEGraph(egraph, OpType::CONTIGUOUS, {canon}, clsShape,
                              calcContiguousStrides(clsShape), clsDType, clsMem,
                              EClassId(), SourceLocation::current(),
                              debugOrigin.empty() ? "copyTo.make_contiguous" : (debugOrigin + " -> make_contiguous"));
    }
    {
        const EClass &cls = egraph.getEClass(canon);
        clsShape = cls.shape;
        clsStrides = cls.strides;
        clsDType = cls.dtype;
    }

    return addOpToEGraph(egraph, OpType::COPY_TO, {canon}, clsShape, clsStrides, clsDType, target_mem_space,
                         EClassId(), SourceLocation::current(), debugOrigin);
}

inline EClassId createCacheInputNode(EGraph &egraph, EClassId sourceClassId,
                                     const std::string &debugOrigin = "")
{
    EClassId canonSrcClass = egraph.findConst(sourceClassId);
    std::vector<uint32_t> shape;
    std::vector<uint64_t> strides;
    DType dtype;
    MemSpace mem_space;
    LogicalId srcLogicalId;
    {
        const EClass &srcClass = egraph.getEClass(canonSrcClass);
        shape = srcClass.shape;
        strides = srcClass.strides;
        dtype = srcClass.dtype;
        mem_space = srcClass.mem_space;
        srcLogicalId = srcClass.logical_id;
    }

    EClassId op_cache =
        egraph.addEClass(shape, strides, dtype, mem_space, srcLogicalId);

    ENode cacheNode(KernelId{0}, OpType::CACHE, "", {}, shape, strides, dtype,
                    mem_space, {}, toString(srcLogicalId), 0, debugOrigin);
    op_cache = egraph.addENode(op_cache, cacheNode);

    return op_cache;
}

struct FusionRule : public Rule
{
    std::unique_ptr<Rule> clone() const override
    {
        return std::make_unique<FusionRule>(*this);
    }

    std::string name() const override
    {
        return "FusionRule";
    }

    struct Pattern
    {
        std::string opName;
        OpType rootOpType;
        LogicalId rootId;
        std::vector<LogicalId> variables;
        std::vector<DType> dtypes;
        std::vector<std::vector<uint32_t>> dummyShapes;
        Graph graph;
        std::string pattern_hash;
        bool is_single_node = false;
        // Levels from the root to its deepest variable.
        uint32_t depth = 0;
        // Pattern node id -> index into `variables`, or -1.
        std::vector<int32_t> varIndex;
        mutable std::vector<KernelId> candidateKernelIds;
        mutable bool candidate_kernels_resolved = false;
        const std::vector<KernelId> &getCandidateKernelIds() const
        {
            if (!candidate_kernels_resolved)
            {
                candidateKernelIds = KernelPatternCache::get().getCandidates(graph, rootId, false);
                candidate_kernels_resolved = true;
            }
            return candidateKernelIds;
        }
    };

    // Pattern-variable bindings with an undo trail, so backtracking does not
    // copy the binding at every candidate enode.
    struct Binding
    {
        std::vector<EClassId> slots;
        std::vector<uint32_t> trail;
        explicit Binding(size_t vars) : slots(vars) {}
        void undo(size_t mark)
        {
            while (trail.size() > mark)
            {
                slots[trail.back()] = EClassId{};
                trail.pop_back();
            }
        }
    };

    struct MatchResult
    {
        const Pattern *pattern;
        std::vector<EClassId> inputs;
    };

    struct PatternSet
    {
        static constexpr size_t kNumOpTypes = static_cast<size_t>(OpType::FUSED) + 1;
        std::array<std::vector<Pattern>, kNumOpTypes> singleNodePatternsByOp;
        std::array<std::vector<Pattern>, kNumOpTypes> multiNodePatternsByOp;
        uint32_t max_pattern_depth = 0;
    };
    // Immutable after construction and shared by clones, so the Pattern
    // addresses in completed_applications stay valid across clones.
    std::shared_ptr<const PatternSet> patterns;
    std::vector<MatchResult> activeMatches;
    std::vector<uint8_t> single_node_visited;

    // Multi-node patterns read only classes within this many levels below
    // the root enode. During a sweep, enodes outside `sweep_filter` (set by
    // the saturation driver: none of those classes changed since the
    // previous sweep) cannot match anything new and are skipped.
    uint32_t max_pattern_depth = 0;
    const std::vector<uint8_t> *sweep_filter = nullptr;

    static uint32_t patternDepth(const Graph &graph, LogicalId id)
    {
        uint32_t depth = 0;
        for (LogicalId child : graph.getNode(id).child_ids)
            depth = std::max(depth, 1 + patternDepth(graph, child));
        return depth;
    }

    struct ApplicationKey
    {
        uint32_t root_idx;
        const Pattern *pattern;

        bool operator==(const ApplicationKey &other) const
        {
            return root_idx == other.root_idx && pattern == other.pattern;
        }
    };

    struct ApplicationHash
    {
        size_t operator()(const ApplicationKey &key) const
        {
            return (static_cast<size_t>(key.root_idx) * 0x9E3779B97F4A7C15ULL) ^
                   reinterpret_cast<uintptr_t>(key.pattern);
        }
    };

    std::shared_ptr<const std::unordered_set<ApplicationKey, ApplicationHash>> base_completed_applications;
    std::unordered_set<ApplicationKey, ApplicationHash> local_completed_applications;
    uint64_t application_cache_hits = 0;

    bool hasCompleted(const ApplicationKey &key) const
    {
        if (base_completed_applications && base_completed_applications->count(key))
            return true;
        return local_completed_applications.count(key) > 0;
    }

    size_t numCompletedApplications() const
    {
        return (base_completed_applications ? base_completed_applications->size() : 0) +
               local_completed_applications.size();
    }

    bool isCompleted(uint32_t eNodeIdx, const Pattern &pattern, const std::vector<EClassId> &inputs, const EGraph &egraph) const
    {
        return hasCompleted(ApplicationKey{eNodeIdx, &pattern});
    }

    // Candidate selection per signature (see selectFusedCandidates). Kept per
    // thread (no locking); the key includes the cost model it was made with.
    struct FusedCandidate;
    struct SelectionKey
    {
        uint64_t h1;
        uint64_t h2;
        bool operator==(const SelectionKey &o) const noexcept { return h1 == o.h1 && h2 == o.h2; }
    };
    struct SelectionKeyHash
    {
        size_t operator()(const SelectionKey &k) const noexcept {
            return static_cast<size_t>(k.h1 ^ (k.h2 * 1099511628211ULL));
        }
    };
    using SelectionMemo = std::unordered_map<SelectionKey, std::shared_ptr<const std::vector<FusedCandidate>>, SelectionKeyHash>;
    static SelectionMemo &selectionMemo()
    {
        thread_local SelectionMemo memo;
        return memo;
    }

    static void appendSignature(std::string &key, const void *data, size_t size)
    {
        key.append(reinterpret_cast<const char *>(&size), sizeof(size));
        key.append(static_cast<const char *>(data), size);
    }

    ~FusionRule() override
    {
        LOG(DEBUG) << "Fusion application cache: " << application_cache_hits << " hits, "
                   << numCompletedApplications() << " completed applications";
    }

    static std::shared_ptr<const PatternSet> getSharedPatterns(bool disableFusion)
    {
        if (disableFusion)
        {
            static auto empty_set = std::make_shared<PatternSet>();
            return empty_set;
        }
        static auto shared_set = []() {
            auto pattern_set = std::make_shared<PatternSet>();
            auto &singleNodePatternsByOp = pattern_set->singleNodePatternsByOp;
            auto &multiNodePatternsByOp = pattern_set->multiNodePatternsByOp;
            const auto &refGraphs = ReferenceGraphRegistry::get().getAll();
            for (const auto &pair : refGraphs)
            {
                const auto &entry = pair.second;
                Pattern pattern;
                pattern.opName = pair.first;

                for (uint64_t i = 0; i < entry.min_num_inputs; ++i)
                {
                    LogicalId inId = pattern.graph.input(entry.dummyShapes[i], entry.dtypes[i]);
                    pattern.variables.push_back(inId);
                }
                pattern.rootId = entry.factory(pattern.variables, pattern.graph);
                pattern.depth = patternDepth(pattern.graph, pattern.rootId);
                uint32_t maxNode = 0;
                for (const auto &nodePair : pattern.graph.nodes)
                    maxNode = std::max(maxNode, nodePair.first.value);
                pattern.varIndex.assign(maxNode + 1, -1);
                for (size_t v = 0; v < pattern.variables.size(); ++v)
                    pattern.varIndex[pattern.variables[v].value] = static_cast<int32_t>(v);

                pattern.rootOpType = pattern.graph.getNode(pattern.rootId).opType;
                pattern.pattern_hash = computeGraphHash(pattern.graph, {pattern.rootId});
                pattern.dtypes = entry.dtypes;
                pattern.dummyShapes = entry.dummyShapes;

                bool is_single = true;
                for (LogicalId cid : pattern.graph.getNode(pattern.rootId).child_ids)
                {
                    if (pattern.graph.getNode(cid).opType != OpType::INPUT)
                    {
                        is_single = false;
                        break;
                    }
                }
                pattern.is_single_node = is_single;

                if (is_single)
                {
                    singleNodePatternsByOp[static_cast<size_t>(pattern.rootOpType)].push_back(std::move(pattern));
                }
                else
                {
                    multiNodePatternsByOp[static_cast<size_t>(pattern.rootOpType)].push_back(std::move(pattern));
                }
            }
            for (const auto &list : pattern_set->multiNodePatternsByOp)
                for (const auto &pattern : list)
                    pattern_set->max_pattern_depth = std::max(pattern_set->max_pattern_depth, pattern.depth);
            return pattern_set;
        }();
        return shared_set;
    }

    FusionRule(bool disableFusion = false)
    {
        patterns = getSharedPatterns(disableFusion);
        max_pattern_depth = patterns->max_pattern_depth;
    }

    void clearVisited() override
    {
        single_node_visited.clear();
        if (!base_completed_applications && !local_completed_applications.empty())
        {
            base_completed_applications = std::make_shared<std::unordered_set<ApplicationKey, ApplicationHash>>(
                std::move(local_completed_applications));
            local_completed_applications.clear();
        }
    }

    bool canMatchOp(OpType op) const override
    {
        const size_t op_val = static_cast<size_t>(op);
        return patterns && op_val < PatternSet::kNumOpTypes &&
               (!patterns->singleNodePatternsByOp[op_val].empty() || !patterns->multiNodePatternsByOp[op_val].empty());
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        activeMatches.clear();
        const EGraph &egraph = ctx.egraph;
        const ENode &eNode = egraph.getENode(ENodeId{eNodeIdx});
        const size_t op_val = static_cast<size_t>(eNode.getOpType());
        if (op_val >= PatternSet::kNumOpTypes)
            return false;

        bool is_new_single = false;
        if (eNodeIdx >= ctx.base_enode_watermark)
        {
            if (eNodeIdx >= single_node_visited.size())
                single_node_visited.resize(std::max<size_t>(eNodeIdx + 128, egraph.getNumENodes()), 0);
            if (!single_node_visited[eNodeIdx])
            {
                single_node_visited[eNodeIdx] = 1;
                is_new_single = true;
            }
        }
        if (is_new_single)
        {
            const auto &single_list = patterns->singleNodePatternsByOp[op_val];
            for (const auto &pattern : single_list)
            {
                if (hasCompleted(ApplicationKey{eNodeIdx, &pattern}))
                    continue;
                Binding binding(pattern.variables.size());
                if (matchPatternNode(ENodeId{eNodeIdx}, egraph, pattern.rootId, pattern, binding, ctx.protectedEClasses))
                {
                    MatchResult mr;
                    mr.pattern = &pattern;
                    if (eNode.getOpType() == OpType::CONCAT && eNode.getChildren().size() > 2)
                    {
                        mr.inputs.reserve(eNode.getChildren().size());
                        mr.inputs.push_back(binding.slots[0]);
                        for (uint64_t i = 1; i < eNode.getChildren().size(); ++i)
                        {
                            mr.inputs.push_back(egraph.findConst(eNode.getChildren()[i]));
                        }
                    }
                    else
                    {
                        mr.inputs = std::move(binding.slots);
                    }

                    activeMatches.push_back(std::move(mr));
                }
            }
        }

        const auto &multi_list = patterns->multiNodePatternsByOp[op_val];
        if (!multi_list.empty())
        {
            uint32_t changedDistance = 0;
            bool skip_multi = false;
            const bool filtered = sweep_filter && eNodeIdx < sweep_filter->size();
            if (filtered)
            {
                changedDistance = (*sweep_filter)[eNodeIdx];
                if (changedDistance == 0)
                    skip_multi = true;
            }
            if (!skip_multi)
            {
                for (const auto &pattern : multi_list)
                {
                    if (filtered && pattern.depth < changedDistance)
                        continue;
                    if (hasCompleted(ApplicationKey{eNodeIdx, &pattern}))
                        continue;
                    Binding binding(pattern.variables.size());
                    bool has_new_node = (ctx.base_enode_watermark > 0 && eNodeIdx >= ctx.base_enode_watermark);
                    if (matchPatternNode(ENodeId{eNodeIdx}, egraph, pattern.rootId, pattern, binding, ctx.protectedEClasses,
                                         false, ctx.base_enode_watermark, ctx.base_class_watermark, ctx.base_staged, &has_new_node))
                    {
                        if (ctx.base_enode_watermark > 0 && !has_new_node)
                            continue;
                        MatchResult mr;
                        mr.pattern = &pattern;
                        if (eNode.getOpType() == OpType::CONCAT && eNode.getChildren().size() > 2)
                        {
                            mr.inputs.reserve(eNode.getChildren().size());
                            mr.inputs.push_back(binding.slots[0]);
                            for (uint64_t i = 1; i < eNode.getChildren().size(); ++i)
                            {
                                mr.inputs.push_back(egraph.findConst(eNode.getChildren()[i]));
                            }
                        }
                        else
                        {
                            mr.inputs = std::move(binding.slots);
                        }

                        activeMatches.push_back(std::move(mr));
                    }
                }
            }
        }

        return !activeMatches.empty();
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        for (const auto &match : activeMatches)
        {
            const Pattern &pattern = *match.pattern;
            const auto &inputs = match.inputs;

            ApplicationKey application_key{eNodeIdx, &pattern};
            if (hasCompleted(application_key))
            {
                ++application_cache_hits;
                continue;
            }

            DType outDtype;
            std::vector<uint32_t> outShape;
            std::vector<uint64_t> outStrides;
            MemSpace outMemSpace;
            {
                const EClass &matchedClass = egraph.getEClass(egraph.getENodeEClass(ENodeId{eNodeIdx}));
                outDtype = matchedClass.dtype;
                outShape = matchedClass.shape;
                outStrides = matchedClass.strides;
                outMemSpace = matchedClass.mem_space;
            }

            std::vector<TensorNode> inputNodes;
            TensorNode outputNode;
            bool nodesBuilt = false;
            auto ensureNodesBuilt = [&]() {
                if (nodesBuilt)
                    return;
                nodesBuilt = true;
                outputNode.opType = OpType::FUSED;
                outputNode.opName = pattern.opName;
                outputNode.dtype = outDtype;
                outputNode.setShape(outShape);
                outputNode.strides = outStrides;

                inputNodes.reserve(inputs.size());
                for (EClassId inId : inputs)
                {
                    const EClass &p = egraph.getEClass(egraph.findConst(inId));
                    TensorNode in;
                    in.opType = OpType::INPUT;
                    in.dtype = p.dtype;
                    in.setShape(p.shape);
                    in.strides = p.strides;
                    inputNodes.push_back(std::move(in));
                }
            };

            if (ctx.costModel && ctx.costModel->referenceObserver)
            {
                ensureNodesBuilt();
                std::vector<std::vector<uint8_t>> constantBytes;
                for (size_t in_i = 0; in_i < inputs.size(); ++in_i)
                {
                    EClassId input_id = egraph.findConst(inputs[in_i]);
                    auto it = egraph.constantStaging.find(input_id);
                    constantBytes.push_back(it == egraph.constantStaging.end() ? std::vector<uint8_t>{}
                                                                               : *it->second);
                }
                ctx.costModel->referenceObserver(pattern.opName, inputNodes, outputNode, constantBytes);
            }

            std::array<MemSpace, 8> candidateSpaces;
            size_t numCandidateSpaces = 0;
            candidateSpaces[numCandidateSpaces++] = outMemSpace;
            for (const auto &avail_ms : System::get().getAvailableMemSpaces())
            {
                if (avail_ms.type != HandleType::STORAGE && !(avail_ms == outMemSpace))
                {
                    if (numCandidateSpaces < candidateSpaces.size())
                        candidateSpaces[numCandidateSpaces++] = avail_ms;
                }
            }

            uint64_t sig_h1 = 14695981039346656037ULL;
            uint64_t sig_h2 = 1099511628211ULL;
            auto addSig = [&](const void *data, size_t size) {
                const uint8_t *p = static_cast<const uint8_t *>(data);
                size_t words = size / 8;
                for (size_t i = 0; i < words; ++i) {
                    uint64_t w;
                    std::memcpy(&w, p + i * 8, 8);
                    sig_h1 = (sig_h1 ^ w) * 1099511628211ULL;
                    sig_h2 = (sig_h2 ^ (w * 14695981039346656037ULL)) * 1099511628211ULL;
                }
                for (size_t i = words * 8; i < size; ++i) {
                    sig_h1 = (sig_h1 ^ p[i]) * 1099511628211ULL;
                    sig_h2 = (sig_h2 ^ p[i]) * 14695981039346656037ULL;
                }
            };
            {
                const CostModel *model = ctx.costModel;
                addSig(&model, sizeof(model));
                const ENode &old = egraph.getENode(ENodeId{eNodeIdx});
                addSig(pattern.pattern_hash.data(), pattern.pattern_hash.size());
                const uint32_t opType = static_cast<uint32_t>(old.getOpType());
                addSig(&opType, sizeof(opType));
                addSig(old.getShape().data(), old.getShape().size() * sizeof(uint32_t));
                addSig(old.getStrides().data(), old.getStrides().size() * sizeof(uint64_t));
                const uint32_t outD = static_cast<uint32_t>(old.getDType());
                addSig(&outD, sizeof(outD));
                addSig(outStrides.data(), outStrides.size() * sizeof(uint64_t));
                addSig(&outMemSpace, sizeof(outMemSpace));
                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    const EClass &child = egraph.getEClass(egraph.findConst(inputs[i]));
                    addSig(child.shape.data(), child.shape.size() * sizeof(uint32_t));
                    addSig(child.strides.data(), child.strides.size() * sizeof(uint64_t));
                    const uint32_t d = static_cast<uint32_t>(child.dtype);
                    addSig(&d, sizeof(d));
                    const MemSpace cms = child.mem_space;
                    addSig(&cms, sizeof(cms));
                    auto staged = egraph.constantStaging.find(egraph.findConst(inputs[i]));
                    if (staged != egraph.constantStaging.end() && !isScatterControlInput(old.getOpType(), i)) {
                        const void *c_ptr = staged->second->data();
                        addSig(&c_ptr, sizeof(c_ptr));
                    }
                }
            }

            for (size_t cs_i = 0; cs_i < numCandidateSpaces; ++cs_i)
            {
                const auto &target_ms = candidateSpaces[cs_i];
                uint64_t ms_val;
                std::memcpy(&ms_val, &target_ms, sizeof(ms_val));
                uint64_t k_h1 = (sig_h1 ^ ms_val) * 1099511628211ULL;
                uint64_t k_h2 = (sig_h2 ^ (ms_val * 14695981039346656037ULL)) * 1099511628211ULL;
                SelectionKey key{k_h1, k_h2};
                std::shared_ptr<const std::vector<FusedCandidate>> selected;
                {
                    auto hit = selectionMemo().find(key);
                    if (hit != selectionMemo().end())
                        selected = hit->second;
                }
                if (selected)
                {
                    materializeFusedCandidates(ctx, *selected, inputs, ENodeId{eNodeIdx});
                    continue;
                }

                ensureNodesBuilt();
                const auto &candidate_kernels = pattern.getCandidateKernelIds();
                std::vector<std::pair<const KernelEntry *, std::pair<std::vector<Engine>, std::vector<MemSpace>>>>
                    matched_kernels;
                matched_kernels.reserve(candidate_kernels.size());
                for (KernelId uid : candidate_kernels)
                {
                    const KernelEntry &kernel = KernelRegistry::get().getKernel(uid);
                    std::vector<Engine> mapped_engines;
                    std::vector<MemSpace> mapped_input_spaces;
                    if (kernel.matches(inputNodes, outputNode, target_ms, {}, {}, false, true, true, true,
                                       &mapped_engines, &mapped_input_spaces))
                    {
                        matched_kernels.push_back({&kernel, {mapped_engines, mapped_input_spaces}});
                    }
                }

                selected = std::make_shared<const std::vector<FusedCandidate>>(selectFusedCandidates(
                    ctx, matched_kernels, target_ms, inputs, egraph.getENode(ENodeId{eNodeIdx})));
                selectionMemo().emplace(key, selected);
                if (const char *dbg = std::getenv("TG_DEBUG_FUSION"))
                    if (pattern.opName.find(dbg) != std::string::npos)
                    {
                        std::string names;
                        for (const auto &c : *selected)
                            names += c.kernel->opName + "(" + std::to_string(c.cost) + "),";
                        std::cout << "[debug fusion] " << pattern.opName << " out=" << toString(outShape)
                                  << " in0=" << toString(inputNodes[0].getShape()) << " patternMatches="
                                  << candidate_kernels.size() << " matched=" << matched_kernels.size()
                                  << " selected=" << names << std::endl;
                    }
                materializeFusedCandidates(ctx, *selected, inputs, ENodeId{eNodeIdx});
            }
            local_completed_applications.insert(std::move(application_key));
        }
    }

    struct FusedCandidate
    {
        const KernelEntry *kernel;
        MemSpace target_mem_space;
        std::vector<Engine> mapped_engines;
        std::vector<MemSpace> mapped_input_spaces;
        std::vector<std::vector<std::vector<MemSpace>>> child_mem_paths;
        std::vector<bool> child_need_contig;
        std::vector<bool> child_need_copy;
        float cost = std::numeric_limits<float>::infinity();
    };

    // Stages 1-2: the candidates worth adding for these kernels. Depends only
    // on the matched kernels, the children's layouts/memory spaces/constant
    // data and the old enode's output, so it is memoized by that signature.
    std::vector<FusedCandidate> selectFusedCandidates(
        RuleCtx &ctx,
        const std::vector<std::pair<const KernelEntry *, std::pair<std::vector<Engine>, std::vector<MemSpace>>>>
            &matched_kernels,
        MemSpace target_mem_space, const std::vector<EClassId> &child_ids, const ENode &oldENode)
    {
        EGraph &egraph = ctx.egraph;

        const std::vector<uint32_t> &outShape = oldENode.getShape();
        DType outDtype = oldENode.getDType();

        std::vector<FusedCandidate> candidates;
        candidates.reserve(matched_kernels.size());

        // -------------------------------------------------------------------------
        // Stage 1: Resolve adaptation requirements & compute true execution cost
        // -------------------------------------------------------------------------
        for (const auto &item : matched_kernels)
        {
            const KernelEntry *kernel = item.first;
            const auto &mapped_engines = item.second.first;
            const auto &mapped_input_spaces = item.second.second;

            // COPY_TO within one memory space is already an identity in the
            // egraph. Reintroducing it as a fused operation creates a cycle of
            // equivalent copies and makes saturation grow without bound.
            if (oldENode.getOpType() == OpType::COPY_TO && mapped_input_spaces.size() == 1 &&
                mapped_input_spaces[0] == target_mem_space)
                continue;

            FusedCandidate cand;
            cand.kernel = kernel;
            cand.target_mem_space = target_mem_space;
            cand.mapped_engines = mapped_engines;
            cand.mapped_input_spaces = mapped_input_spaces;
            cand.child_mem_paths.resize(child_ids.size());
            cand.child_need_contig.resize(child_ids.size(), false);
            cand.child_need_copy.resize(child_ids.size(), false);

            bool valid_paths = true;
            std::vector<std::vector<uint32_t>> actualInShapes;
            std::vector<std::vector<uint64_t>> actualInStrides;
            std::vector<DType> actualInDTypes;
            std::vector<std::vector<uint8_t>> actualInConstants;

            for (uint64_t i = 0; i < child_ids.size(); ++i)
            {
                EClassId pid = child_ids[i];
                const EClass &parent = egraph.getEClass(egraph.findConst(pid));

                MemSpace expectedMemSpace =
                    (i < mapped_input_spaces.size()) ? mapped_input_spaces[i] : target_mem_space;
                bool foundMemSpace = (parent.mem_space == expectedMemSpace);
                bool needCopy = !foundMemSpace;
                bool needContig = false;

                uint64_t ruleIdx =
                    std::min(i, static_cast<uint64_t>(
                                    kernel->requiresContiguous.empty() ? 0 : kernel->requiresContiguous.size() - 1));
                if (ruleIdx < kernel->requiresContiguous.size())
                {
                    needContig = (kernel->requiresContiguous[ruleIdx] || needCopy) && !isContiguous(parent);
                }
                else
                {
                    needContig = needCopy && !isContiguous(parent);
                }

                cand.child_need_contig[i] = needContig;
                cand.child_need_copy[i] = needCopy;

                if (needCopy)
                {
                    TensorNode dummyNode;
                    dummyNode.opType = OpType::INPUT;
                    dummyNode.dtype = parent.dtype;
                    dummyNode.setShape(parent.shape);
                    dummyNode.strides = needContig ? calcContiguousStrides(parent.shape) : parent.strides;

                    cand.child_mem_paths[i] =
                        findMemSpacePaths(parent.mem_space, expectedMemSpace, dummyNode, mapped_engines);
                    if (cand.child_mem_paths[i].empty())
                    {
                        valid_paths = false;
                        break;
                    }
                }

                // Determine the ACTUAL strides the kernel will receive at execution
                actualInShapes.push_back(parent.shape);
                actualInDTypes.push_back(parent.dtype);
                if (needContig || (needCopy && !isContiguous(parent)))
                {
                    actualInStrides.push_back(calcContiguousStrides(parent.shape));
                }
                else
                {
                    actualInStrides.push_back(parent.strides);
                }

                EClassId canonChild = egraph.findConst(pid);
                if (egraph.constantStaging.count(canonChild))
                    actualInConstants.push_back(*egraph.constantStaging.at(canonChild));
                else
                    actualInConstants.push_back({});
            }

            if (!valid_paths)
                continue;

            std::vector<uint64_t> actualOutStrides =
                kernel->is_view ? oldENode.getStrides() : calcContiguousStrides(outShape);

            if (ctx.costModel)
            {
                cand.cost =
                    ctx.costModel->estimateCost(kernel->uid, outShape, actualOutStrides, outDtype, actualInShapes,
                                                actualInStrides, actualInDTypes, actualInConstants,
                                                /*exactRecordOnly=*/true);
            }

            candidates.push_back(std::move(cand));
        }

        // -------------------------------------------------------------------------
        // Stage 2: Filter equivalent candidates using exact benchmark costs
        // -------------------------------------------------------------------------
        std::vector<bool> keep(candidates.size(), true);
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            if (candidates[i].cost == std::numeric_limits<float>::infinity())
                continue; // Keep unbenchmarked candidates

            for (size_t j = 0; j < candidates.size(); ++j)
            {
                if (i == j || candidates[j].cost == std::numeric_limits<float>::infinity())
                    continue;

                const auto &c1 = candidates[i];
                const auto &c2 = candidates[j];

                if (c1.mapped_engines != c2.mapped_engines)
                    continue;
                if (c1.mapped_input_spaces != c2.mapped_input_spaces)
                    continue;
                if (c1.kernel->requiresContiguous != c2.kernel->requiresContiguous)
                    continue;
                if (c1.kernel->safe_inplace_idxs != c2.kernel->safe_inplace_idxs)
                    continue;
                if (c1.kernel->is_view != c2.kernel->is_view)
                    continue;

                if (c2.cost < c1.cost - 1e-9f)
                {
                    keep[i] = false;
                    break;
                }
                if (std::abs(c1.cost - c2.cost) <= 1e-9f && c2.kernel->uid < c1.kernel->uid)
                {
                    keep[i] = false;
                    break;
                }
            }
        }

        std::vector<FusedCandidate> kept;
        for (size_t k = 0; k < candidates.size(); ++k)
            if (keep[k])
                kept.push_back(std::move(candidates[k]));
        return kept;
    }

    // Stage 3: materialize the selected candidates into the e-graph.
    void materializeFusedCandidates(RuleCtx &ctx, const std::vector<FusedCandidate> &candidates,
                                    const std::vector<EClassId> &child_ids, ENodeId eNodeIdx)
    {
        EGraph &egraph = ctx.egraph;
        std::vector<uint32_t> oldShape;
        std::vector<uint64_t> oldStrides;
        DType oldDType;
        std::string oldHash;
        std::string oldOrigin;
        {
            const ENode &old = egraph.getENode(eNodeIdx);
            oldShape = old.getShape();
            oldStrides = old.getStrides();
            oldDType = old.getDType();
            oldHash = old.getContentHash();
            oldOrigin = old.getDebugOrigin();
        }
        const std::vector<uint64_t> contigStrides = calcContiguousStrides(oldShape);
        EClassId e_class_id = egraph.getENodeEClass(eNodeIdx);
        // -------------------------------------------------------------------------
        // Stage 3: Materialize only the winning candidates into the EGraph
        // -------------------------------------------------------------------------
        for (size_t k = 0; k < candidates.size(); ++k)
        {
            const auto &cand = candidates[k];
            std::vector<EClassId> adapted_children;
            adapted_children.reserve(child_ids.size());

            for (uint64_t i = 0; i < child_ids.size(); ++i)
            {
                EClassId pid = child_ids[i];
                bool needCopy = cand.child_need_copy[i];
                bool needContig = cand.child_need_contig[i];

                if (!needCopy && !needContig)
                {
                    adapted_children.push_back(pid);
                    continue;
                }

                EClassId currentPid = pid;
                std::vector<uint32_t> curShape;
                std::vector<uint64_t> curStrides;
                DType curDType;
                MemSpace curMem;
                {
                    const EClass &cls = egraph.getEClass(egraph.findConst(pid));
                    curShape = cls.shape;
                    curStrides = cls.strides;
                    curDType = cls.dtype;
                    curMem = cls.mem_space;
                }

                if (needContig)
                {
                    currentPid = addOpToEGraph(egraph, OpType::CONTIGUOUS, {currentPid}, curShape,
                                               calcContiguousStrides(curShape), curDType, curMem);
                    const EClass &cls = egraph.getEClass(egraph.findConst(currentPid));
                    curShape = cls.shape;
                    curStrides = cls.strides;
                    curDType = cls.dtype;
                    curMem = cls.mem_space;
                }

                if (needCopy)
                {
                    EClassId finalTargetClass = EClassId();
                    for (const auto &path : cand.child_mem_paths[i])
                    {
                        EClassId pathPid = currentPid;
                        auto pathShape = curShape;
                        auto pathStrides = curStrides;
                        auto pathDType = curDType;
                        auto pathMem = curMem;
                        for (uint64_t p_idx = 1; p_idx < path.size(); ++p_idx)
                        {
                            MemSpace next_ms = path[p_idx];
                            if (pathStrides != calcContiguousStrides(pathShape))
                            {
                                pathPid = addOpToEGraph(egraph, OpType::CONTIGUOUS, {pathPid}, pathShape,
                                                        calcContiguousStrides(pathShape), pathDType, pathMem);
                                const EClass &cls = egraph.getEClass(egraph.findConst(pathPid));
                                pathStrides = cls.strides;
                                pathMem = cls.mem_space;
                            }
                            pathPid = addOpToEGraph(egraph, OpType::COPY_TO, {pathPid}, pathShape,
                                                    pathStrides, pathDType, next_ms,
                                                    (p_idx == path.size() - 1) ? finalTargetClass : EClassId());
                            if (p_idx == path.size() - 1)
                            {
                                finalTargetClass = pathPid;
                            }
                            const EClass &cls = egraph.getEClass(egraph.findConst(pathPid));
                            pathStrides = cls.strides;
                            pathMem = cls.mem_space;
                        }
                    }
                    currentPid = finalTargetClass;
                }

                adapted_children.push_back(currentPid);
            }

            const auto &cand_strides = cand.kernel->is_view ? oldStrides : contigStrides;

            const EClassId origCanon = egraph.findConst(e_class_id);
            MemSpace originalMemSpace;
            bool stridesMatch = false;
            bool duplicate = false;
            bool origIsContig = false;
            {
                const EClass &cls = egraph.getEClass(origCanon);
                originalMemSpace = cls.mem_space;
                origIsContig = isContiguous(cls);
                if (cand.target_mem_space == originalMemSpace && cand_strides == cls.strides)
                {
                    stridesMatch = true;
                    const size_t num_enode_children = adapted_children.size();
                    for (ENodeId existing : cls.enodes)
                    {
                        const ENode &other = egraph.getENode(existing);
                        if (other.getKernelId() != cand.kernel->uid ||
                            other.getMemSpace() != cand.target_mem_space ||
                            other.getStrides() != cand_strides ||
                            other.getChildren().size() != num_enode_children)
                            continue;
                        bool same_children = true;
                        for (size_t c = 0; c < num_enode_children && same_children; ++c)
                            same_children = (egraph.findConst(other.getChildren()[c]) == adapted_children[c]);
                        if (same_children)
                        {
                            duplicate = true;
                            break;
                        }
                    }
                }
            }

            if (cand.target_mem_space == originalMemSpace)
            {
                if (!stridesMatch || duplicate)
                    continue;

                ENode enode(cand.kernel->uid, cand.kernel->opType, cand.kernel->opName, std::move(adapted_children),
                            oldShape, cand_strides, oldDType, cand.target_mem_space, cand.mapped_engines,
                            oldHash, 0, oldOrigin);
                egraph.addENode(e_class_id, std::move(enode));
            }
            else
            {
                ENode enode(cand.kernel->uid, cand.kernel->opType, cand.kernel->opName, std::move(adapted_children),
                            oldShape, cand_strides, oldDType, cand.target_mem_space, cand.mapped_engines,
                            oldHash, 0, oldOrigin);
                EClassId newEClass =
                    egraph.addEClass(enode.getShape(), enode.getStrides(), enode.getDType(), cand.target_mem_space);
                newEClass = egraph.addENode(newEClass, std::move(enode));

                EClassId srcForCopy = newEClass;
                std::vector<uint32_t> copyShape;
                std::vector<uint64_t> copyStrides;
                DType copyDType;
                MemSpace copyMem;
                {
                    const EClass &cCls = egraph.getEClass(egraph.findConst(srcForCopy));
                    copyShape = cCls.shape;
                    copyStrides = cCls.strides;
                    copyDType = cCls.dtype;
                    copyMem = cCls.mem_space;
                }

                if (copyStrides != calcContiguousStrides(copyShape))
                {
                    srcForCopy = addOpToEGraph(egraph, OpType::CONTIGUOUS, {srcForCopy}, copyShape,
                                               calcContiguousStrides(copyShape), copyDType, copyMem);
                    const EClass cCls = egraph.getEClass(egraph.findConst(srcForCopy));
                    copyStrides = cCls.strides;
                    copyMem = cCls.mem_space;
                }

                if (origIsContig)
                {
                    addOpToEGraph(egraph, OpType::COPY_TO, {srcForCopy}, copyShape, copyStrides,
                                  copyDType, originalMemSpace, e_class_id);
                }
                else
                {
                    addOpToEGraph(egraph, OpType::COPY_TO, {srcForCopy}, copyShape, copyStrides,
                                  copyDType, originalMemSpace);
                }
            }
        }
    }

    static bool matchPatternClass(EClassId eClassIdx, const EGraph &egraph, LogicalId patternId, const Pattern &pattern,
                                  Binding &binding, const std::unordered_set<EClassId> &protectedEClasses,
                                  bool ignoreConstantData = false, uint32_t enode_watermark = 0,
                                  uint32_t class_watermark = 0, const std::vector<uint8_t> *base_staged = nullptr,
                                  bool *has_new_node = nullptr)
    {
        eClassIdx = egraph.findConst(eClassIdx);

        const int32_t varIdx = patternId.value < pattern.varIndex.size() ? pattern.varIndex[patternId.value] : -1;
        if (varIdx >= 0)
        {
            const EClass &eclass = egraph.getEClass(eClassIdx);
            if (static_cast<size_t>(varIdx) < pattern.dtypes.size() && eclass.dtype != pattern.dtypes[varIdx])
                return false;
            EClassId &slot = binding.slots[varIdx];
            if (slot != EClassId{})
                return slot == eClassIdx;
            slot = eClassIdx;
            binding.trail.push_back(static_cast<uint32_t>(varIdx));
            if (has_new_node)
            {
                if (class_watermark > 0 && eClassIdx.value >= class_watermark &&
                    egraph.getEClass(eClassIdx).base_eclass_id == BaseEClassId{})
                    *has_new_node = true;
                if (base_staged && (eClassIdx.value >= base_staged->size() || !(*base_staged)[eClassIdx.value]))
                {
                    if (egraph.constantStaging.count(eClassIdx))
                        *has_new_node = true;
                }
            }
            return true;
        }

        if (patternId != pattern.rootId)
        {
            if (isEClassProtected(eClassIdx, protectedEClasses, egraph))
                return false;
        }

        const EClass &eclass = egraph.getEClass(eClassIdx);
        for (ENodeId enodeId : eclass.enodes)
        {
            const size_t mark = binding.trail.size();
            bool child_has_new = has_new_node ? *has_new_node : false;
            if (enode_watermark > 0 && enodeId.value >= enode_watermark)
                child_has_new = true;
            if (matchPatternNode(enodeId, egraph, patternId, pattern, binding, protectedEClasses, ignoreConstantData,
                                 enode_watermark, class_watermark, base_staged, &child_has_new))
            {
                if (has_new_node && child_has_new)
                    *has_new_node = true;
                return true;
            }
            binding.undo(mark);
        }
        return false;
    }

    static bool matchPatternNode(ENodeId eNodeId, const EGraph &egraph, LogicalId patternId, const Pattern &pattern,
                                 Binding &binding,
                                 const std::unordered_set<EClassId> &protectedEClasses, bool ignoreConstantData = false,
                                 uint32_t enode_watermark = 0, uint32_t class_watermark = 0,
                                 const std::vector<uint8_t> *base_staged = nullptr, bool *has_new_node = nullptr)
    {
        if (enode_watermark > 0 && eNodeId.value >= enode_watermark && has_new_node)
            *has_new_node = true;
        const ENode &eNode = egraph.getENode(eNodeId);
        const auto &pNode = pattern.graph.getNode(patternId);

        if (eNode.getOpType() != pNode.opType)
            return false;
        if (eNode.getOpType() == OpType::FUSED && eNode.getOpName() != pNode.opName)
            return false;

        if (eNode.getOpType() == OpType::INPUT && !pNode.contentHash.empty())
        {
            if (!ignoreConstantData)
            {
                EClassId eNodeEClass = egraph.getENodeEClass(eNodeId);
                eNodeEClass = egraph.findConst(eNodeEClass);

                auto egraphIt = egraph.constantStaging.find(eNodeEClass);
                if (egraphIt == egraph.constantStaging.end())
                    return false;

                auto patternIt = pattern.graph.constantStaging.find(patternId);
                if (patternIt == pattern.graph.constantStaging.end())
                    return false;

                const auto &egraphData = *egraphIt->second;
                const auto &patternData = *patternIt->second;
                if (egraphData.size() != patternData.size())
                    return false;
                if (std::memcmp(egraphData.data(), patternData.data(), egraphData.size()) != 0)
                    return false;
            }
        }

        if (eNode.getOpType() == OpType::CONCAT && eNode.getChildren().size() != pNode.child_ids.size())
        {
            if (pNode.child_ids.size() < 2)
                return false;
            if (eNode.getChildren().size() < 2)
                return false;

            // Match axis (first element)
            if (!matchPatternClass(eNode.getChildren()[0], egraph, pNode.child_ids[0], pattern, binding,
                                   protectedEClasses, false, enode_watermark, class_watermark, base_staged, has_new_node))
                return false;

            // The rest are tensors. The pattern has one tensor at index 1.
            bool firstTensor = true;
            for (uint64_t i = 1; i < eNode.getChildren().size(); ++i)
            {
                if (firstTensor)
                {
                    if (!matchPatternClass(eNode.getChildren()[i], egraph, pNode.child_ids[1], pattern, binding,
                                           protectedEClasses, true, enode_watermark, class_watermark, base_staged, has_new_node))
                        return false;
                    firstTensor = false;
                }
                else
                {
                    EClassId canonChild = egraph.findConst(eNode.getChildren()[i]);
                    const EClass &childCls = egraph.getEClass(canonChild);
                    if (childCls.dtype != pattern.dtypes[1])
                        return false;
                }
            }
            return true;
        }

        if (eNode.getChildren().size() != pNode.child_ids.size())
            return false;

        for (uint64_t i = 0; i < eNode.getChildren().size(); ++i)
        {
            bool childIgnoreConst = isConstant(eNode.getOpType(), i, eNode.getChildren().size());
            if (!matchPatternClass(eNode.getChildren()[i], egraph, pNode.child_ids[i], pattern, binding,
                                   protectedEClasses, childIgnoreConst, enode_watermark, class_watermark, base_staged, has_new_node))
            {
                return false;
            }
        }
        return true;
    }
};

// Pure functional replacement: every cell comes from the explicit base or
// the update. Unlike a cached scatter this result is valid on its first use.
inline EClassId replaceRegion(EGraph &egraph, EClassId base, EClassId update, const Region &box)
{
    const EClass initial = egraph.getEClass(egraph.findConst(base));
    if (box.region.size() != initial.shape.size())
        Error::throw_err("[replaceRegion] rank mismatch");
    std::function<EClassId(EClassId, size_t)> build = [&](EClassId source, size_t axis) -> EClassId {
        if (axis == initial.shape.size())
            return update;
        const EClass cls = egraph.getEClass(egraph.findConst(source));
        const Dim interval = box.region[axis];
        if (interval.start >= interval.stop || interval.stop > cls.shape[axis])
            Error::throw_err("[replaceRegion] invalid update bounds");
        auto slice = [&](uint32_t start, uint32_t stop) {
            std::vector<int32_t> starts(cls.shape.size(), 0), ends(cls.shape.begin(), cls.shape.end());
            std::vector<int32_t> steps(cls.shape.size(), 1);
            starts[axis] = start;
            ends[axis] = stop;
            std::vector<uint32_t> shape = cls.shape;
            shape[axis] = stop - start;
            const EClassId sliced = addOpToEGraph(egraph, OpType::SLICE,
                {source, egraph.addIntConst(starts), egraph.addIntConst(ends), egraph.addIntConst(steps)},
                shape, cls.strides, cls.dtype, cls.mem_space);
            return addOpToEGraph(egraph, OpType::CONTIGUOUS, {sliced}, shape,
                calcContiguousStrides(shape), cls.dtype, cls.mem_space);
        };
        std::vector<EClassId> pieces{egraph.addIntConst({static_cast<int32_t>(axis)})};
        if (interval.start > 0)
            pieces.push_back(slice(0, interval.start));
        pieces.push_back(build(slice(interval.start, interval.stop), axis + 1));
        if (interval.stop < cls.shape[axis])
            pieces.push_back(slice(interval.stop, cls.shape[axis]));
        if (pieces.size() == 2)
            return pieces[1];
        return addOpToEGraph(egraph, OpType::CONCAT, pieces, cls.shape,
            calcContiguousStrides(cls.shape), cls.dtype, cls.mem_space);
    };
    return build(base, 0);
}

struct InfinityDomination : public Rule
{
    std::unique_ptr<Rule> clone() const override
    {
        return std::make_unique<InfinityDomination>(*this);
    }

    // A causal or sliding mask has one unmasked run per row; splitting into
    // one piece per row would give every row its own sliced computation. Cover
    // the unmasked cells with at most this many boxes instead (consecutive
    // regions share a bounding box). Covered masked cells are still computed
    // as value + mask, so the result is unchanged.
    static constexpr size_t kMaxRegions = 8;

    // Constant data/layout -> (no masked element, unmasked boxes).
    std::unordered_map<std::string, std::pair<bool, std::vector<Region>>> unmasked_memo;

    static std::vector<Region> coverWithFewRegions(std::vector<Region> regions, size_t max_regions)
    {
        if (regions.size() <= max_regions)
            return regions;
        std::sort(regions.begin(), regions.end(), [](const Region &a, const Region &b) {
            for (size_t d = 0; d < a.region.size() && d < b.region.size(); ++d)
            {
                if (a.region[d].start != b.region[d].start)
                    return a.region[d].start < b.region[d].start;
            }
            return false;
        });
        std::vector<Region> covered;
        const size_t group_size = (regions.size() + max_regions - 1) / max_regions;
        for (size_t first = 0; first < regions.size(); first += group_size)
        {
            Region box = regions[first];
            for (size_t k = first + 1; k < std::min(regions.size(), first + group_size); ++k)
            {
                if (regions[k].region.size() != box.region.size())
                    Error::throw_err("[InfinityDomination] regions of different rank");
                for (size_t d = 0; d < box.region.size(); ++d)
                {
                    box.region[d].start = std::min(box.region[d].start, regions[k].region[d].start);
                    box.region[d].stop = std::max(box.region[d].stop, regions[k].region[d].stop);
                }
            }
            covered.push_back(box);
        }
        return mergeRegions(covered);
    }

    std::vector<uint8_t> visited_enodes;

    void markVisited(uint32_t eNodeIdx)
    {
        if (eNodeIdx >= visited_enodes.size())
            visited_enodes.resize(eNodeIdx + 1, 0);
        visited_enodes[eNodeIdx] = 1;
    }

    std::string name() const override
    {
        return "InfinityDomination";
    }

    bool isFillWithNegInfEClass(EClassId cid, const EGraph &egraph, RuleCtx *ctx = nullptr, int depth = 8) const
    {
        if (depth <= 0)
            return false;
        cid = egraph.findConst(cid);
        const EClass &cls = egraph.getEClass(cid);
        for (ENodeId enode_id : cls.enodes)
        {
            const ENode &enode = egraph.getENode(enode_id);
            if (enode.getOpType() == OpType::FILL && enode.getChildren().size() >= 2)
            {
                EClassId val_cid = egraph.findConst(enode.getChildren()[0]);
                const EClass &val_cls = egraph.getEClass(val_cid);
                if (val_cls.dtype == DType::FLOAT32)
                {
                    if (egraph.constantStaging.count(val_cid))
                    {
                        const auto &data = *egraph.constantStaging.at(val_cid);
                        if (data.size() >= sizeof(float))
                        {
                            float val;
                            std::memcpy(&val, data.data(), sizeof(float));
                            if (val == TGConstants::NEG_INF)
                                return true;
                        }
                    }
                    if (ctx && ctx->repo && ctx->repo->isValid())
                    {
                        LogicalId lid = val_cls.logical_id;
                        if (lid != LogicalId() && ctx->repo->has(lid))
                        {
                            auto data = ctx->repo->read(lid);
                            if (data.size() >= sizeof(float))
                            {
                                float val;
                                std::memcpy(&val, data.data(), sizeof(float));
                                if (val == TGConstants::NEG_INF)
                                    return true;
                            }
                        }
                    }
                }
            }
            const bool isView = (enode.getOpType() == OpType::REPEAT || enode.getOpType() == OpType::RESHAPE ||
                                 enode.getOpType() == OpType::PERMUTE || enode.getOpType() == OpType::SLICE ||
                                 enode.getOpType() == OpType::CONTIGUOUS || enode.getOpType() == OpType::COPY_TO);
            if (isView && !enode.getChildren().empty())
            {
                if (isFillWithNegInfEClass(enode.getChildren()[0], egraph, ctx, depth - 1))
                    return true;
            }
        }
        return false;
    }

    bool mapTriuBoundForwardENode(const ENode &enode, const EGraph &egraph,
                                  const TriuBound &in_bound, TriuBound &out_bound) const
    {
        switch (enode.getOpType())
        {
        case OpType::CONTIGUOUS:
        case OpType::COPY_TO:
            out_bound = in_bound;
            return true;
        case OpType::REPEAT: {
            if (enode.getChildren().size() < 3)
                return false;
            EClassId r_cid = egraph.findConst(enode.getChildren()[1]);
            EClassId a_cid = egraph.findConst(enode.getChildren()[2]);
            if (!egraph.constantStaging.count(r_cid) || !egraph.constantStaging.count(a_cid))
                return false;
            auto r_vec = egraph.getConstantInt32(r_cid);
            auto a_vec = egraph.getConstantInt32(a_cid);
            if (r_vec.empty() || a_vec.empty())
                return false;
            int32_t repeats = r_vec[0];
            int32_t axis = a_vec[0];
            EClassId c0 = egraph.findConst(enode.getChildren()[0]);
            size_t in_rank = egraph.getEClass(c0).shape.size();
            if (axis < 0)
                axis += static_cast<int32_t>(in_rank);
            if (repeats == 1 || (static_cast<uint32_t>(axis) != in_bound.row_axis &&
                                 static_cast<uint32_t>(axis) != in_bound.col_axis))
            {
                out_bound = in_bound;
                return true;
            }
            return false;
        }
        case OpType::PERMUTE: {
            if (enode.getChildren().size() < 2)
                return false;
            EClassId d_cid = egraph.findConst(enode.getChildren()[1]);
            if (!egraph.constantStaging.count(d_cid))
                return false;
            auto dims = egraph.getConstantInt32(d_cid);
            uint32_t new_row = UINT32_MAX, new_col = UINT32_MAX;
            for (uint32_t i = 0; i < dims.size(); ++i)
            {
                if (static_cast<uint32_t>(dims[i]) == in_bound.row_axis)
                    new_row = i;
                if (static_cast<uint32_t>(dims[i]) == in_bound.col_axis)
                    new_col = i;
            }
            if (new_row != UINT32_MAX && new_col != UINT32_MAX)
            {
                out_bound = TriuBound{new_row, new_col, in_bound.diag_k};
                return true;
            }
            return false;
        }
        case OpType::RESHAPE: {
            EClassId c0 = egraph.findConst(enode.getChildren()[0]);
            const auto &in_shape = egraph.getEClass(c0).shape;
            const auto &out_shape = enode.getShape();
            if (in_shape.size() >= 2 && out_shape.size() >= 2)
            {
                size_t in_rank = in_shape.size();
                size_t out_rank = out_shape.size();
                if (in_bound.row_axis == in_rank - 2 && in_bound.col_axis == in_rank - 1 &&
                    in_shape[in_rank - 2] == out_shape[out_rank - 2] &&
                    in_shape[in_rank - 1] == out_shape[out_rank - 1])
                {
                    out_bound = TriuBound{static_cast<uint32_t>(out_rank - 2),
                                          static_cast<uint32_t>(out_rank - 1), in_bound.diag_k};
                    return true;
                }
            }
            return false;
        }
        case OpType::SLICE: {
            if (enode.getChildren().size() < 4)
                return false;
            EClassId st_cid = egraph.findConst(enode.getChildren()[1]);
            EClassId sp_cid = egraph.findConst(enode.getChildren()[3]);
            if (!egraph.constantStaging.count(st_cid) || !egraph.constantStaging.count(sp_cid))
                return false;
            auto starts = egraph.getConstantInt32(st_cid);
            auto steps = egraph.getConstantInt32(sp_cid);
            if (in_bound.row_axis < steps.size() && in_bound.col_axis < steps.size() &&
                steps[in_bound.row_axis] == 1 && steps[in_bound.col_axis] == 1 &&
                in_bound.row_axis < starts.size() && in_bound.col_axis < starts.size())
            {
                int32_t new_k = in_bound.diag_k + starts[in_bound.row_axis] - starts[in_bound.col_axis];
                out_bound = TriuBound{in_bound.row_axis, in_bound.col_axis, new_k};
                return true;
            }
            return false;
        }
        default:
            return false;
        }
    }

    bool isTriuMaskEClass(EClassId cid, const EGraph &egraph, TriuBound &out_bound,
                          std::unordered_set<EClassId> &visited, RuleCtx *ctx = nullptr) const
    {
        cid = egraph.findConst(cid);
        if (visited.count(cid))
            return false;
        visited.insert(cid);

        const EClass &cls = egraph.getEClass(cid);
        for (ENodeId enode_id : cls.enodes)
        {
            const ENode &enode = egraph.getENode(enode_id);
            if (enode.getOpType() == OpType::TRIU && enode.getChildren().size() == 2 && enode.getShape().size() >= 2)
            {
                EClassId k_cid = egraph.findConst(enode.getChildren()[1]);
                std::optional<int32_t> diag_k;
                if (egraph.constantStaging.count(k_cid))
                {
                    const auto &k_data = *egraph.constantStaging.at(k_cid);
                    if (k_data.size() >= sizeof(int32_t))
                    {
                        int32_t k_val;
                        std::memcpy(&k_val, k_data.data(), sizeof(int32_t));
                        diag_k = k_val;
                    }
                }
                else if (ctx && ctx->repo && ctx->repo->isValid())
                {
                    const EClass &k_cls = egraph.getEClass(k_cid);
                    if (k_cls.logical_id != LogicalId() && ctx->repo->has(k_cls.logical_id))
                    {
                        auto bytes = ctx->repo->read(k_cls.logical_id);
                        if (bytes.size() >= sizeof(int32_t))
                        {
                            int32_t k_val;
                            std::memcpy(&k_val, bytes.data(), sizeof(int32_t));
                            diag_k = k_val;
                        }
                    }
                }
                if (diag_k)
                {
                    EClassId fill_cid = egraph.findConst(enode.getChildren()[0]);
                    if (isFillWithNegInfEClass(fill_cid, egraph, ctx))
                    {
                        out_bound.row_axis = static_cast<uint32_t>(enode.getShape().size() - 2);
                        out_bound.col_axis = static_cast<uint32_t>(enode.getShape().size() - 1);
                        out_bound.diag_k = *diag_k;
                        return true;
                    }
                }
            }
            const bool isView = (enode.getOpType() == OpType::REPEAT || enode.getOpType() == OpType::RESHAPE ||
                                 enode.getOpType() == OpType::PERMUTE || enode.getOpType() == OpType::SLICE ||
                                 enode.getOpType() == OpType::CONTIGUOUS || enode.getOpType() == OpType::COPY_TO);
            if (isView && !enode.getChildren().empty())
            {
                TriuBound child_bound;
                if (isTriuMaskEClass(enode.getChildren()[0], egraph, child_bound, visited, ctx))
                {
                    if (mapTriuBoundForwardENode(enode, egraph, child_bound, out_bound))
                        return true;
                }
            }
        }
        return false;
    }

    bool isTriuMaskEClass(EClassId cid, const EGraph &egraph, TriuBound &out_bound, RuleCtx *ctx = nullptr) const
    {
        std::unordered_set<EClassId> visited;
        return isTriuMaskEClass(cid, egraph, out_bound, visited, ctx);
    }

    void clearVisited() override
    {
        visited_enodes.clear();
    }

    bool canMatchOp(OpType op) const override
    {
        return op == OpType::ADD;
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (eNodeIdx < ctx.base_enode_watermark && ctx.sweep_filter &&
            eNodeIdx < ctx.sweep_filter->size() && (*ctx.sweep_filter)[eNodeIdx] == 0)
            return false;
        if (eNodeIdx < visited_enodes.size() && visited_enodes[eNodeIdx])
            return false;

        const ENode &enode = ctx.egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::ADD || enode.getChildren().size() != 2)
            return false;

        const EClassId cid = ctx.egraph.getENodeEClass(ENodeId{eNodeIdx});
        const EClass &cls = ctx.egraph.getEClass(ctx.egraph.findConst(cid));
        if (cls.logical_id == LogicalId{})
        {
            markVisited(eNodeIdx);
            return false;
        }
        if (ctx.regions && cls.base_eclass_id == BaseEClassId{})
        {
            markVisited(eNodeIdx);
            return false;
        }

        TriuBound dummy;
        return isConstantFloat(enode.getChildren()[0], ctx) || isConstantFloat(enode.getChildren()[1], ctx) ||
               isTriuMaskEClass(enode.getChildren()[0], ctx.egraph, dummy, &ctx) ||
               isTriuMaskEClass(enode.getChildren()[1], ctx.egraph, dummy, &ctx);
    }

    bool isConstantFloat(EClassId e_class_id, RuleCtx &ctx) const
    {
        e_class_id = ctx.egraph.findConst(e_class_id);
        const EClass &cls = ctx.egraph.getEClass(e_class_id);
        if (cls.dtype != DType::FLOAT32)
            return false;
        if (ctx.egraph.constantStaging.find(e_class_id) != ctx.egraph.constantStaging.end())
            return true;

        if (ctx.repo && ctx.repo->isValid())
        {
            LogicalId logical_id = cls.logical_id;

            if (logical_id != LogicalId() && ctx.repo->has(logical_id))
            {
                auto data = ctx.repo->read(logical_id);
                ctx.egraph.constantStaging[e_class_id] = std::make_shared<std::vector<uint8_t>>(std::move(data));
                return true;
            }
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        markVisited(eNodeIdx);

        // COPY instead of reference to prevent UAF
        const ENode addNode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        TriuBound triu_bound_0, triu_bound_1;
        bool is_triu_1 = isTriuMaskEClass(addNode.getChildren()[1], egraph, triu_bound_1, &ctx);
        bool is_triu_0 = isTriuMaskEClass(addNode.getChildren()[0], egraph, triu_bound_0, &ctx);
        bool is_const_1 = isConstantFloat(addNode.getChildren()[1], ctx);
        bool is_const_0 = isConstantFloat(addNode.getChildren()[0], ctx);

        uint32_t constIdx;
        bool is_triu = false;
        TriuBound triu_bound;
        if (is_triu_1 || is_const_1)
        {
            constIdx = 1;
            if (is_triu_1) { is_triu = true; triu_bound = triu_bound_1; }
        }
        else if (is_triu_0 || is_const_0)
        {
            constIdx = 0;
            if (is_triu_0) { is_triu = true; triu_bound = triu_bound_0; }
        }
        else
            return;

        uint32_t varIdx = 1 - constIdx;

        EClassId constClass = egraph.findConst(addNode.getChildren()[constIdx]);
        EClassId varClass = egraph.findConst(addNode.getChildren()[varIdx]);

        const EClass variable = egraph.getEClass(varClass);
        const EClass cClass = egraph.getEClass(constClass);
        if (cClass.shape != variable.shape)
            return;

        std::vector<Region> nonInfRegions;
        if (is_triu)
        {
            const uint32_t R = cClass.shape[triu_bound.row_axis];
            const uint32_t C = cClass.shape[triu_bound.col_axis];
            if (R > 0 && C > 0)
            {
                const size_t num_regions = std::min<size_t>(kMaxRegions, R);
                const size_t group_size = (R + num_regions - 1) / num_regions;
                std::vector<Region> boxes;
                for (size_t first = 0; first < R; first += group_size)
                {
                    const size_t last = std::min<size_t>(R, first + group_size);
                    const int64_t max_col = static_cast<int64_t>(last - 1) + triu_bound.diag_k;
                    if (max_col <= 0)
                        continue;
                    const uint32_t col_stop = std::min<uint32_t>(C, static_cast<uint32_t>(max_col));
                    Region reg;
                    for (size_t d = 0; d < cClass.shape.size(); ++d)
                    {
                        if (d == triu_bound.row_axis)
                            reg.region.push_back({static_cast<uint32_t>(first), static_cast<uint32_t>(last)});
                        else if (d == triu_bound.col_axis)
                            reg.region.push_back({0, col_stop});
                        else
                            reg.region.push_back({0, cClass.shape[d]});
                    }
                    boxes.push_back(std::move(reg));
                }
                nonInfRegions = mergeRegions(boxes);
            }
        }
        else
        {
            const auto &constData = *egraph.constantStaging.at(constClass);
            const float *data = reinterpret_cast<const float *>(constData.data());

            uint64_t numElements = countElements(cClass.shape);
            if (constData.size() != numElements * sizeof(float))
                Error::throw_err("[InfinityDomination] constant data size mismatch");

            // Unmasked boxes of this constant (memoized: one mask feeds every
            // layer). Runs along the last axis, merged, equal to merging the
            // unmasked cells one by one.
            std::string memoKey(reinterpret_cast<const char *>(&data), sizeof(data));
            memoKey.append(reinterpret_cast<const char *>(cClass.shape.data()), cClass.shape.size() * sizeof(uint32_t));
            memoKey.append(reinterpret_cast<const char *>(cClass.strides.data()), cClass.strides.size() * sizeof(uint64_t));
            auto memo = unmasked_memo.find(memoKey);
            if (memo == unmasked_memo.end())
            {
                std::vector<Region> runs;
                bool anyMasked = false;
                const uint32_t cols = cClass.shape.empty() ? 1 : cClass.shape.back();
                const uint64_t rows = cols == 0 ? 0 : numElements / cols;
                for (uint64_t row = 0; row < rows; ++row)
                {
                    const std::vector<uint32_t> coords = coordsFromFlatIndex(row * cols, cClass.shape);
                    uint32_t c = 0;
                    while (c < cols)
                    {
                        while (c < cols && data[getStridedIndex(row * cols + c, cClass.shape, cClass.strides)] == TGConstants::NEG_INF)
                        {
                            anyMasked = true;
                            ++c;
                        }
                        const uint32_t start = c;
                        while (c < cols && data[getStridedIndex(row * cols + c, cClass.shape, cClass.strides)] != TGConstants::NEG_INF)
                            ++c;
                        if (c > start)
                        {
                            Region r;
                            for (size_t d = 0; d + 1 < coords.size(); ++d)
                                r.region.push_back({coords[d], coords[d] + 1});
                            r.region.push_back({start, c});
                            runs.push_back(std::move(r));
                        }
                    }
                }
                std::vector<Region> boxes;
                if (anyMasked)
                    boxes = coverWithFewRegions(mergeRegions(runs), kMaxRegions);
                memo = unmasked_memo.emplace(memoKey, std::make_pair(!anyMasked, std::move(boxes))).first;
            }
            const bool noneInf = memo->second.first;
            if (noneInf)
            {
                return;
            }
            nonInfRegions = memo->second.second;
        }

        const EClass vClass = egraph.getEClass(varClass);
        const EClass outClass = egraph.getEClass(e_class_id);

        std::vector<uint64_t> contigStrides = calcContiguousStrides(outClass.shape);
        if (outClass.strides != contigStrides)
            return;

        if (nonInfRegions.empty())
        {
            EClassId currentTarget = constClass;
            if (cClass.strides != contigStrides)
            {
                currentTarget = addOpToEGraph(egraph, OpType::CONTIGUOUS, {currentTarget}, outClass.shape,
                                              contigStrides, outClass.dtype, outClass.mem_space);
            }
            if (cClass.mem_space != outClass.mem_space)
            {
                currentTarget = addOpToEGraph(egraph, OpType::COPY_TO, {currentTarget}, outClass.shape, contigStrides,
                                              outClass.dtype, outClass.mem_space);
            }
            egraph.merge(e_class_id, currentTarget);
            return;
        }

        // A cached update writes one box covering this bucket's entire live
        // domain. The full computation initializes all other cells first.
        bool cached_update = false;
        if (ctx.regions && outClass.logical_id != LogicalId{} && outClass.base_eclass_id != BaseEClassId{})
        {
            const auto &live = ctx.regions->liveOf(outClass.logical_id);
            if (!live.empty())
            {
                const Region box = bucket_regions::boundingBox(live);
                if (scatterFormValid(ctx, outClass, box))
                {
                    nonInfRegions = {box};
                    cached_update = true;
                }
            }
        }
        if (ctx.regions && !cached_update)
            return;

        if (nonInfRegions.size() == 1)
        {
            bool strictlySmaller = false;
            const Region &reg = nonInfRegions[0];
            for (uint64_t d = 0; d < cClass.shape.size(); ++d)
            {
                if (reg.region[d].start > 0 || reg.region[d].stop < cClass.shape[d])
                {
                    strictlySmaller = true;
                    break;
                }
            }
            if (!strictlySmaller)
                return;
        }

        EClassId currentTarget = constClass;
        if (cClass.strides != contigStrides)
        {
            currentTarget = addOpToEGraph(egraph, OpType::CONTIGUOUS, {currentTarget}, outClass.shape, contigStrides,
                                          outClass.dtype, outClass.mem_space);
        }
        if (cClass.mem_space != outClass.mem_space)
        {
            currentTarget = addOpToEGraph(egraph, OpType::COPY_TO, {currentTarget}, outClass.shape, contigStrides,
                                          outClass.dtype, outClass.mem_space);
        }

        for (const Region &reg : nonInfRegions)
        {
            std::vector<int32_t> starts, ends, steps;
            for (const Dim &d : reg.region)
            {
                starts.push_back(d.start);
                ends.push_back(d.stop);
                steps.push_back(1);
            }

            EClassId startsId = egraph.addIntConst(starts);
            EClassId endsId = egraph.addIntConst(ends);
            EClassId stepsId = egraph.addIntConst(steps);

            std::vector<uint32_t> sliceShape;
            for (uint64_t d = 0; d < starts.size(); ++d)
            {
                sliceShape.push_back(ends[d] - starts[d]);
            }

            std::vector<uint64_t> sliceStridesV = vClass.strides;
            for (uint64_t d = 0; d < starts.size(); ++d)
            {
                sliceStridesV[d] *= steps[d];
            }
            EClassId sliceV = addOpToEGraph(egraph, OpType::SLICE, {varClass, startsId, endsId, stepsId}, sliceShape,
                                            sliceStridesV, vClass.dtype, vClass.mem_space);

            std::vector<uint64_t> sliceStridesC = cClass.strides;
            for (uint64_t d = 0; d < starts.size(); ++d)
            {
                sliceStridesC[d] *= steps[d];
            }
            EClassId sliceC = addOpToEGraph(egraph, OpType::SLICE, {constClass, startsId, endsId, stepsId},
                                            sliceShape, sliceStridesC, cClass.dtype, cClass.mem_space);

            std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);
            EClassId contigV = addOpToEGraph(egraph, OpType::CONTIGUOUS, {sliceV}, sliceShape, sliceContigStrides,
                                             vClass.dtype, vClass.mem_space);
            EClassId contigC = addOpToEGraph(egraph, OpType::CONTIGUOUS, {sliceC}, sliceShape, sliceContigStrides,
                                             cClass.dtype, cClass.mem_space);

            contigV = copyTo(egraph, contigV, outClass.mem_space);
            contigC = copyTo(egraph, contigC, outClass.mem_space);

            EClassId child0 = (constIdx == 0) ? contigC : contigV;
            EClassId child1 = (constIdx == 1) ? contigC : contigV;
            EClassId addId = addOpToEGraph(egraph, OpType::ADD, {child0, child1}, sliceShape, sliceContigStrides,
                                           outClass.dtype, outClass.mem_space);

            if (cached_update)
            {
                const EClassId shape_id = egraph.addIntConst(
                    std::vector<int32_t>(outClass.shape.begin(), outClass.shape.end()));
                const EClassId scatter = addOpToEGraph(egraph, OpType::SCATTER,
                    {addId, startsId, endsId, stepsId, shape_id}, outClass.shape, outClass.strides,
                    outClass.dtype, outClass.mem_space, EClassId(), SourceLocation::current(), "",
                    scatterFormTag(egraph, e_class_id));
                egraph.merge(e_class_id, scatter);
                return;
            }
            currentTarget = replaceRegion(egraph, currentTarget, addId, reg);
        }

        egraph.merge(e_class_id, currentTarget);
    }
};

// A class whose input was partially updated (one of its enode's children has
// a SCATTER form: "previous value with a region replaced") changes only in
// its own dirty region, and only the needed part of that matters. Expose
// contiguous(slice(class, box(dirty & needed))) so the slice push-down rules
// rewrite the class as a scatter of that region's computation, keeping the
// rest of its cached value.
struct ScatterPullUp : public Rule
{
    std::unique_ptr<Rule> clone() const override
    {
        return std::make_unique<ScatterPullUp>(*this);
    }

    std::vector<uint8_t> visited_classes;

    std::string name() const override
    {
        return "ScatterPullUp";
    }

    void clearVisited() override
    {
        visited_classes.clear();
    }

    void markVisited(uint32_t cid_val)
    {
        if (cid_val >= visited_classes.size())
            visited_classes.resize(cid_val + 1, 0);
        visited_classes[cid_val] = 1;
    }

    bool canMatchOp(OpType op) const override
    {
        return op != OpType::SCATTER && op != OpType::INPUT && op != OpType::CACHE;
    }

    bool prepareSweep(RuleCtx &ctx) override
    {
        return ctx.regions != nullptr;
    }

    static bool hasScatter(const EGraph &egraph, EClassId cid)
    {
        return ::hasScatter(egraph, cid);
    }

    // An input holding new values only in part of it (its dirty region).
    static bool isPartiallyDirtyInput(const EGraph &egraph, EClassId cid, const RuleCtx &ctx)
    {
        if (!ctx.regions)
            return false;
        const EClass &cls = egraph.getEClass(egraph.findConst(cid));
        if (cls.logical_id == LogicalId{})
            return false;
        bool input = false;
        for (ENodeId id : cls.enodes)
            input = input || egraph.getENode(id).getOpType() == OpType::INPUT;
        if (!input)
            return false;
        const std::vector<Region> &dirty = ctx.regions->dirtyOf(cls.logical_id);
        return !dirty.empty() && !bucket_regions::isFull(bucket_regions::boundingBox(dirty), cls.shape);
    }

    // A class is updated when it has a scatter form, is an input with a
    // partial dirty region, or is a view of an updated class: views are not
    // contiguous, so they cannot hold a scatter form themselves, but a
    // contiguous consumer of one can.
    static bool isUpdated(const EGraph &egraph, EClassId cid, uint32_t depth, const RuleCtx &ctx)
    {
        if (hasScatter(egraph, cid) || isPartiallyDirtyInput(egraph, cid, ctx))
            return true;
        if (depth == 0)
            return false;
        for (ENodeId id : egraph.getEClass(egraph.findConst(cid)).enodes)
        {
            const ENode &enode = egraph.getENode(id);
            const OpType op = enode.getOpType();
            if ((op == OpType::PERMUTE || op == OpType::RESHAPE || op == OpType::SLICE || op == OpType::REPEAT) &&
                !enode.getChildren().empty() && isUpdated(egraph, enode.getChildren()[0], depth - 1, ctx))
                return true;
        }
        return false;
    }

    bool regionFor(const EClass &cls, const RuleCtx &ctx, Region &box) const
    {
        if (!ctx.regions || cls.logical_id == LogicalId{})
            return false;
        // dirty & needed, cached per node by BucketRegions.
        const std::vector<Region> region = ctx.regions->liveOf(cls.logical_id);
        if (region.empty())
            return false;
        box = bucket_regions::boundingBox(region);
        return box.region.size() == cls.shape.size() && !bucket_regions::isFull(box, cls.shape);
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (!ctx.regions)
            return false;
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() == OpType::SCATTER || enode.getOpType() == OpType::INPUT ||
            enode.getOpType() == OpType::CACHE)
            return false;
        const EClassId cid = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx}));
        if (cid.value < visited_classes.size() && visited_classes[cid.value])
            return false;
        const EClass &cls = egraph.getEClass(cid);
        if (cls.logical_id == LogicalId{})
        {
            markVisited(cid.value);
            return false;
        }
        Region box;
        if (!regionFor(cls, ctx, box))
        {
            markVisited(cid.value);
            return false;
        }
        for (EClassId child : enode.getChildren())
        {
            if (isUpdated(egraph, child, 4, ctx))
                return true;
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const EClassId cid = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx}));
        markVisited(cid.value);
        const EClass cls = egraph.getEClass(cid);
        Region box;
        if (!regionFor(cls, ctx, box))
            return;
        std::vector<int32_t> starts, ends, steps;
        std::vector<uint32_t> shape;
        for (const Dim &d : box.region)
        {
            starts.push_back(static_cast<int32_t>(d.start));
            ends.push_back(static_cast<int32_t>(d.stop));
            steps.push_back(1);
            shape.push_back(d.stop - d.start);
        }
        EClassId slice = addOpToEGraph(egraph, OpType::SLICE,
                                       {cid, egraph.addIntConst(starts), egraph.addIntConst(ends), egraph.addIntConst(steps)},
                                       shape, cls.strides, cls.dtype, cls.mem_space);
        addOpToEGraph(egraph, OpType::CONTIGUOUS, {slice}, shape, calcContiguousStrides(shape), cls.dtype,
                      cls.mem_space);
    }
};

struct SlicePushDownElementwise : public Rule
{
    std::unique_ptr<Rule> clone() const override
    {
        return std::make_unique<SlicePushDownElementwise>(*this);
    }

    struct MatchKey
    {
        uint32_t contigClass;
        uint32_t sliceIdx;
        uint32_t srcNodeIdx;
        bool operator==(const MatchKey &o) const
        {
            return contigClass == o.contigClass && sliceIdx == o.sliceIdx && srcNodeIdx == o.srcNodeIdx;
        }
    };
    struct MatchKeyHash
    {
        std::uint64_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.contigClass) ^ (std::hash<uint32_t>{}(k.sliceIdx) << 1) ^
                   (std::hash<uint32_t>{}(k.srcNodeIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;
    SlicePushDownElementwise(bool allowPushDownOnProtected = false) : allowPushDownOnProtected(allowPushDownOnProtected)
    {
    }

    std::string name() const override
    {
        return "SlicePushDownElementwise";
    }

    bool canMatchOp(OpType op) const override
    {
        return op == OpType::CONTIGUOUS;
    }

    void clearVisited() override
    {
        visited.clear();
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (eNodeIdx < ctx.base_enode_watermark && ctx.sweep_filter &&
            eNodeIdx < ctx.sweep_filter->size() && (*ctx.sweep_filter)[eNodeIdx] == 0)
            return false;
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::CONTIGUOUS || enode.getChildren().empty())
            return false;

        const uint32_t contig_class = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx})).value;
        EClassId childClass = egraph.findConst(enode.getChildren()[0]);
        const EClass &childCls = egraph.getEClass(childClass);
        const bool is_base = (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark);

        for (ENodeId childNodeIdx : childCls.enodes)
        {
            const ENode &childNode = egraph.getENode(childNodeIdx);
            if (childNode.getOpType() != OpType::SLICE || childNode.getChildren().size() != 4)
                continue;

            EClassId srcClass = egraph.findConst(childNode.getChildren()[0]);
            if (!allowPushDownOnProtected && isEClassProtected(srcClass, ctx.protectedEClasses, egraph))
                continue;

            const EClass &srcCls = egraph.getEClass(srcClass);
            bool has_elem = false;
            for (ENodeId srcNodeIdx : srcCls.enodes)
            {
                OpType op = egraph.getENode(srcNodeIdx).getOpType();
                if (isElementwise(op) && op != OpType::COPY_TO && op != OpType::CONTIGUOUS)
                {
                    has_elem = true;
                    break;
                }
            }
            if (!has_elem)
                continue;

            const bool has_new_src = (!srcCls.enodes.empty() && srcCls.enodes.back().value >= ctx.base_enode_watermark);
            const bool can_scatter = (ctx.regions && srcCls.base_eclass_id != BaseEClassId{} && srcCls.logical_id != LogicalId{});
            if (is_base && childNodeIdx.value < ctx.base_enode_watermark && !has_new_src && !can_scatter)
                continue;

            Region box;
            if (!sliceBox(egraph, childNode, srcCls.shape, box))
                continue;

            const bool scatter_valid = can_scatter && ctx.regions->coversLive(srcCls.logical_id, box);
            const bool has_scatter = scatter_valid && hasScatter(egraph, srcClass);
            if (is_base && childNodeIdx.value < ctx.base_enode_watermark && !has_new_src && (!scatter_valid || has_scatter))
                continue;

            for (ENodeId srcNodeIdx : srcCls.enodes)
            {
                const ENode &opNode = egraph.getENode(srcNodeIdx);
                OpType op = opNode.getOpType();
                if (!isElementwise(op) || op == OpType::COPY_TO || op == OpType::CONTIGUOUS)
                    continue;
                if (is_base && childNodeIdx.value < ctx.base_enode_watermark && srcNodeIdx.value < ctx.base_enode_watermark &&
                    (!scatter_valid || has_scatter))
                    continue;
                if (visited.count(MatchKey{contig_class, childNodeIdx.value, srcNodeIdx.value}))
                    continue;
                const auto &opShape = opNode.getShape();
                bool hasBroadcastChild = false;
                for (EClassId cid : opNode.getChildren())
                    if (egraph.getEClass(egraph.findConst(cid)).shape != opShape)
                    {
                        hasBroadcastChild = true;
                        break;
                    }
                if (!hasBroadcastChild)
                    return true;
            }
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const uint32_t contig_class = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx})).value;
        const ENode contigNode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        EClassId sliceClass = egraph.findConst(contigNode.getChildren()[0]);
        const size_t numSlices = egraph.getEClass(sliceClass).enodes.size();

        for (size_t sl_i = 0; sl_i < numSlices; ++sl_i)
        {
            ENodeId sliceNodeIdx = egraph.getEClass(sliceClass).enodes[sl_i];
            const ENode sliceNode = egraph.getENode(sliceNodeIdx);
            if (sliceNode.getOpType() != OpType::SLICE || sliceNode.getChildren().size() != 4)
                continue;

            EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            bool has_elem = false;
            {
                const EClass &src = egraph.getEClass(srcClass);
                for (ENodeId s : src.enodes)
                {
                    OpType op = egraph.getENode(s).getOpType();
                    if (isElementwise(op) && op != OpType::COPY_TO && op != OpType::CONTIGUOUS)
                    {
                        has_elem = true;
                        break;
                    }
                }
            }
            if (!has_elem)
                continue;

            EClassId startsId = sliceNode.getChildren()[1];
            EClassId endsId = sliceNode.getChildren()[2];
            EClassId stepsId = sliceNode.getChildren()[3];

            const std::vector<uint32_t> sliceShape = sliceNode.getShape();
            std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);

            const EClass srcCls = egraph.getEClass(srcClass);
            Region box;
            if (!sliceBox(egraph, sliceNode, srcCls.shape, box))
                continue;

            const bool scatter_valid = scatterFormValid(ctx, srcCls, box);
            const bool has_scatter = scatter_valid && hasScatter(egraph, srcClass);
            if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                sliceNodeIdx.value < ctx.base_enode_watermark && (!scatter_valid || has_scatter) &&
                (srcCls.enodes.empty() || srcCls.enodes.back().value < ctx.base_enode_watermark))
                continue;

            const EClassId shapeId = scatter_valid
                ? egraph.addIntConst(std::vector<int32_t>(srcCls.shape.begin(), srcCls.shape.end()))
                : EClassId();

            const size_t numSrc = egraph.getEClass(srcClass).enodes.size();
            for (size_t s_i = 0; s_i < numSrc; ++s_i)
            {
                ENodeId srcNodeIdx = egraph.getEClass(srcClass).enodes[s_i];
                const ENode opNode = egraph.getENode(srcNodeIdx);
                OpType op = opNode.getOpType();
                if (!isElementwise(op) || op == OpType::COPY_TO || op == OpType::CONTIGUOUS)
                {
                    continue;
                }

                bool hasBroadcastChild = false;
                for (EClassId cid : opNode.getChildren())
                {
                    const auto &cls = egraph.getEClass(egraph.findConst(cid));
                    if (cls.shape != opNode.getShape())
                    {
                        hasBroadcastChild = true;
                        break;
                    }
                }
                if (hasBroadcastChild)
                    continue;

                if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                    sliceNodeIdx.value < ctx.base_enode_watermark && srcNodeIdx.value < ctx.base_enode_watermark &&
                    (!scatter_valid || has_scatter))
                    continue;

                MatchKey key{contig_class, sliceNodeIdx.value, srcNodeIdx.value};
                if (!visited.insert(key).second)
                    continue;

                std::vector<EClassId> newChildren;
                for (uint64_t childIdx = 0; childIdx < opNode.getChildren().size(); ++childIdx)
                {
                    EClassId childId = opNode.getChildren()[childIdx];
                    EClassId canonChildId = egraph.findConst(childId);
                    EClassId fillValue{};
                    DType childDType;
                    MemSpace childMem;
                    std::vector<uint64_t> childSliceStrides;
                    {
                        const EClass &childCls = egraph.getEClass(canonChildId);
                        childDType = childCls.dtype;
                        childMem = childCls.mem_space;
                        for (ENodeId e : childCls.enodes)
                        {
                            const ENode &cn = egraph.getENode(e);
                            if (cn.getOpType() == OpType::FILL && cn.getChildren().size() == 2)
                            {
                                fillValue = cn.getChildren()[0];
                                break;
                            }
                        }
                        if (fillValue == EClassId{})
                        {
                            childSliceStrides = childCls.strides;
                        }
                    }
                    if (fillValue != EClassId{})
                    {
                        newChildren.push_back(addOpToEGraph(
                            egraph, OpType::FILL,
                            {fillValue, egraph.addIntConst(std::vector<int32_t>(sliceShape.begin(), sliceShape.end()))},
                            sliceShape, std::vector<uint64_t>(sliceShape.size(), 0), childDType,
                            childMem));
                        continue;
                    }

                    EClassId childSlice =
                        addOpToEGraph(egraph, OpType::SLICE, {canonChildId, startsId, endsId, stepsId}, sliceShape,
                                      childSliceStrides, childDType, childMem);

                    // Determine if the op kernel requires a contiguous tensor for this input
                    bool reqContig = true;
                    if (opNode.getKernelId().value != 0 && KernelRegistry::get().hasKernel(opNode.getKernelId()))
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(opNode.getKernelId());
                        if (!kernel.requiresContiguous.empty())
                        {
                            uint64_t ruleIdx =
                                std::min(childIdx, static_cast<uint64_t>(kernel.requiresContiguous.size() - 1));
                            reqContig = kernel.requiresContiguous[ruleIdx];
                        }
                    }

                    if (reqContig)
                    {
                        childSlice = addOpToEGraph(egraph, OpType::CONTIGUOUS, {childSlice}, sliceShape,
                                                   sliceContigStrides, childDType, childMem);
                    }
                    newChildren.push_back(childSlice);
                }

                EClassId opEClass = addOpToEGraph(egraph, op, newChildren, sliceShape, sliceContigStrides,
                                                  sliceNode.getDType(), sliceNode.getMemSpace());
                // Keep the standalone result even when a cached update is valid.
                // Extraction must choose an acyclic implementation of either form.
                mergeRestricted(egraph, egraph.getENodeEClass(ENodeId{eNodeIdx}), opEClass);
                if (scatter_valid && !hasScatter(egraph, srcClass))
                {
                    srcClass = egraph.findConst(srcClass);
                    std::vector<uint32_t> sShape;
                    std::vector<uint64_t> sStrides;
                    DType sDType;
                    MemSpace sMem;
                    {
                        const EClass &srcEClass = egraph.getEClass(srcClass);
                        sShape = srcEClass.shape;
                        sStrides = srcEClass.strides;
                        sDType = srcEClass.dtype;
                        sMem = srcEClass.mem_space;
                    }
                    EClassId scatterClass =
                        addOpToEGraph(egraph, OpType::SCATTER, {opEClass, startsId, endsId, stepsId, shapeId},
                            sShape, sStrides, sDType, sMem,
                            EClassId(), SourceLocation::current(), "", scatterFormTag(egraph, srcClass));

                    egraph.merge(srcClass, scatterClass);
                }
            }
        }
    }
};

struct SlicePushDownDot : public Rule
{
    std::unique_ptr<Rule> clone() const override
    {
        return std::make_unique<SlicePushDownDot>(*this);
    }

    struct MatchKey
    {
        uint32_t contigClass;
        uint32_t childNodeIdx;
        uint32_t srcNodeIdx;
        bool operator==(const MatchKey &o) const
        {
            return contigClass == o.contigClass && childNodeIdx == o.childNodeIdx && srcNodeIdx == o.srcNodeIdx;
        }
    };

    struct MatchKeyHash
    {
        std::uint64_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.contigClass) ^ (std::hash<uint32_t>{}(k.childNodeIdx) << 1) ^
                   (std::hash<uint32_t>{}(k.srcNodeIdx) << 2);
        }
    };

    std::unordered_set<MatchKey, MatchKeyHash> visited;
    bool allowPushDownOnProtected;

    SlicePushDownDot(bool allowPushDownOnProtected = false) : allowPushDownOnProtected(allowPushDownOnProtected)
    {
    }

    std::string name() const override
    {
        return "SlicePushDownDot";
    }

    bool canMatchOp(OpType op) const override
    {
        return op == OpType::CONTIGUOUS;
    }

    void clearVisited() override
    {
        visited.clear();
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (eNodeIdx < ctx.base_enode_watermark && ctx.sweep_filter &&
            eNodeIdx < ctx.sweep_filter->size() && (*ctx.sweep_filter)[eNodeIdx] == 0)
            return false;
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::CONTIGUOUS || enode.getChildren().empty())
            return false;

        const uint32_t contig_class = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx})).value;
        EClassId childClass = egraph.findConst(enode.getChildren()[0]);
        for (ENodeId childNodeIdx : egraph.getEClass(childClass).enodes)
        {
            const ENode &childNode = egraph.getENode(childNodeIdx);
            if (childNode.getOpType() == OpType::SLICE && childNode.getChildren().size() == 4)
            {
                EClassId srcClass = egraph.findConst(childNode.getChildren()[0]);
                if (!allowPushDownOnProtected && isEClassProtected(srcClass, ctx.protectedEClasses, egraph))
                    continue;

                const EClass &srcCls = egraph.getEClass(srcClass);
                bool has_dot = false;
                for (ENodeId srcNodeIdx : srcCls.enodes)
                {
                    if (egraph.getENode(srcNodeIdx).getOpType() == OpType::DOT)
                    {
                        has_dot = true;
                        break;
                    }
                }
                if (!has_dot)
                    continue;

                Region box;
                if (!sliceBox(egraph, childNode, srcCls.shape, box))
                    continue;

                const bool scatter_valid = scatterFormValid(ctx, srcCls, box);
                const bool has_scatter = scatter_valid && hasScatter(egraph, srcClass);
                if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                    childNodeIdx.value < ctx.base_enode_watermark && (!scatter_valid || has_scatter) &&
                    (srcCls.enodes.empty() || srcCls.enodes.back().value < ctx.base_enode_watermark))
                    continue;

                for (ENodeId srcNodeIdx : srcCls.enodes)
                {
                    const ENode &opNode = egraph.getENode(srcNodeIdx);
                    if (opNode.getOpType() == OpType::DOT)
                    {
                        if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                            childNodeIdx.value < ctx.base_enode_watermark && srcNodeIdx.value < ctx.base_enode_watermark &&
                            (!scatter_valid || has_scatter))
                            continue;
                        MatchKey key{contig_class, childNodeIdx.value, srcNodeIdx.value};
                        if (visited.find(key) == visited.end())
                            return true;
                    }
                }
            }
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const uint32_t contig_class = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx})).value;
        const ENode contigNode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        EClassId sliceClass = egraph.findConst(contigNode.getChildren()[0]);
        const size_t numSlices = egraph.getEClass(sliceClass).enodes.size();

        for (size_t sl_i = 0; sl_i < numSlices; ++sl_i)
        {
            ENodeId sliceNodeIdx = egraph.getEClass(sliceClass).enodes[sl_i];
            const ENode sliceNode = egraph.getENode(sliceNodeIdx);
            if (sliceNode.getOpType() != OpType::SLICE || sliceNode.getChildren().size() != 4)
                continue;

            EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            bool has_dot = false;
            {
                const EClass &src = egraph.getEClass(srcClass);
                for (ENodeId s : src.enodes)
                {
                    if (egraph.getENode(s).getOpType() == OpType::DOT)
                    {
                        has_dot = true;
                        break;
                    }
                }
            }
            if (!has_dot)
                continue;

            EClassId startsId = sliceNode.getChildren()[1];
            EClassId endsId = sliceNode.getChildren()[2];
            EClassId stepsId = sliceNode.getChildren()[3];

            const EClass srcCls = egraph.getEClass(srcClass);
            Region box;
            if (!sliceBox(egraph, sliceNode, srcCls.shape, box))
                continue;

            const bool scatter_valid = scatterFormValid(ctx, srcCls, box);
            const bool has_scatter = scatter_valid && hasScatter(egraph, srcClass);
            if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                sliceNodeIdx.value < ctx.base_enode_watermark && (!scatter_valid || has_scatter) &&
                (srcCls.enodes.empty() || srcCls.enodes.back().value < ctx.base_enode_watermark))
                continue;

            const uint32_t rank = static_cast<uint32_t>(box.region.size());
            if (rank != 2 && rank != 3 && rank != 4)
                continue;

            const std::vector<uint32_t> sliceShape = sliceNode.getShape();
            const std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);

            const size_t numSrc = srcCls.enodes.size();
            for (size_t s_i = 0; s_i < numSrc; ++s_i)
            {
                ENodeId srcNodeIdx = egraph.getEClass(srcClass).enodes[s_i];
                const ENode dotNode = egraph.getENode(srcNodeIdx);
                if (dotNode.getOpType() != OpType::DOT)
                    continue;

                if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                    sliceNodeIdx.value < ctx.base_enode_watermark && srcNodeIdx.value < ctx.base_enode_watermark &&
                    (!scatter_valid || has_scatter))
                    continue;

                MatchKey key{contig_class, sliceNodeIdx.value, srcNodeIdx.value};
                if (!visited.insert(key).second)
                    continue;

                EClassId aClassId = dotNode.getChildren()[0];
                EClassId bClassId = dotNode.getChildren()[1];

                uint32_t K = (rank == 2)   ? egraph.getEClass(egraph.findConst(aClassId)).shape[1]
                             : (rank == 3) ? egraph.getEClass(egraph.findConst(aClassId)).shape[2]
                                           : egraph.getEClass(egraph.findConst(aClassId)).shape[3];

                std::vector<int32_t> startsA, endsA, stepsA(rank, 1);
                std::vector<int32_t> startsB, endsB, stepsB(rank, 1);

                if (rank == 2)
                {
                    startsA = {static_cast<int32_t>(box.region[0].start), 0};
                    endsA = {static_cast<int32_t>(box.region[0].stop), static_cast<int32_t>(K)};

                    startsB = {0, static_cast<int32_t>(box.region[1].start)};
                    endsB = {static_cast<int32_t>(K), static_cast<int32_t>(box.region[1].stop)};
                }
                else if (rank == 3)
                {
                    int32_t k0 = 0, k1 = static_cast<int32_t>(K);
                    const EClass aCls = egraph.getEClass(egraph.findConst(aClassId));
                    if (ctx.regions && ctx.regions->facts && aCls.logical_id != LogicalId{})
                    {
                        auto it = ctx.regions->facts->support.find(aCls.logical_id);
                        if (it != ctx.regions->facts->support.end())
                        {
                            Region aBox;
                            aBox.region = {box.region[0], box.region[1], {0, K}};
                            if (auto range = bucket_regions::axisRange(it->second, aBox, 2, false))
                            {
                                k0 = static_cast<int32_t>(range->start);
                                k1 = static_cast<int32_t>(range->stop);
                            }
                        }
                    }
                    startsA = {static_cast<int32_t>(box.region[0].start), static_cast<int32_t>(box.region[1].start), k0};
                    endsA = {static_cast<int32_t>(box.region[0].stop), static_cast<int32_t>(box.region[1].stop), k1};

                    startsB = {static_cast<int32_t>(box.region[0].start), k0, static_cast<int32_t>(box.region[2].start)};
                    endsB = {static_cast<int32_t>(box.region[0].stop), k1, static_cast<int32_t>(box.region[2].stop)};
                }
                else if (rank == 4)
                {
                    startsA = {static_cast<int32_t>(box.region[0].start), static_cast<int32_t>(box.region[1].start),
                               static_cast<int32_t>(box.region[2].start), 0};
                    endsA = {static_cast<int32_t>(box.region[0].stop), static_cast<int32_t>(box.region[1].stop),
                             static_cast<int32_t>(box.region[2].stop), static_cast<int32_t>(K)};
                    startsB = {static_cast<int32_t>(box.region[0].start), static_cast<int32_t>(box.region[1].start), 0,
                               static_cast<int32_t>(box.region[3].start)};
                    endsB = {static_cast<int32_t>(box.region[0].stop), static_cast<int32_t>(box.region[1].stop),
                             static_cast<int32_t>(K), static_cast<int32_t>(box.region[3].stop)};
                }

                for (size_t d = 0; d < rank; ++d)
                {
                    if (endsA[d] <= startsA[d] || endsB[d] <= startsB[d])
                        Error::throw_err("[SlicePushDownDot.apply] empty operand slice");
                }
                EClassId startsIdA = egraph.addIntConst(startsA);
                EClassId endsIdA = egraph.addIntConst(endsA);
                EClassId stepsIdA = egraph.addIntConst(stepsA);

                EClassId startsIdB = egraph.addIntConst(startsB);
                EClassId endsIdB = egraph.addIntConst(endsB);
                EClassId stepsIdB = egraph.addIntConst(stepsB);

                auto createSlice = [&](EClassId classId, const std::vector<int32_t> &st, const std::vector<int32_t> &en,
                                       EClassId stId, EClassId enId, EClassId stepId) {
                    EClassId canonId = egraph.findConst(classId);
                    const EClass cls = egraph.getEClass(canonId);
                    bool whole = st.size() == cls.shape.size();
                    for (size_t d = 0; whole && d < st.size(); ++d)
                        whole = st[d] == 0 && en[d] == static_cast<int32_t>(cls.shape[d]);
                    if (whole)
                        return canonId;
                    const std::vector<uint64_t> sStrides = cls.strides;
                    DType cDtype = cls.dtype;

                    std::vector<uint32_t> sShape;
                    for (uint64_t d = 0; d < st.size(); ++d)
                        sShape.push_back(en[d] - st[d]);

                    EClassId sClass = addOpToEGraph(egraph, OpType::SLICE, {canonId, stId, enId, stepId}, sShape,
                                                    sStrides, cDtype, sliceNode.getMemSpace());
                    EClassId sContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {sClass}, sShape,
                                                     calcContiguousStrides(sShape), cDtype, sliceNode.getMemSpace());
                    return sContig;
                };

                EClassId aSliced = createSlice(aClassId, startsA, endsA, startsIdA, endsIdA, stepsIdA);
                EClassId bSliced = createSlice(bClassId, startsB, endsB, startsIdB, endsIdB, stepsIdB);

                EClassId dotEClass = addOpToEGraph(egraph, OpType::DOT, {aSliced, bSliced}, sliceShape,
                                                   sliceContigStrides, sliceNode.getDType(), sliceNode.getMemSpace());
                srcClass = egraph.findConst(srcClass);
                mergeRestricted(egraph, egraph.getENodeEClass(ENodeId{eNodeIdx}), dotEClass);
                if (scatter_valid && !hasScatter(egraph, srcClass))
                {
                    const EClass srcEClass = egraph.getEClass(srcClass);
                    EClassId shapeId =
                        egraph.addIntConst(std::vector<int32_t>(srcEClass.shape.begin(), srcEClass.shape.end()));
                    EClassId scatterClass =
                        addOpToEGraph(egraph, OpType::SCATTER, {dotEClass, startsId, endsId, stepsId, shapeId},
                            srcEClass.shape, srcEClass.strides, srcEClass.dtype, srcEClass.mem_space,
                            EClassId(), SourceLocation::current(), "", scatterFormTag(egraph, srcClass));

                    egraph.merge(srcClass, scatterClass);
                }
            }
        }
    }
};

// contiguous(slice(src)) where src is a structural (non-elementwise) op:
// rewrite src as a scatter of the same op computed only for the slice
// region, from slices of its inputs. Covers gather (table rows by index),
// repeat of a broadcast axis, sum/max over an axis, reshape and permute
// that keep the sliced axes, concat, and slice of slice.
struct SlicePushDownStructural : public Rule
{
    std::unique_ptr<Rule> clone() const override
    {
        return std::make_unique<SlicePushDownStructural>(*this);
    }

    struct MatchKey
    {
        uint32_t contigClass;
        uint32_t sliceIdx;
        bool operator==(const MatchKey &o) const
        {
            return contigClass == o.contigClass && sliceIdx == o.sliceIdx;
        }
    };
    struct MatchKeyHash
    {
        std::uint64_t operator()(const MatchKey &k) const
        {
            return (static_cast<uint64_t>(k.contigClass) * 0x9E3779B97F4A7C15ULL) ^
                   (static_cast<uint64_t>(k.sliceIdx) << 21) ^ k.sliceIdx;
        }
    };
    std::unordered_set<MatchKey, MatchKeyHash> visited;
    // Regions of the bucket being saturated, set by apply.
    const BucketRegions *regions_ = nullptr;

    std::string name() const override
    {
        return "SlicePushDownStructural";
    }

    bool canMatchOp(OpType op) const override
    {
        return op == OpType::CONTIGUOUS;
    }

    static bool isStructural(OpType op)
    {
        return op == OpType::GATHER || op == OpType::REPEAT || op == OpType::SUM || op == OpType::MAX ||
               op == OpType::RESHAPE || op == OpType::PERMUTE || op == OpType::CONCAT || op == OpType::SLICE ||
               op == OpType::CONTIGUOUS || op == OpType::FILL;
    }

    static bool sliceBox(const EGraph &egraph, const ENode &sliceNode, const std::vector<uint32_t> &srcShape,
                         Region &box)
    {
        return ::sliceBox(egraph, sliceNode, srcShape, box);
    }

    static std::vector<uint32_t> boxShape(const Region &box)
    {
        return ::boxShape(box);
    }

    static Region fullBox(const std::vector<uint32_t> &shape)
    {
        return ::fullBox(shape);
    }

    SliceMemoTable local_memo;
    SliceMemoTable *memo_ = nullptr;

    struct RestrictedMemoKey
    {
        uint32_t op_id;
        Region box;
        bool operator==(const RestrictedMemoKey &o) const
        {
            return op_id == o.op_id && box == o.box;
        }
    };
    struct RestrictedMemoKeyHash
    {
        size_t operator()(const RestrictedMemoKey &k) const
        {
            size_t h = std::hash<uint32_t>{}(k.op_id);
            for (const auto &d : k.box.region)
            {
                h = h * 31 + d.start;
                h = h * 31 + d.stop;
            }
            return h;
        }
    };
    std::unordered_map<RestrictedMemoKey, EClassId, RestrictedMemoKeyHash> restricted_memo;

    // contiguous(slice(cls, box)), or cls itself when box covers it.
    static EClassId sliceOf(EGraph &egraph, EClassId cid, const Region &box, SliceMemoTable *memo = nullptr)
    {
        cid = egraph.findConst(cid);
        if (memo)
        {
            SliceMemoKey key{cid.value, box};
            auto it = memo->find(key);
            if (it != memo->end())
                return egraph.findConst(it->second);
        }
        std::vector<uint32_t> shape;
        std::vector<uint64_t> strides;
        DType dtype;
        MemSpace mem_space;
        {
            const EClass &cls = egraph.getEClass(cid);
            if (box.region.size() != cls.shape.size())
                Error::throw_err("[SlicePushDownStructural] slice rank mismatch");
            bool full = true;
            for (size_t d = 0; d < box.region.size(); ++d)
            {
                if (box.region[d].start >= box.region[d].stop || box.region[d].stop > cls.shape[d])
                    Error::throw_err("[SlicePushDownStructural] slice box outside its tensor");
                full = full && box.region[d].start == 0 && box.region[d].stop == cls.shape[d];
            }
            if (full)
            {
                if (memo)
                    (*memo)[SliceMemoKey{cid.value, box}] = cid;
                return cid;
            }
            shape = ::boxShape(box);
            strides = cls.strides;
            dtype = cls.dtype;
            mem_space = cls.mem_space;
        }
        std::vector<int32_t> starts, ends, steps;
        starts.reserve(box.region.size());
        ends.reserve(box.region.size());
        steps.reserve(box.region.size());
        for (size_t d = 0; d < box.region.size(); ++d)
        {
            starts.push_back(static_cast<int32_t>(box.region[d].start));
            ends.push_back(static_cast<int32_t>(box.region[d].stop));
            steps.push_back(1);
        }
        EClassId slice = addOpToEGraph(egraph, OpType::SLICE,
                                       {cid, egraph.addIntConst(starts), egraph.addIntConst(ends), egraph.addIntConst(steps)},
                                       shape, strides, dtype, mem_space);
        EClassId result = addOpToEGraph(egraph, OpType::CONTIGUOUS, {slice}, shape, calcContiguousStrides(shape), dtype,
                                        mem_space);
        if (memo)
            (*memo)[SliceMemoKey{cid.value, box}] = result;
        return result;
    }

    EClassId memoSliceOf(EGraph &egraph, EClassId cid, const Region &box)
    {
        return sliceOf(egraph, cid, box, memo_ ? memo_ : &local_memo);
    }

    // src's op computed only for `box`; an invalid id if the op cannot be
    // restricted to this box.
    EClassId buildRestricted(EGraph &egraph, ENodeId opId, const std::vector<uint32_t> &srcShape,
                            DType srcDType, MemSpace srcMemSpace, const Region &box)
    {
        RestrictedMemoKey rkey{opId.value, box};
        auto rit = restricted_memo.find(rkey);
        if (rit != restricted_memo.end())
            return egraph.findConst(rit->second);
        EClassId res = buildRestrictedImpl(egraph, opId, srcShape, srcDType, srcMemSpace, box);
        restricted_memo[rkey] = res;
        return res;
    }

    EClassId buildRestrictedImpl(EGraph &egraph, ENodeId opId, const std::vector<uint32_t> &srcShape,
                                DType srcDType, MemSpace srcMemSpace, const Region &box)
    {
        std::vector<EClassId> children;
        OpType opType;
        {
            const ENode &op = egraph.getENode(opId);
            children = op.getChildren();
            opType = op.getOpType();
        }
        const std::vector<uint32_t> outShape = boxShape(box);
        switch (opType)
        {
        case OpType::GATHER: {
            const size_t idxRank = egraph.getEClass(egraph.findConst(children[1])).shape.size();
            Region idxBox;
            for (size_t d = 0; d < idxRank; ++d)
                idxBox.region.push_back(box.region[d]);
            for (size_t d = idxRank; d < box.region.size(); ++d)
                if (box.region[d].start != 0 || box.region[d].stop != srcShape[d])
                    return EClassId{};
            EClassId idxSlice = memoSliceOf(egraph, children[1], idxBox);
            return addOpToEGraph(egraph, OpType::GATHER, {children[0], idxSlice}, outShape,
                                 calcContiguousStrides(outShape), srcDType, srcMemSpace);
        }
        case OpType::REPEAT: {
            std::vector<int32_t> axisV;
            if (!constInts(egraph, children[2], axisV) || axisV.size() != 1)
                return EClassId{};
            const auto inShape = egraph.getEClass(egraph.findConst(children[0])).shape;
            const int32_t axis = axisV[0] < 0 ? axisV[0] + (int32_t)inShape.size() : axisV[0];
            Region inBox = box;
            if (inShape[axis] == 1)
                inBox.region[axis] = {0, 1};
            else if (box.region[axis].start == 0 && box.region[axis].stop == srcShape[axis])
                inBox.region[axis] = {0, inShape[axis]};
            else
                return EClassId{};
            EClassId inSlice = memoSliceOf(egraph, children[0], inBox);
            const uint32_t reps = inShape[axis] == 1 ? outShape[axis] : srcShape[axis] / inShape[axis];
            const std::vector<uint32_t> inBoxSh = boxShape(inBox);
            std::vector<uint64_t> strides = egraph.getEClass(egraph.findConst(inSlice)).strides;
            for (size_t d = 0; d < outShape.size(); ++d)
                if (inBoxSh[d] != outShape[d])
                    strides[d] = 0;
            return addOpToEGraph(egraph, OpType::REPEAT, {inSlice, egraph.addIntConst({(int32_t)reps}), children[2]},
                                 outShape, strides, srcDType, srcMemSpace);
        }
        case OpType::SUM:
        case OpType::MAX: {
            std::vector<int32_t> axisV;
            if (!constInts(egraph, children[1], axisV) || axisV.size() != 1)
                return EClassId{};
            const auto inClsId = egraph.findConst(children[0]);
            std::vector<uint32_t> inShape;
            LogicalId inLogicalId;
            {
                const EClass &in = egraph.getEClass(inClsId);
                inShape = in.shape;
                inLogicalId = in.logical_id;
            }
            const int32_t axis = axisV[0] < 0 ? axisV[0] + (int32_t)inShape.size() : axisV[0];
            Region inBox = box;
            inBox.region[axis] = {0, inShape[axis]};
            if (regions_ && regions_->facts && inLogicalId != LogicalId{})
            {
                const auto &valid =
                    opType == OpType::SUM ? regions_->facts->support : regions_->facts->finite;
                auto it = valid.find(inLogicalId);
                if (it != valid.end())
                    if (auto range = bucket_regions::axisRange(it->second, inBox, axis,
                                                               opType == OpType::MAX))
                        inBox.region[axis] = *range;
            }
            EClassId inSlice = memoSliceOf(egraph, children[0], inBox);
            return addOpToEGraph(egraph, opType, {inSlice, children[1]}, outShape,
                                 calcContiguousStrides(outShape), srcDType, srcMemSpace);
        }
        case OpType::RESHAPE: {
            const auto inClsId = egraph.findConst(children[0]);
            std::vector<uint32_t> inShape;
            {
                const EClass &in = egraph.getEClass(inClsId);
                if (!isContiguous(in))
                    return EClassId{};
                inShape = in.shape;
            }
            auto prod = [](const std::vector<uint32_t> &s, size_t a, size_t b) {
                uint64_t p = 1;
                for (size_t i = a; i < b; ++i)
                    p *= s[i];
                return p;
            };
            Region inBox = fullBox(inShape);
            std::vector<int32_t> newShape(srcShape.begin(), srcShape.end());
            for (size_t d = 0; d < box.region.size(); ++d)
            {
                if (box.region[d].start == 0 && box.region[d].stop == srcShape[d])
                    continue;
                bool mapped = false;
                for (size_t e = 0; e < inShape.size() && !mapped; ++e)
                {
                    if (inShape[e] == srcShape[d] && prod(inShape, 0, e) == prod(srcShape, 0, d) &&
                        prod(inShape, e + 1, inShape.size()) == prod(srcShape, d + 1, srcShape.size()))
                    {
                        inBox.region[e] = box.region[d];
                        mapped = true;
                    }
                }
                if (!mapped)
                    return EClassId{};
                newShape[d] = (int32_t)(box.region[d].stop - box.region[d].start);
            }
            EClassId inSlice = memoSliceOf(egraph, children[0], inBox);
            return addOpToEGraph(egraph, OpType::RESHAPE, {inSlice, egraph.addIntConst(newShape)}, outShape,
                                 calcContiguousStrides(outShape), srcDType, srcMemSpace);
        }
        case OpType::PERMUTE: {
            std::vector<int32_t> dims;
            if (!constInts(egraph, children[1], dims) || dims.size() != box.region.size())
                return EClassId{};
            const auto inShape = egraph.getEClass(egraph.findConst(children[0])).shape;
            Region inBox = fullBox(inShape);
            for (size_t i = 0; i < dims.size(); ++i)
                inBox.region[dims[i]] = box.region[i];
            EClassId inSlice = memoSliceOf(egraph, children[0], inBox);
            const std::vector<uint64_t> inStrides = egraph.getEClass(egraph.findConst(inSlice)).strides;
            std::vector<uint64_t> strides(dims.size());
            for (size_t i = 0; i < dims.size(); ++i)
                strides[i] = inStrides[dims[i]];
            return addOpToEGraph(egraph, OpType::PERMUTE, {inSlice, children[1]}, outShape, strides, srcDType,
                                 srcMemSpace);
        }
        case OpType::CONCAT: {
            std::vector<int32_t> axisV;
            if (!constInts(egraph, children[0], axisV) || axisV.size() != 1)
                return EClassId{};
            const int32_t axis = axisV[0] < 0 ? axisV[0] + (int32_t)srcShape.size() : axisV[0];
            std::vector<EClassId> pieces = {children[0]};
            uint32_t offset = 0;
            for (size_t i = 1; i < children.size(); ++i)
            {
                const auto &partShape = egraph.getEClass(egraph.findConst(children[i])).shape;
                const uint32_t lo = offset, hi = offset + partShape[axis];
                offset = hi;
                if (lo >= box.region[axis].stop)
                    break;
                const uint32_t st = std::max(lo, box.region[axis].start), en = std::min(hi, box.region[axis].stop);
                if (st >= en)
                    continue;
                Region partBox = box;
                partBox.region[axis] = {st - lo, en - lo};
                pieces.push_back(memoSliceOf(egraph, children[i], partBox));
            }
            if (pieces.size() == 2)
                return pieces[1];
            return addOpToEGraph(egraph, OpType::CONCAT, pieces, outShape, calcContiguousStrides(outShape), srcDType,
                                 srcMemSpace);
        }
        case OpType::SLICE: {
            std::vector<int32_t> starts, steps;
            if (!constInts(egraph, children[1], starts) || !constInts(egraph, children[3], steps))
                return EClassId{};
            const auto inShape = egraph.getEClass(egraph.findConst(children[0])).shape;
            if (starts.size() != inShape.size() || steps.size() != inShape.size())
                return EClassId{};
            Region inBox = box;
            for (size_t d = 0; d < starts.size(); ++d)
            {
                const int32_t st = starts[d] < 0 ? starts[d] + (int32_t)inShape[d] : starts[d];
                if (steps[d] != 1 || st < 0)
                    return EClassId{};
                inBox.region[d].start += (uint32_t)st;
                inBox.region[d].stop += (uint32_t)st;
            }
            return memoSliceOf(egraph, children[0], inBox);
        }
        case OpType::CONTIGUOUS:
            // A box of a contiguous copy is a contiguous copy of the box.
            return memoSliceOf(egraph, children[0], box);
        case OpType::FILL:
            // A box of a fill is a fill of the box's shape (a view: zero
            // strides over its scalar value).
            return addOpToEGraph(egraph, OpType::FILL,
                                 {children[0], egraph.addIntConst(std::vector<int32_t>(outShape.begin(), outShape.end()))},
                                 outShape, std::vector<uint64_t>(outShape.size(), 0), srcDType, srcMemSpace);
        default:
            return EClassId{};
        }
    }

    void clearVisited() override
    {
        visited.clear();
        local_memo.clear();
        // restricted_memo is retained because it memoizes buildRestricted,
        // which is a deterministic function of (ENodeId, box).
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (eNodeIdx < ctx.base_enode_watermark && ctx.sweep_filter &&
            eNodeIdx < ctx.sweep_filter->size() && (*ctx.sweep_filter)[eNodeIdx] == 0)
            return false;
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::CONTIGUOUS || enode.getChildren().empty())
            return false;
        const uint32_t contig_class = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx})).value;
        EClassId sliceClass = egraph.findConst(enode.getChildren()[0]);
        const auto &sliceEnodes = egraph.getEClass(sliceClass).enodes;
        for (ENodeId sliceIdx : sliceEnodes)
        {
            if (visited.count(MatchKey{contig_class, sliceIdx.value}))
                continue;
            const ENode &sliceNode = egraph.getENode(sliceIdx);
            if (sliceNode.getOpType() != OpType::SLICE || sliceNode.getChildren().size() != 4)
                continue;
            EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            if (isEClassProtected(srcClass, ctx.protectedEClasses, egraph))
                continue;
            const EClass &src = egraph.getEClass(srcClass);
            bool has_structural = false;
            for (ENodeId srcIdx : src.enodes)
            {
                if (isStructural(egraph.getENode(srcIdx).getOpType()))
                {
                    has_structural = true;
                    break;
                }
            }
            if (!has_structural)
                continue;
            Region box;
            if (!sliceBox(egraph, sliceNode, src.shape, box))
                continue;
            const bool scatter_valid = scatterFormValid(ctx, src, box);
            const bool has_scatter = scatter_valid && hasScatter(egraph, srcClass);
            if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                sliceIdx.value < ctx.base_enode_watermark && (!scatter_valid || has_scatter) &&
                (src.enodes.empty() || src.enodes.back().value < ctx.base_enode_watermark))
                continue;
            const auto &srcEnodes = src.enodes;
            for (ENodeId srcIdx : srcEnodes)
            {
                if (!isStructural(egraph.getENode(srcIdx).getOpType()))
                    continue;
                if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                    sliceIdx.value < ctx.base_enode_watermark && srcIdx.value < ctx.base_enode_watermark &&
                    (!scatter_valid || has_scatter))
                    continue;
                return true;
            }
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const uint32_t contig_class = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx})).value;
        regions_ = ctx.regions;
        memo_ = ctx.slice_memo;
        const ENode contigNode = egraph.getENode(ENodeId{eNodeIdx});
        const EClassId sliceClass = egraph.findConst(contigNode.getChildren()[0]);
        const size_t numSlices = egraph.getEClass(sliceClass).enodes.size();
        for (size_t sl_i = 0; sl_i < numSlices; ++sl_i)
        {
            ENodeId sliceIdx = egraph.getEClass(sliceClass).enodes[sl_i];
            if (visited.count(MatchKey{contig_class, sliceIdx.value}))
                continue;
            const ENode sliceNode = egraph.getENode(sliceIdx);
            if (sliceNode.getOpType() != OpType::SLICE || sliceNode.getChildren().size() != 4)
                continue;
            EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            bool has_structural = false;
            {
                const EClass &src = egraph.getEClass(srcClass);
                for (ENodeId s : src.enodes)
                {
                    if (isStructural(egraph.getENode(s).getOpType()))
                    {
                        has_structural = true;
                        break;
                    }
                }
            }
            if (!has_structural)
                continue;

            std::vector<uint32_t> srcShape;
            DType srcDType;
            MemSpace srcMemSpace;
            {
                const EClass &src = egraph.getEClass(srcClass);
                srcShape = src.shape;
                srcDType = src.dtype;
                srcMemSpace = src.mem_space;
            }
            Region box;
            if (!sliceBox(egraph, sliceNode, srcShape, box))
                continue;
            const bool scatter_valid = scatterFormValid(ctx, egraph.getEClass(srcClass), box);
            const bool has_scatter = scatter_valid && hasScatter(egraph, srcClass);
            if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                sliceIdx.value < ctx.base_enode_watermark && (!scatter_valid || has_scatter) &&
                (egraph.getEClass(srcClass).enodes.empty() || egraph.getEClass(srcClass).enodes.back().value < ctx.base_enode_watermark))
                continue;

            EClassId startsConstId{}, endsConstId{}, stepsConstId{}, shapeId{};
            bool consts_prepared = false;

            visited.insert(MatchKey{contig_class, sliceIdx.value});

            const size_t numSrc = egraph.getEClass(srcClass).enodes.size();
            for (size_t s_i = 0; s_i < numSrc; ++s_i)
            {
                ENodeId srcIdx = egraph.getEClass(srcClass).enodes[s_i];
                if (!isStructural(egraph.getENode(srcIdx).getOpType()))
                    continue;
                if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                    sliceIdx.value < ctx.base_enode_watermark && srcIdx.value < ctx.base_enode_watermark &&
                    (!scatter_valid || has_scatter))
                    continue;
                const EClassId restricted = buildRestricted(egraph, srcIdx, srcShape, srcDType, srcMemSpace, box);
                if (restricted == EClassId{})
                    continue;
                EClassId restrictedContig = egraph.findConst(restricted);
                std::vector<uint32_t> rShape;
                DType rDType;
                MemSpace rMem;
                bool needsContig = false;
                {
                    const EClass &rc = egraph.getEClass(restrictedContig);
                    if (!isContiguous(rc))
                    {
                        needsContig = true;
                        rShape = rc.shape;
                        rDType = rc.dtype;
                        rMem = rc.mem_space;
                    }
                }
                if (needsContig)
                {
                    restrictedContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {restrictedContig}, rShape,
                                                     calcContiguousStrides(rShape), rDType, rMem);
                }
                mergeRestricted(egraph, egraph.getENodeEClass(ENodeId{eNodeIdx}), restrictedContig);
                if (scatter_valid && !hasScatter(egraph, srcClass))
                {
                    srcClass = egraph.findConst(srcClass);
                    const EClass srcAfter = egraph.getEClass(srcClass);
                    if (!isContiguous(srcAfter) || !scatterFormValid(ctx, srcAfter, box))
                        continue;
                    if (!consts_prepared)
                    {
                        std::vector<int32_t> starts, ends, steps;
                        starts.reserve(box.region.size());
                        ends.reserve(box.region.size());
                        steps.reserve(box.region.size());
                        for (const Dim &d : box.region)
                        {
                            starts.push_back(static_cast<int32_t>(d.start));
                            ends.push_back(static_cast<int32_t>(d.stop));
                            steps.push_back(1);
                        }
                        startsConstId = egraph.addIntConst(starts);
                        endsConstId = egraph.addIntConst(ends);
                        stepsConstId = egraph.addIntConst(steps);
                        shapeId = egraph.addIntConst(std::vector<int32_t>(srcShape.begin(), srcShape.end()));
                        consts_prepared = true;
                    }
                    const EClassId scatterClass = addOpToEGraph(egraph, OpType::SCATTER,
                        {restricted, startsConstId, endsConstId, stepsConstId, shapeId},
                        srcAfter.shape, srcAfter.strides, srcAfter.dtype, srcAfter.mem_space, EClassId(), SourceLocation::current(),
                        "", scatterFormTag(egraph, srcClass));
                    egraph.merge(srcClass, scatterClass);
                    srcClass = egraph.findConst(srcClass);
                }
                if (!scatter_valid || hasScatter(egraph, srcClass))
                    break;
            }
        }
    }
};

// Row-locality of a fused kernel, from its reference pattern: every op that
// reads row data (data depending on an input whose leading dims are the
// output's) is elementwise, a contiguous copy, a reduction over the last axis
// or a repeat along the last axis. Output row r then depends only on row r of
// those inputs and on the other (broadcast) inputs as a whole.
struct FusedRowLocality
{
    bool rowLocal = false;
    std::vector<bool> rowInput;
};

inline const FusedRowLocality &fusedRowLocality(const std::string &opName)
{
    static std::mutex mtx;
    static std::unordered_map<std::string, FusedRowLocality> cache;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = cache.find(opName);
    if (it != cache.end())
        return it->second;
    FusedRowLocality result;
    const ReferenceGraphEntry *entry = ReferenceGraphRegistry::get().getFactory(opName);
    if (entry && entry->factory && entry->dummyShapes.size() >= entry->min_num_inputs &&
        entry->dtypes.size() >= entry->min_num_inputs)
    {
        Graph g;
        std::vector<LogicalId> vars;
        for (uint32_t i = 0; i < entry->min_num_inputs; ++i)
            vars.push_back(g.input(entry->dummyShapes[i], entry->dtypes[i]));
        const LogicalId patternRoot = entry->factory(vars, g);
        const std::vector<LogicalId> topo = topologicalSort({patternRoot}, g);
        ShapePropagator prop;
        // Patterns whose shapes depend on runtime inputs (shape tensors) cannot
        // be shown row-local from the pattern alone.
        bool shapesKnown = true;
        try
        {
            for (LogicalId id : topo)
                if (g.getNode(id).opType != OpType::INPUT)
                    prop.inferShape(id, g);
        }
        catch (const std::exception &)
        {
            shapesKnown = false;
        }
        const auto outShape = shapesKnown ? g.getNode(patternRoot).getShape() : std::vector<uint32_t>{};
        if (outShape.size() >= 2)
        {
            const size_t rank = outShape.size();
            auto leadingMatch = [&](const std::vector<uint32_t> &shape) {
                if (shape.size() != rank)
                    return false;
                for (size_t d = 0; d + 1 < rank; ++d)
                    if (shape[d] != outShape[d])
                        return false;
                return true;
            };
            std::unordered_map<LogicalId, int32_t> varIndex;
            for (size_t i = 0; i < vars.size(); ++i)
            {
                varIndex[vars[i]] = static_cast<int32_t>(i);
                result.rowInput.push_back(leadingMatch(g.getNode(vars[i]).getShape()));
            }
            auto constAxis = [&](LogicalId id, int32_t &axis) {
                auto data = g.constantStaging.find(id);
                if (data == g.constantStaging.end() || data->second->size() != sizeof(int32_t))
                    return false;
                std::memcpy(&axis, data->second->data(), sizeof(axis));
                if (axis < 0)
                    axis += static_cast<int32_t>(rank);
                return true;
            };
            std::unordered_map<LogicalId, bool> isRow;
            bool ok = true;
            for (LogicalId id : topo)
            {
                const TensorNode &node = g.getNode(id);
                if (node.opType == OpType::INPUT)
                {
                    auto v = varIndex.find(id);
                    isRow[id] = v != varIndex.end() && result.rowInput[v->second];
                    continue;
                }
                bool anyRow = false;
                for (LogicalId c : node.child_ids)
                    anyRow = anyRow || isRow[c];
                isRow[id] = anyRow;
                if (!anyRow)
                    continue;
                int32_t axis = -1;
                const OpType op = node.opType;
                const bool allowed =
                    (isElementwise(op) && op != OpType::COPY_TO) || op == OpType::CONTIGUOUS ||
                    ((op == OpType::SUM || op == OpType::MAX) && node.child_ids.size() == 2 &&
                     constAxis(node.child_ids[1], axis) && axis == static_cast<int32_t>(rank) - 1) ||
                    (op == OpType::REPEAT && node.child_ids.size() == 3 && constAxis(node.child_ids[2], axis) &&
                     axis == static_cast<int32_t>(rank) - 1);
                if (!allowed || !leadingMatch(node.getShape()))
                {
                    ok = false;
                    break;
                }
            }
            result.rowLocal = ok && isRow[patternRoot];
        }
    }
    return cache.emplace(opName, std::move(result)).first->second;
}

// contiguous(slice(src)) where src has a row-local fused enode and the slice
// keeps the whole last axis: the same fused kernel on the row slices of its
// row inputs (broadcast inputs unchanged). Lets fused kernels (norms, gelu,
// exp, ...) run on the rows a bucket updates.
struct SlicePushDownFused : public Rule
{
    std::unique_ptr<Rule> clone() const override
    {
        return std::make_unique<SlicePushDownFused>(*this);
    }

    struct MatchKey
    {
        uint32_t contigClass;
        uint32_t sliceIdx;
        bool operator==(const MatchKey &o) const
        {
            return contigClass == o.contigClass && sliceIdx == o.sliceIdx;
        }
    };
    struct MatchKeyHash
    {
        std::uint64_t operator()(const MatchKey &k) const
        {
            return (static_cast<uint64_t>(k.contigClass) * 0x9E3779B97F4A7C15ULL) ^
                   (static_cast<uint64_t>(k.sliceIdx) << 21);
        }
    };
    std::unordered_set<MatchKey, MatchKeyHash> visited;

    std::string name() const override
    {
        return "SlicePushDownFused";
    }

    bool canMatchOp(OpType op) const override
    {
        return op == OpType::CONTIGUOUS;
    }

    static bool candidate(const EGraph &egraph, ENodeId srcIdx)
    {
        const ENode &op = egraph.getENode(srcIdx);
        return op.getOpType() == OpType::FUSED && op.getKernelId().value != 0 &&
               KernelRegistry::get().hasKernel(op.getKernelId()) && fusedRowLocality(op.getOpName()).rowLocal;
    }

    void clearVisited() override
    {
        visited.clear();
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (eNodeIdx < ctx.base_enode_watermark && ctx.sweep_filter &&
            eNodeIdx < ctx.sweep_filter->size() && (*ctx.sweep_filter)[eNodeIdx] == 0)
            return false;
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::CONTIGUOUS || enode.getChildren().empty())
            return false;
        const uint32_t contig_class = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx})).value;
        const EClassId sliceClass = egraph.findConst(enode.getChildren()[0]);
        for (ENodeId sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            if (visited.count(MatchKey{contig_class, sliceIdx.value}))
                continue;
            const ENode &sliceNode = egraph.getENode(sliceIdx);
            if (sliceNode.getOpType() != OpType::SLICE || sliceNode.getChildren().size() != 4)
                continue;
            const EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            const EClass &src = egraph.getEClass(srcClass);
            bool has_fused = false;
            for (ENodeId srcIdx : src.enodes)
            {
                if (egraph.getENode(srcIdx).getOpType() == OpType::FUSED && candidate(egraph, srcIdx))
                {
                    has_fused = true;
                    break;
                }
            }
            if (!has_fused)
                continue;

            Region box;
            if (!SlicePushDownStructural::sliceBox(egraph, sliceNode, src.shape, box))
                continue;
            const size_t rank = src.shape.size();
            if (rank < 2 || box.region.back().start != 0 || box.region.back().stop != src.shape.back())
                continue;
            const bool scatter_valid = scatterFormValid(ctx, src, box);
            const bool has_scatter = scatter_valid && hasScatter(egraph, srcClass);
            if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                sliceIdx.value < ctx.base_enode_watermark && (!scatter_valid || has_scatter) &&
                (src.enodes.empty() || src.enodes.back().value < ctx.base_enode_watermark))
                continue;
            for (ENodeId srcIdx : src.enodes)
            {
                if (!candidate(egraph, srcIdx))
                    continue;
                if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                    sliceIdx.value < ctx.base_enode_watermark && srcIdx.value < ctx.base_enode_watermark &&
                    (!scatter_valid || has_scatter))
                    continue;
                return true;
            }
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const uint32_t contig_class = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx})).value;
        const ENode contigNode = egraph.getENode(ENodeId{eNodeIdx});
        const EClassId sliceClass = egraph.findConst(contigNode.getChildren()[0]);
        const size_t numSlices = egraph.getEClass(sliceClass).enodes.size();
        for (size_t sl_i = 0; sl_i < numSlices; ++sl_i)
        {
            ENodeId sliceIdx = egraph.getEClass(sliceClass).enodes[sl_i];
            const ENode sliceNode = egraph.getENode(sliceIdx);
            if (sliceNode.getOpType() != OpType::SLICE || sliceNode.getChildren().size() != 4)
                continue;
            EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            bool has_fused = false;
            {
                const EClass &src = egraph.getEClass(srcClass);
                for (ENodeId s : src.enodes)
                {
                    if (egraph.getENode(s).getOpType() == OpType::FUSED && candidate(egraph, s))
                    {
                        has_fused = true;
                        break;
                    }
                }
            }
            if (!has_fused)
                continue;

            const std::vector<uint32_t> srcShape = egraph.getEClass(srcClass).shape;
            Region box;
            if (!SlicePushDownStructural::sliceBox(egraph, sliceNode, srcShape, box))
                continue;
            const size_t rank = srcShape.size();
            if (rank < 2 || box.region.back().start != 0 || box.region.back().stop != srcShape.back())
                continue;
            const bool scatter_valid = scatterFormValid(ctx, egraph.getEClass(srcClass), box);
            const bool has_scatter = scatter_valid && hasScatter(egraph, srcClass);
            if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                sliceIdx.value < ctx.base_enode_watermark && (!scatter_valid || has_scatter) &&
                (egraph.getEClass(srcClass).enodes.empty() || egraph.getEClass(srcClass).enodes.back().value < ctx.base_enode_watermark))
                continue;

            if (!visited.insert(MatchKey{contig_class, sliceIdx.value}).second)
                continue;

            EClassId startsConstId{}, endsConstId{}, stepsConstId{}, shapeId{};
            bool consts_prepared = false;

            const size_t numSrc = egraph.getEClass(srcClass).enodes.size();
            for (size_t s_i = 0; s_i < numSrc; ++s_i)
            {
                ENodeId srcIdx = egraph.getEClass(srcClass).enodes[s_i];
                if (!candidate(egraph, srcIdx))
                    continue;
                if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark &&
                    sliceIdx.value < ctx.base_enode_watermark && srcIdx.value < ctx.base_enode_watermark &&
                    (!scatter_valid || has_scatter))
                    continue;
                const ENode op = egraph.getENode(srcIdx);
                const EClass src = egraph.getEClass(egraph.findConst(srcClass));
                const FusedRowLocality &loc = fusedRowLocality(op.getOpName());
                const auto &children = op.getChildren();
                if (children.size() < loc.rowInput.size())
                    continue;
                std::vector<EClassId> newChildren;
                std::vector<TensorNode> inputNodes;
                bool valid = true;
                for (size_t i = 0; i < children.size() && valid; ++i)
                {
                    EClassId child = egraph.findConst(children[i]);
                    const EClass childCls = egraph.getEClass(child);
                    const bool row = i < loc.rowInput.size() && loc.rowInput[i];
                    if (row)
                    {
                        if (childCls.shape.size() != rank)
                        {
                            valid = false;
                            break;
                        }
                        for (size_t d = 0; d + 1 < rank; ++d)
                            valid = valid && childCls.shape[d] == src.shape[d];
                        if (!valid)
                            break;
                        Region childBox = box;
                        childBox.region.back() = {0, childCls.shape.back()};
                        child = egraph.findConst(SlicePushDownStructural::sliceOf(egraph, child, childBox, ctx.slice_memo));
                    }
                    newChildren.push_back(child);
                    const EClass c = egraph.getEClass(child);
                    TensorNode in;
                    in.opType = OpType::INPUT;
                    in.dtype = c.dtype;
                    in.setShape(c.shape);
                    in.strides = c.strides;
                    inputNodes.push_back(in);
                }
                if (!valid)
                    continue;
                const std::vector<uint32_t> outShape = SlicePushDownStructural::boxShape(box);
                TensorNode outNode;
                outNode.opType = OpType::FUSED;
                outNode.opName = op.getOpName();
                outNode.dtype = src.dtype;
                outNode.setShape(outShape);
                outNode.strides = calcContiguousStrides(outShape);
                const KernelEntry &kernel = KernelRegistry::get().getKernel(op.getKernelId());
                if (!kernel.matches(inputNodes, outNode, op.getMemSpace(), {}, {}, false, true, true, true))
                    continue;
                const EClassId restricted =
                    egraph.addEClass(outShape, outNode.strides, src.dtype, op.getMemSpace());
                egraph.addENode(restricted, ENode(op.getKernelId(), OpType::FUSED, op.getOpName(), newChildren,
                                                  outShape, outNode.strides, src.dtype, op.getMemSpace(),
                                                  op.getEngines(), "", 0, op.getDebugOrigin()));
                srcClass = egraph.findConst(srcClass);
                mergeRestricted(egraph, egraph.getENodeEClass(ENodeId{eNodeIdx}), restricted);
                if (scatter_valid && !hasScatter(egraph, srcClass))
                {
                    srcClass = egraph.findConst(srcClass);
                    const EClass srcAfter = egraph.getEClass(srcClass);
                    if (!isContiguous(srcAfter) || !scatterFormValid(ctx, srcAfter, box))
                        continue;
                    if (!consts_prepared)
                    {
                        std::vector<int32_t> starts, ends, steps;
                        starts.reserve(box.region.size());
                        ends.reserve(box.region.size());
                        steps.reserve(box.region.size());
                        for (const Dim &d : box.region)
                        {
                            starts.push_back((int32_t)d.start);
                            ends.push_back((int32_t)d.stop);
                            steps.push_back(1);
                        }
                        startsConstId = egraph.addIntConst(starts);
                        endsConstId = egraph.addIntConst(ends);
                        stepsConstId = egraph.addIntConst(steps);
                        shapeId = egraph.addIntConst(std::vector<int32_t>(srcAfter.shape.begin(), srcAfter.shape.end()));
                        consts_prepared = true;
                    }
                    const EClassId scatterClass = addOpToEGraph(egraph, OpType::SCATTER,
                        {restricted, startsConstId, endsConstId, stepsConstId, shapeId},
                        srcAfter.shape, srcAfter.strides, srcAfter.dtype, srcAfter.mem_space, EClassId(), SourceLocation::current(),
                        "", scatterFormTag(egraph, srcClass));
                    egraph.merge(srcClass, scatterClass);
                    srcClass = egraph.findConst(srcClass);
                }
                if (!scatter_valid || hasScatter(egraph, srcClass))
                    break;
            }
        }
    }
};

// Splits up dot into 2 smaller dots. Can reduce peak memory usage, and combined with FusionRule naturally discovers
// tensor parallism across engines.
struct DotSplitRule : public Rule
{
    std::unique_ptr<Rule> clone() const override
    {
        return std::make_unique<DotSplitRule>(*this);
    }

    uint32_t splitThreshold = 32768; // Only split dimensions >= splitThreshold
    std::vector<uint8_t> visited;

    std::string name() const override
    {
        return "DotSplitRule";
    }

    bool canMatchOp(OpType op) const override
    {
        return op == OpType::DOT;
    }

    void clearVisited() override
    {
        visited.clear();
    }

    void markVisited(uint32_t eNodeIdx)
    {
        if (eNodeIdx >= visited.size())
            visited.resize(eNodeIdx + 1, 0);
        visited[eNodeIdx] = 1;
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (eNodeIdx < ctx.base_enode_watermark)
            return false;
        if (eNodeIdx < visited.size() && visited[eNodeIdx])
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getNumENodes())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::DOT)
            return false;

        EClassId aClass = egraph.findConst(enode.getChildren()[0]);
        EClassId bClass = egraph.findConst(enode.getChildren()[1]);
        const EClass &aCls = egraph.getEClass(aClass);
        const EClass &bCls = egraph.getEClass(bClass);

        // Rank 3: A [B, M, K], B [B, K, N]
        if (aCls.shape.size() == 3 && bCls.shape.size() == 3)
        {
            uint32_t K = aCls.shape[2];
            uint32_t N = bCls.shape[2];
            // Match if Column dimension N or Reduction dimension K is large
            if ((N >= splitThreshold && N % 2 == 0) || (K >= splitThreshold && K % 2 == 0))
                return true;
        }

        markVisited(eNodeIdx);
        return false;
    }

    // -----------------------------------------------------------------
    // Strategy A: Column-Parallel Split (Split N -> N1, N2)
    // A [B, M, K] * B1 [B, K, N1] -> C1 [B, M, N1]
    // A [B, M, K] * B2 [B, K, N2] -> C2 [B, M, N2]
    // C = CONCAT([C1, C2], axis=2)
    // -----------------------------------------------------------------
    void applyStrategyA(EGraph &egraph, EClassId e_class_id, const ENode dotNode, EClassId aClass, EClassId bClass,
                        const EClass aCls, const EClass bCls)
    {
        uint32_t B = aCls.shape[0];
        uint32_t M = aCls.shape[1];
        uint32_t K = aCls.shape[2];
        uint32_t N = bCls.shape[2];

        uint32_t N1 = N / 2;
        uint32_t N2 = N - N1;

        // B1 Slice: [0..B, 0..K, 0..N1]
        EClassId st1 = egraph.addIntConst({0, 0, 0});
        EClassId en1 = egraph.addIntConst({(int32_t)B, (int32_t)K, (int32_t)N1});
        EClassId step = egraph.addIntConst({1, 1, 1});

        EClassId b1 = addOpToEGraph(egraph, OpType::SLICE, {bClass, st1, en1, step}, {B, K, N1},
                                    {bCls.strides[0], bCls.strides[1], bCls.strides[2]}, bCls.dtype, bCls.mem_space);

        // B2 Slice: [0..B, 0..K, N1..N]
        EClassId st2 = egraph.addIntConst({0, 0, (int32_t)N1});
        EClassId en2 = egraph.addIntConst({(int32_t)B, (int32_t)K, (int32_t)N});

        EClassId b2 = addOpToEGraph(egraph, OpType::SLICE, {bClass, st2, en2, step}, {B, K, N2},
                                    {bCls.strides[0], bCls.strides[1], bCls.strides[2]}, bCls.dtype, bCls.mem_space);

        EClassId b1_contig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {b1}, {B, K, N1},
                                           calcContiguousStrides({B, K, N1}), bCls.dtype, bCls.mem_space);
        EClassId b2_contig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {b2}, {B, K, N2},
                                           calcContiguousStrides({B, K, N2}), bCls.dtype, bCls.mem_space);

        // C1 = A * B1, C2 = A * B2
        EClassId c1 = addOpToEGraph(egraph, OpType::DOT, {aClass, b1_contig}, {B, M, N1},
                                    calcContiguousStrides({B, M, N1}), dotNode.getDType(), dotNode.getMemSpace());
        EClassId c2 = addOpToEGraph(egraph, OpType::DOT, {aClass, b2_contig}, {B, M, N2},
                                    calcContiguousStrides({B, M, N2}), dotNode.getDType(), dotNode.getMemSpace());

        // C = CONCAT([C1, C2], axis=2)
        EClassId axis2 = egraph.addIntConst({2});
        EClassId c_concat = addOpToEGraph(egraph, OpType::CONCAT, {axis2, c1, c2}, {B, M, N},
                                          calcContiguousStrides({B, M, N}), dotNode.getDType(), dotNode.getMemSpace());

        egraph.merge(e_class_id, c_concat);
    }

    // -----------------------------------------------------------------
    // Strategy B: Row-Parallel Split (Split K -> K1, K2)
    // A1 [B, M, K1] * B1 [B, K1, N] -> C1 [B, M, N]
    // A2 [B, M, K2] * B2 [B, K2, N] -> C2 [B, M, N]
    // C = ADD(C1, C2)
    // -----------------------------------------------------------------
    void applyStrategyB(EGraph &egraph, EClassId e_class_id, const ENode dotNode, EClassId aClass, EClassId bClass,
                        const EClass aCls, const EClass bCls)
    {
        uint32_t B = aCls.shape[0];
        uint32_t M = aCls.shape[1];
        uint32_t K = aCls.shape[2];
        uint32_t N = bCls.shape[2];

        uint32_t K1 = K / 2;
        uint32_t K2 = K - K1;

        EClassId step = egraph.addIntConst({1, 1, 1});

        // A1 Slice [0..B, 0..M, 0..K1], A2 Slice [0..B, 0..M, K1..K]
        EClassId a_st1 = egraph.addIntConst({0, 0, 0});
        EClassId a_en1 = egraph.addIntConst({(int32_t)B, (int32_t)M, (int32_t)K1});
        EClassId a_st2 = egraph.addIntConst({0, 0, (int32_t)K1});
        EClassId a_en2 = egraph.addIntConst({(int32_t)B, (int32_t)M, (int32_t)K});

        EClassId a1 = addOpToEGraph(egraph, OpType::SLICE, {aClass, a_st1, a_en1, step}, {B, M, K1},
                                    {aCls.strides[0], aCls.strides[1], aCls.strides[2]}, aCls.dtype, aCls.mem_space);
        EClassId a2 = addOpToEGraph(egraph, OpType::SLICE, {aClass, a_st2, a_en2, step}, {B, M, K2},
                                    {aCls.strides[0], aCls.strides[1], aCls.strides[2]}, aCls.dtype, aCls.mem_space);

        // B1 Slice [0..B, 0..K1, 0..N], B2 Slice [0..B, K1..K, 0..N]
        EClassId b_st1 = egraph.addIntConst({0, 0, 0});
        EClassId b_en1 = egraph.addIntConst({(int32_t)B, (int32_t)K1, (int32_t)N});
        EClassId b_st2 = egraph.addIntConst({0, (int32_t)K1, 0});
        EClassId b_en2 = egraph.addIntConst({(int32_t)B, (int32_t)K, (int32_t)N});

        EClassId b1 = addOpToEGraph(egraph, OpType::SLICE, {bClass, b_st1, b_en1, step}, {B, K1, N},
                                    {bCls.strides[0], bCls.strides[1], bCls.strides[2]}, bCls.dtype, bCls.mem_space);
        EClassId b2 = addOpToEGraph(egraph, OpType::SLICE, {bClass, b_st2, b_en2, step}, {B, K2, N},
                                    {bCls.strides[0], bCls.strides[1], bCls.strides[2]}, bCls.dtype, bCls.mem_space);

        EClassId a1_contig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {a1}, {B, M, K1},
                                           calcContiguousStrides({B, M, K1}), aCls.dtype, aCls.mem_space);
        EClassId a2_contig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {a2}, {B, M, K2},
                                           calcContiguousStrides({B, M, K2}), aCls.dtype, aCls.mem_space);
        EClassId b1_contig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {b1}, {B, K1, N},
                                           calcContiguousStrides({B, K1, N}), bCls.dtype, bCls.mem_space);
        EClassId b2_contig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {b2}, {B, K2, N},
                                           calcContiguousStrides({B, K2, N}), bCls.dtype, bCls.mem_space);

        // C1 = A1 * B1, C2 = A2 * B2
        EClassId c1 = addOpToEGraph(egraph, OpType::DOT, {a1_contig, b1_contig}, {B, M, N},
                                    calcContiguousStrides({B, M, N}), dotNode.getDType(), dotNode.getMemSpace());
        EClassId c2 = addOpToEGraph(egraph, OpType::DOT, {a2_contig, b2_contig}, {B, M, N},
                                    calcContiguousStrides({B, M, N}), dotNode.getDType(), dotNode.getMemSpace());

        // C = ADD(C1, C2)
        EClassId c_add = addOpToEGraph(egraph, OpType::ADD, {c1, c2}, {B, M, N}, calcContiguousStrides({B, M, N}),
                                       dotNode.getDType(), dotNode.getMemSpace());

        egraph.merge(e_class_id, c_add);
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        markVisited(eNodeIdx);

        const ENode dotNode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        EClassId aClass = egraph.findConst(dotNode.getChildren()[0]);
        EClassId bClass = egraph.findConst(dotNode.getChildren()[1]);
        const EClass aCls = egraph.getEClass(aClass);
        const EClass bCls = egraph.getEClass(bClass);

        if (aCls.shape.size() != 3 || bCls.shape.size() != 3)
            return;

        if (!isContiguous(aCls))
        {
            aClass = addOpToEGraph(egraph, OpType::CONTIGUOUS, {aClass}, aCls.shape, calcContiguousStrides(aCls.shape),
                                   aCls.dtype, aCls.mem_space);
        }
        if (!isContiguous(bCls))
        {
            bClass = addOpToEGraph(egraph, OpType::CONTIGUOUS, {bClass}, bCls.shape, calcContiguousStrides(bCls.shape),
                                   bCls.dtype, bCls.mem_space);
        }

        uint32_t K = aCls.shape[2];
        uint32_t N = bCls.shape[2];

        if (N >= splitThreshold)
        {
            applyStrategyA(egraph, e_class_id, dotNode, aClass, bClass, aCls, bCls);
        }

        if (K >= splitThreshold)
        {
            applyStrategyB(egraph, e_class_id, dotNode, aClass, bClass, aCls, bCls);
        }
    }
};

// Remove unneeded contiguous ops. op(contiguous(x)) -> op(x)
struct RemoveContiguous : public Rule
{
    std::unique_ptr<Rule> clone() const override
    {
        return std::make_unique<RemoveContiguous>(*this);
    }

    std::vector<uint8_t> visited;

    std::string name() const override
    {
        return "RemoveContiguous";
    }

    bool canMatchOp(OpType op) const override
    {
        return op != OpType::CONTIGUOUS && op != OpType::INPUT && op != OpType::CACHE;
    }

    void clearVisited() override
    {
        visited.clear();
    }

    void markVisited(uint32_t eNodeIdx)
    {
        if (eNodeIdx >= visited.size())
            visited.resize(eNodeIdx + 1, 0);
        visited[eNodeIdx] = 1;
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (eNodeIdx < ctx.base_enode_watermark && ctx.sweep_filter &&
            eNodeIdx < ctx.sweep_filter->size() && (*ctx.sweep_filter)[eNodeIdx] == 0)
            return false;
        if (eNodeIdx < visited.size() && visited[eNodeIdx])
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getNumENodes())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});

        if (enode.getOpType() == OpType::CONTIGUOUS || enode.getOpType() == OpType::INPUT ||
            enode.getOpType() == OpType::CACHE)
            return false;

        const bool is_base = (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark);
        if (is_base)
        {
            bool has_new_child = false;
            for (EClassId child : enode.getChildren())
            {
                const EClass &cls = egraph.getEClass(egraph.findConst(child));
                if (!cls.enodes.empty() && cls.enodes.back().value >= ctx.base_enode_watermark)
                {
                    has_new_child = true;
                    break;
                }
            }
            if (!has_new_child)
                return false;
        }

        // Check if the kernel explicitly requires contiguous inputs for any child
        if (!KernelRegistry::get().hasKernel(enode.getKernelId()))
            return false;
        const KernelEntry &kernel = KernelRegistry::get().getKernel(enode.getKernelId());

        if (kernel.is_view)
        {
            markVisited(eNodeIdx);
            return false; // view kernels output strides depend on input strides so will break things when merge eclass
        }

        const auto &children = enode.getChildren();
        for (uint64_t i = 0; i < children.size(); ++i)
        {
            // A kernel that requires a contiguous input here keeps the
            // CONTIGUOUS unless the unwrapped value is already contiguous.
            const bool needsContig = requiresContiguousAt(kernel, i);

            EClassId childClsId = egraph.findConst(children[i]);
            const EClass &childCls = egraph.getEClass(childClsId);
            for (ENodeId cEnodeId : childCls.enodes)
            {
                const ENode &cEnode = egraph.getENode(cEnodeId);
                if (cEnode.getOpType() == OpType::CONTIGUOUS && !cEnode.getChildren().empty())
                {
                    EClassId unwrappedChildId = egraph.findConst(cEnode.getChildren()[0]);
                    if (unwrappedChildId != childClsId &&
                        (!needsContig || isContiguous(egraph.getEClass(unwrappedChildId))))
                    {
                        if (!is_base || cEnodeId.value >= ctx.base_enode_watermark)
                            return true;
                    }
                }
            }
        }

        return false;
    }

    static bool requiresContiguousAt(const KernelEntry &kernel, uint64_t i)
    {
        if (kernel.requiresContiguous.empty())
            return false;
        return kernel.requiresContiguous[std::min(i, static_cast<uint64_t>(kernel.requiresContiguous.size() - 1))];
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        markVisited(eNodeIdx);
        EGraph &egraph = ctx.egraph;

        const ENode enode = egraph.getENode(ENodeId{eNodeIdx});
        const bool is_base = (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark);
        if (is_base)
        {
            bool has_new_child = false;
            for (EClassId child : enode.getChildren())
            {
                const EClass &cls = egraph.getEClass(egraph.findConst(child));
                if (!cls.enodes.empty() && cls.enodes.back().value >= ctx.base_enode_watermark)
                {
                    has_new_child = true;
                    break;
                }
            }
            if (!has_new_child)
                return;
        }
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});
        const auto children = enode.getChildren();
        const KernelEntry &kernel = KernelRegistry::get().getKernel(enode.getKernelId());

        bool has_unwrapped_candidate = false;
        struct CandidateChild
        {
            EClassId cid;
            bool is_new;
        };
        std::vector<std::vector<CandidateChild>> candidateChildrenPerPos(children.size());

        for (uint64_t i = 0; i < children.size(); ++i)
        {
            EClassId childClsId = egraph.findConst(children[i]);
            candidateChildrenPerPos[i].push_back({childClsId, false});
            const bool needsContig = requiresContiguousAt(kernel, i);

            const EClass &childCls = egraph.getEClass(childClsId);
            for (ENodeId cEnodeId : childCls.enodes)
            {
                const ENode &cEnode = egraph.getENode(cEnodeId);
                if (cEnode.getOpType() == OpType::CONTIGUOUS && !cEnode.getChildren().empty())
                {
                    EClassId unwrappedChildId = egraph.findConst(cEnode.getChildren()[0]);
                    if (unwrappedChildId != childClsId &&
                        (!needsContig || isContiguous(egraph.getEClass(unwrappedChildId))))
                    {
                        bool already_present = false;
                        for (const auto &existing : candidateChildrenPerPos[i])
                        {
                            if (existing.cid == unwrappedChildId)
                            {
                                already_present = true;
                                break;
                            }
                        }
                        if (!already_present)
                        {
                            const bool is_new = (cEnodeId.value >= ctx.base_enode_watermark);
                            candidateChildrenPerPos[i].push_back({unwrappedChildId, is_new});
                            if (!is_base || is_new)
                                has_unwrapped_candidate = true;
                        }
                    }
                }
            }
        }

        if (!has_unwrapped_candidate)
            return;

        std::vector<std::vector<EClassId>> childCombinations;
        std::vector<EClassId> currentCombination(children.size());

        auto generateCombos = [&](auto &self, uint64_t pos, bool hasUnwrapped, bool hasNew) -> void {
            if (pos == children.size())
            {
                if (hasUnwrapped && (!is_base || hasNew))
                {
                    childCombinations.push_back(currentCombination);
                }
                return;
            }
            for (uint64_t cIdx = 0; cIdx < candidateChildrenPerPos[pos].size(); ++cIdx)
            {
                currentCombination[pos] = candidateChildrenPerPos[pos][cIdx].cid;
                self(self, pos + 1, hasUnwrapped || (cIdx > 0), hasNew || candidateChildrenPerPos[pos][cIdx].is_new);
            }
        };

        generateCombos(generateCombos, 0, false, false);

        const EClass outCls = egraph.getEClass(egraph.findConst(e_class_id));
        TensorNode outNode;
        outNode.opType = enode.getOpType();
        outNode.opName = enode.getOpName();
        outNode.dtype = outCls.dtype;
        outNode.setShape(outCls.shape);
        outNode.strides = outCls.strides;

        std::vector<TensorNode> inNodes;
        std::vector<MemSpace> inMemSpaces;
        for (const auto &newChildren : childCombinations)
        {
            inNodes.clear();
            inMemSpaces.clear();
            for (EClassId cid : newChildren)
            {
                const EClass cCls = egraph.getEClass(egraph.findConst(cid));
                TensorNode inNode;
                inNode.opType = OpType::INPUT;
                inNode.dtype = cCls.dtype;
                inNode.setShape(cCls.shape);
                inNode.strides = cCls.strides;
                inNodes.push_back(inNode);
                inMemSpaces.push_back(cCls.mem_space);
            }

            if (kernel.matches(inNodes, outNode, outCls.mem_space, inMemSpaces, enode.getEngines()))
            {
                ENode newENode(kernel.uid, enode.getOpType(), enode.getOpName(), newChildren, enode.getShape(),
                               enode.getStrides(), enode.getDType(), enode.getMemSpace(), enode.getEngines(), enode.getContentHash(), 0,
                               enode.getDebugOrigin());
                const size_t beforeSize = egraph.getNumENodes();
                egraph.addENode(e_class_id, newENode);
                for (size_t added_idx = beforeSize; added_idx < egraph.getNumENodes(); ++added_idx)
                {
                    markVisited(static_cast<uint32_t>(added_idx));
                }
            }
        }
    }
};

// Remove chains of COPY_TO operations where the start and end MemSpaces match.
// Matches the consumer node and rewrites its children to point directly to the
// earlier EClass in the same MemSpace, bypassing intermediate COPY_TO nodes.
struct RemoveCopyChains : public Rule
{
    std::unique_ptr<Rule> clone() const override
    {
        auto copy = std::make_unique<RemoveCopyChains>(*this);
        copy->scanned_enodes = 0;
        copy->copy_enodes.clear();
        return copy;
    }

    std::vector<uint8_t> visited;
    size_t scanned_enodes = 0;
    std::vector<uint32_t> copy_enodes;

    void markVisited(uint32_t eNodeIdx)
    {
        if (eNodeIdx >= visited.size())
            visited.resize(eNodeIdx + 1, 0);
        visited[eNodeIdx] = 1;
    }

    void refreshCopyIndex(const EGraph &egraph)
    {
        const size_t total_enodes = egraph.getNumENodes();
        if (total_enodes < scanned_enodes)
        {
            scanned_enodes = 0;
            copy_enodes.clear();
        }
        for (; scanned_enodes < total_enodes; ++scanned_enodes)
            if (egraph.getENode(ENodeId{static_cast<uint32_t>(scanned_enodes)}).getOpType() == OpType::COPY_TO)
                copy_enodes.push_back(static_cast<uint32_t>(scanned_enodes));
    }

    std::string name() const override
    {
        return "RemoveCopyChains";
    }

    bool canMatchOp(OpType op) const override
    {
        return op != OpType::INPUT && op != OpType::CACHE;
    }

    static bool hasCopyToNode(const EClass &cls, const EGraph &egraph)
    {
        for (ENodeId enodeId : cls.enodes)
        {
            if (egraph.getENode(enodeId).getOpType() == OpType::COPY_TO)
                return true;
        }
        return false;
    }

    static std::vector<EClassId> findAllCopyChainOrigins(EClassId startClassId, const EGraph &egraph)
    {
        EClassId startCanon = egraph.findConst(startClassId);
        MemSpace targetMemSpace = egraph.getEClass(startCanon).mem_space;

        std::vector<EClassId> origins;
        std::unordered_set<EClassId> visited = {startCanon};
        std::vector<EClassId> queue = {startCanon};

        for (size_t qi = 0; qi < queue.size(); ++qi)
        {
            EClassId currClass = queue[qi];
            const EClass &cls = egraph.getEClass(currClass);

            for (ENodeId enodeId : cls.enodes)
            {
                const ENode &enode = egraph.getENode(enodeId);
                if (enode.getOpType() == OpType::COPY_TO && !enode.getChildren().empty())
                {
                    EClassId nextClass = egraph.findConst(enode.getChildren()[0]);
                    if (visited.insert(nextClass).second)
                    {
                        if (egraph.getEClass(nextClass).mem_space == targetMemSpace)
                        {
                            origins.push_back(nextClass);
                        }
                        queue.push_back(nextClass);
                    }
                }
            }
        }
        return origins;
    }

    static bool hasCopyChainOrigin(EClassId startClassId, const EGraph &egraph)
    {
        EClassId startCanon = egraph.findConst(startClassId);
        MemSpace targetMemSpace = egraph.getEClass(startCanon).mem_space;

        constexpr size_t MAX_STACK = 32;
        std::array<EClassId, MAX_STACK> queue;
        size_t queue_size = 0;
        queue[queue_size++] = startCanon;

        for (size_t qi = 0; qi < queue_size; ++qi)
        {
            EClassId currClass = queue[qi];
            const EClass &cls = egraph.getEClass(currClass);

            for (ENodeId enodeId : cls.enodes)
            {
                const ENode &enode = egraph.getENode(enodeId);
                if (enode.getOpType() == OpType::COPY_TO && !enode.getChildren().empty())
                {
                    EClassId nextClass = egraph.findConst(enode.getChildren()[0]);
                    bool already_visited = false;
                    for (size_t vi = 0; vi < queue_size; ++vi)
                    {
                        if (queue[vi] == nextClass)
                        {
                            already_visited = true;
                            break;
                        }
                    }
                    if (!already_visited)
                    {
                        if (egraph.getEClass(nextClass).mem_space == targetMemSpace)
                        {
                            return true;
                        }
                        if (queue_size < MAX_STACK)
                        {
                            queue[queue_size++] = nextClass;
                        }
                        else
                        {
                            std::vector<EClassId> origins = findAllCopyChainOrigins(startClassId, egraph);
                            return !origins.empty();
                        }
                    }
                }
            }
        }
        return false;
    }

    void clearVisited() override
    {
        visited.clear();
        copy_enodes.clear();
        scanned_enodes = 0;
    }

    bool prepareSweep(RuleCtx &ctx) override
    {
        refreshCopyIndex(ctx.egraph);
        return !copy_enodes.empty();
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (copy_enodes.empty())
            return false;

        if (eNodeIdx < ctx.base_enode_watermark && ctx.sweep_filter &&
            eNodeIdx < ctx.sweep_filter->size() && (*ctx.sweep_filter)[eNodeIdx] == 0)
            return false;
        if (eNodeIdx < visited.size() && visited[eNodeIdx])
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getNumENodes())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});

        if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
            return false;

        if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark)
        {
            bool has_new_child = false;
            for (EClassId child : enode.getChildren())
            {
                const EClass &cls = egraph.getEClass(egraph.findConst(child));
                if (!cls.enodes.empty() && cls.enodes.back().value >= ctx.base_enode_watermark)
                {
                    has_new_child = true;
                    break;
                }
            }
            if (!has_new_child)
                return false;
        }

        const auto &children = enode.getChildren();
        for (uint64_t i = 0; i < children.size(); ++i)
        {
            EClassId childCanon = egraph.findConst(children[i]);
            const EClass &cCls = egraph.getEClass(childCanon);
            if (!hasCopyToNode(cCls, egraph))
                continue;
            if (hasCopyChainOrigin(childCanon, egraph))
            {
                return true;
            }
        }

        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        markVisited(eNodeIdx);

        const ENode enode = egraph.getENode(ENodeId{eNodeIdx});
        if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark)
        {
            bool has_new_child = false;
            for (EClassId child : enode.getChildren())
            {
                const EClass &cls = egraph.getEClass(egraph.findConst(child));
                if (!cls.enodes.empty() && cls.enodes.back().value >= ctx.base_enode_watermark)
                {
                    has_new_child = true;
                    break;
                }
            }
            if (!has_new_child)
                return;
        }
        EClassId eClassId = egraph.getENodeEClass(ENodeId{eNodeIdx});

        std::vector<std::vector<EClassId>> child_origin_options;
        child_origin_options.reserve(enode.getChildren().size());
        bool any_has_origin = false;

        for (uint64_t i = 0; i < enode.getChildren().size(); ++i)
        {
            EClassId childCanon = egraph.findConst(enode.getChildren()[i]);
            std::vector<EClassId> origins;
            bool checkCopy = false;
            {
                const EClass &cCls = egraph.getEClass(childCanon);
                checkCopy = hasCopyToNode(cCls, egraph);
            }
            if (checkCopy)
                origins = findAllCopyChainOrigins(childCanon, egraph);
            if (!origins.empty())
            {
                any_has_origin = true;
                child_origin_options.push_back(std::move(origins));
            }
            else
            {
                child_origin_options.push_back({childCanon});
            }
        }

        if (!any_has_origin)
            return;

        std::vector<std::vector<EClassId>> combos = {{}};
        for (const auto &options : child_origin_options)
        {
            std::vector<std::vector<EClassId>> next_combos;
            for (const auto &combo : combos)
            {
                for (EClassId opt : options)
                {
                    auto new_combo = combo;
                    new_combo.push_back(opt);
                    next_combos.push_back(std::move(new_combo));
                    if (next_combos.size() >= 16)
                        break;
                }
                if (next_combos.size() >= 16)
                    break;
            }
            combos = std::move(next_combos);
        }

        for (const auto &newChildren : combos)
        {
            if (newChildren == enode.getChildren())
                continue;

            if (KernelRegistry::get().hasKernel(enode.getKernelId()))
            {
                const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
                bool contig_ok = true;
                if (!kernel.requiresContiguous.empty())
                {
                    for (size_t c_idx = 0; c_idx < newChildren.size(); ++c_idx)
                    {
                        size_t rule_idx = std::min(c_idx, kernel.requiresContiguous.size() - 1);
                        if (kernel.requiresContiguous[rule_idx] &&
                            !isContiguous(egraph.getEClass(egraph.findConst(newChildren[c_idx]))))
                        {
                            contig_ok = false;
                            break;
                        }
                    }
                }
                if (!contig_ok)
                    continue;
            }

            ENode newENode(enode.getKernelId(), enode.getOpType(), enode.getOpName(), newChildren, enode.getShape(),
                           enode.getStrides(), enode.getDType(), enode.getMemSpace(), enode.getEngines(),
                           enode.getContentHash(), 0, enode.getDebugOrigin());
            const size_t beforeSize = egraph.getNumENodes();
            egraph.addENode(eClassId, newENode);
            for (size_t added_idx = beforeSize; added_idx < egraph.getNumENodes(); ++added_idx)
            {
                markVisited(static_cast<uint32_t>(added_idx));
            }
        }
    }
};

struct ConsumerWeightReuseRule : public Rule
{
    std::unique_ptr<Rule> clone() const override
    {
        auto copy = std::make_unique<ConsumerWeightReuseRule>(*this);
        // The COPY_TO index belongs to the egraph it was built on.
        copy->scanned_enodes = 0;
        copy->copy_enodes.clear();
        return copy;
    }

    std::vector<uint8_t> visited_enodes;

    void markVisited(uint32_t eNodeIdx)
    {
        if (eNodeIdx >= visited_enodes.size())
            visited_enodes.resize(eNodeIdx + 1, 0);
        visited_enodes[eNodeIdx] = 1;
    }

    // Every COPY_TO enode seen so far (enodes are append-only): only classes
    // holding one can be a storage weight, so duplicate lookups scan these
    // instead of every class.
    size_t scanned_enodes = 0;
    std::vector<uint32_t> copy_enodes;

    void refreshCopyIndex(const EGraph &egraph)
    {
        const size_t total_enodes = egraph.getNumENodes();
        if (total_enodes < scanned_enodes)
        {
            scanned_enodes = 0;
            copy_enodes.clear();
        }
        for (; scanned_enodes < total_enodes; ++scanned_enodes)
            if (egraph.getENode(ENodeId{static_cast<uint32_t>(scanned_enodes)}).getOpType() == OpType::COPY_TO)
                copy_enodes.push_back(static_cast<uint32_t>(scanned_enodes));
    }

    // Smallest canonical class below childCanon holding the same storage
    // weight (same name, memory space, dtype and shape).
    bool earlierWeightClass(const EGraph &egraph, EClassId childCanon, const std::string &tensor_name,
                            EClassId &out) const
    {
        bool found = false;
        const EClass &c1 = egraph.getEClass(childCanon);
        for (uint32_t e : copy_enodes)
        {
            const EClassId otherCanon = egraph.findConst(egraph.getENodeEClass(ENodeId{e}));
            if (otherCanon.value >= childCanon.value || (found && otherCanon.value >= out.value))
                continue;
            std::string other_name;
            EClassId dummy;
            if (!getStorageWeightName(egraph, otherCanon, other_name, dummy) || other_name != tensor_name)
                continue;
            const EClass &c2 = egraph.getEClass(otherCanon);
            if (c1.mem_space == c2.mem_space && c1.dtype == c2.dtype && c1.shape == c2.shape)
            {
                out = otherCanon;
                found = true;
            }
        }
        return found;
    }

    std::string name() const override
    {
        return "ConsumerWeightReuseRule";
    }

    bool canMatchOp(OpType op) const override
    {
        return op != OpType::INPUT && op != OpType::CACHE && op != OpType::COPY_TO;
    }

    // Helper: Checks if an E-class represents a loaded storage weight and returns its tensor name
    bool getStorageWeightName(const EGraph &egraph, EClassId classId, std::string &out_name, EClassId &out_canon) const
    {
        out_canon = egraph.findConst(classId);
        const EClass &cls = egraph.getEClass(out_canon);
        for (ENodeId eid : cls.enodes)
        {
            const ENode &enode = egraph.getENode(eid);
            if (enode.getOpType() == OpType::COPY_TO && !enode.getChildren().empty())
            {
                EClassId storageClsId = egraph.findConst(enode.getChildren()[0]);
                const EClass &storageCls = egraph.getEClass(storageClsId);
                if (storageCls.mem_space.type == HandleType::STORAGE)
                {
                    for (ENodeId sEid : storageCls.enodes)
                    {
                        const ENode &sNode = egraph.getENode(sEid);
                        if (sNode.getOpType() == OpType::INPUT && !sNode.getOpName().empty())
                        {
                            out_name = sNode.getOpName();
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    void clearVisited() override
    {
        visited_enodes.clear();
        copy_enodes.clear();
        scanned_enodes = 0;
    }

    bool prepareSweep(RuleCtx &ctx) override
    {
        refreshCopyIndex(ctx.egraph);
        return !copy_enodes.empty();
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (copy_enodes.empty())
            return false;

        if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark)
            return false;
        if (eNodeIdx < ctx.base_enode_watermark && ctx.sweep_filter &&
            eNodeIdx < ctx.sweep_filter->size() && (*ctx.sweep_filter)[eNodeIdx] == 0)
            return false;
        if (eNodeIdx < visited_enodes.size() && visited_enodes[eNodeIdx])
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getNumENodes())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE ||
            enode.getOpType() == OpType::COPY_TO)
            return false;

        // Check if any child is a loaded storage weight that has an earlier duplicate in the EGraph
        for (EClassId child : enode.getChildren())
        {
            std::string tensor_name;
            EClassId childCanon;
            EClassId earlier;
            if (getStorageWeightName(egraph, child, tensor_name, childCanon))
            {
                refreshCopyIndex(egraph);
                if (earlierWeightClass(egraph, childCanon, tensor_name, earlier))
                    return true;
            }
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        markVisited(eNodeIdx);

        const ENode enode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId eclass_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        std::vector<EClassId> newChildren = enode.getChildren();
        bool changed = false;

        for (size_t i = 0; i < newChildren.size(); ++i)
        {
            std::string tensor_name;
            EClassId childCanon;
            EClassId earlier;
            if (getStorageWeightName(egraph, newChildren[i], tensor_name, childCanon))
            {
                // The earliest matching class avoids cycles and keeps a canonical root.
                refreshCopyIndex(egraph);
                if (earlierWeightClass(egraph, childCanon, tensor_name, earlier))
                {
                    newChildren[i] = earlier;
                    changed = true;
                }
            }
        }

        if (changed)
        {
            ENode altENode(enode.getKernelId(), enode.getOpType(), enode.getOpName(), newChildren, enode.getShape(),
                           enode.getStrides(), enode.getDType(), enode.getMemSpace(), enode.getEngines(),
                           enode.getContentHash());
            egraph.addENode(eclass_id, altENode);
        }
    }
};

// Remove redundant and identity RESHAPE operations.
// 1. Identity Reshape: RESHAPE(x, S) where x.shape == S -> merge(RESHAPE, x)
// 2. Chained Reshapes: RESHAPE(RESHAPE(x, S1), S2) -> RESHAPE(x, S2)
// 3. Consumer Unwrapping: op(..., RESHAPE(x), ...) -> op(..., x, ...) when x.shape matches
struct RemoveRedundantReshape : public Rule
{
    std::unique_ptr<Rule> clone() const override
    {
        auto copy = std::make_unique<RemoveRedundantReshape>(*this);
        copy->scanned_enodes = 0;
        copy->reshape_enodes.clear();
        return copy;
    }

    std::vector<uint8_t> visited;

    void markVisited(uint32_t eNodeIdx)
    {
        if (eNodeIdx >= visited.size())
            visited.resize(eNodeIdx + 1, 0);
        visited[eNodeIdx] = 1;
    }
    size_t scanned_enodes = 0;
    std::vector<uint32_t> reshape_enodes;

    void refreshReshapeIndex(const EGraph &egraph)
    {
        const size_t total_enodes = egraph.getNumENodes();
        if (total_enodes < scanned_enodes)
        {
            scanned_enodes = 0;
            reshape_enodes.clear();
        }
        for (; scanned_enodes < total_enodes; ++scanned_enodes)
            if (egraph.getENode(ENodeId{static_cast<uint32_t>(scanned_enodes)}).getOpType() == OpType::RESHAPE)
                reshape_enodes.push_back(static_cast<uint32_t>(scanned_enodes));
    }

    std::string name() const override
    {
        return "RemoveRedundantReshape";
    }

    bool canMatchOp(OpType op) const override
    {
        return op != OpType::INPUT && op != OpType::CACHE;
    }

    void clearVisited() override
    {
        visited.clear();
        reshape_enodes.clear();
        scanned_enodes = 0;
    }

    bool prepareSweep(RuleCtx &ctx) override
    {
        refreshReshapeIndex(ctx.egraph);
        return !reshape_enodes.empty();
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (reshape_enodes.empty())
            return false;
        if (eNodeIdx < ctx.base_enode_watermark && ctx.sweep_filter &&
            eNodeIdx < ctx.sweep_filter->size() && (*ctx.sweep_filter)[eNodeIdx] == 0)
            return false;
        if (eNodeIdx < visited.size() && visited[eNodeIdx])
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getNumENodes())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});

        if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark)
        {
            bool has_new_child = false;
            for (EClassId child : enode.getChildren())
            {
                const EClass &cls = egraph.getEClass(egraph.findConst(child));
                if (!cls.enodes.empty() && cls.enodes.back().value >= ctx.base_enode_watermark)
                {
                    has_new_child = true;
                    break;
                }
            }
            if (!has_new_child)
                return false;
        }

        // Pattern 1 & 2: Direct RESHAPE node
        if (enode.getOpType() == OpType::RESHAPE && !enode.getChildren().empty())
        {
            EClassId srcClassId = egraph.findConst(enode.getChildren()[0]);
            const EClass &srcCls = egraph.getEClass(srcClassId);

            // Case 1: Identity Reshape (input shape == output shape)
            if (srcCls.shape == enode.getShape() && srcCls.dtype == enode.getDType())
            {
                EClassId e_class_id = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx}));
                if (e_class_id != srcClassId)
                {
                    return true;
                }
            }

            // Case 2: Chained Reshapes
            for (ENodeId cEnodeId : srcCls.enodes)
            {
                const ENode &cEnode = egraph.getENode(cEnodeId);
                if (cEnode.getOpType() == OpType::RESHAPE && !cEnode.getChildren().empty())
                {
                    return true;
                }
            }
        }

        // Pattern 3: Consumer node taking an identity RESHAPE child
        if (enode.getOpType() != OpType::INPUT && enode.getOpType() != OpType::CACHE)
        {
            if (KernelRegistry::get().hasKernel(enode.getKernelId()))
            {
                const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
                if (kernel.is_view)
                    return false; // Skip view kernels where stride calculation depends on input view
            }

            for (EClassId child : enode.getChildren())
            {
                EClassId childClsId = egraph.findConst(child);
                const EClass &childCls = egraph.getEClass(childClsId);

                for (ENodeId cEnodeId : childCls.enodes)
                {
                    const ENode &cEnode = egraph.getENode(cEnodeId);
                    if (cEnode.getOpType() == OpType::RESHAPE && !cEnode.getChildren().empty())
                    {
                        EClassId unwrappedChildId = egraph.findConst(cEnode.getChildren()[0]);
                        const EClass &unwrappedCls = egraph.getEClass(unwrappedChildId);
                        if (unwrappedCls.shape == childCls.shape && unwrappedChildId != childClsId)
                        {
                            return true;
                        }
                    }
                }
            }
        }

        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        markVisited(eNodeIdx);
        EGraph &egraph = ctx.egraph;
        const ENode enode = egraph.getENode(ENodeId{eNodeIdx});
        if (ctx.base_enode_watermark > 0 && eNodeIdx < ctx.base_enode_watermark)
        {
            bool has_new_child = false;
            for (EClassId child : enode.getChildren())
            {
                const EClass &cls = egraph.getEClass(egraph.findConst(child));
                if (!cls.enodes.empty() && cls.enodes.back().value >= ctx.base_enode_watermark)
                {
                    has_new_child = true;
                    break;
                }
            }
            if (!has_new_child)
                return;
        }
        EClassId e_class_id = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx}));

        // 1. Direct Identity Reshape -> Merge EClasses
        if (enode.getOpType() == OpType::RESHAPE && !enode.getChildren().empty())
        {
            EClassId srcClassId = egraph.findConst(enode.getChildren()[0]);
            const EClass srcCls = egraph.getEClass(srcClassId);

            if (srcCls.shape == enode.getShape() && srcCls.dtype == enode.getDType() &&
                srcCls.mem_space == enode.getMemSpace() && srcCls.strides == enode.getStrides())
            {
                egraph.merge(e_class_id, srcClassId);
                return;
            }

            // 2. Chained Reshape: RESHAPE(RESHAPE(x, S1), S2) -> add RESHAPE(x, S2)
            for (ENodeId cEnodeId : srcCls.enodes)
            {
                const ENode cEnode = egraph.getENode(cEnodeId);
                if (cEnode.getOpType() == OpType::RESHAPE && !cEnode.getChildren().empty())
                {
                    EClassId grandChildId = egraph.findConst(cEnode.getChildren()[0]);
                    std::vector<EClassId> newChildren = enode.getChildren();
                    newChildren[0] = grandChildId;

                    ENode collapsed(enode.getKernelId(), OpType::RESHAPE, "", newChildren, enode.getShape(),
                                    enode.getStrides(), enode.getDType(), enode.getMemSpace(), enode.getEngines(),
                                    enode.getContentHash());
                    egraph.addENode(e_class_id, collapsed);
                }
            }
        }

        // 3. Consumer child unwrapping
        if (enode.getOpType() != OpType::INPUT && enode.getOpType() != OpType::CACHE &&
            enode.getOpType() != OpType::RESHAPE)
        {
            std::vector<std::vector<EClassId>> candidateChildrenPerPos(enode.getChildren().size());

            for (uint64_t i = 0; i < enode.getChildren().size(); ++i)
            {
                EClassId childClsId = egraph.findConst(enode.getChildren()[i]);
                candidateChildrenPerPos[i].push_back(childClsId);

                const EClass &childCls = egraph.getEClass(childClsId);
                for (ENodeId cEnodeId : childCls.enodes)
                {
                    const ENode &cEnode = egraph.getENode(cEnodeId);
                    if (cEnode.getOpType() == OpType::RESHAPE && !cEnode.getChildren().empty())
                    {
                        EClassId unwrappedChildId = egraph.findConst(cEnode.getChildren()[0]);
                        const EClass &unwrappedCls = egraph.getEClass(unwrappedChildId);
                        if (unwrappedCls.shape == childCls.shape && unwrappedChildId != childClsId)
                        {
                            bool contig_ok = true;
                            if (KernelRegistry::get().hasKernel(enode.getKernelId()))
                            {
                                const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
                                if (!kernel.requiresContiguous.empty())
                                {
                                    size_t rule_idx =
                                        std::min(static_cast<size_t>(i), kernel.requiresContiguous.size() - 1);
                                    if (kernel.requiresContiguous[rule_idx] && !isContiguous(unwrappedCls))
                                    {
                                        contig_ok = false;
                                    }
                                }
                            }
                            if (contig_ok)
                            {
                                candidateChildrenPerPos[i].push_back(unwrappedChildId);
                            }
                        }
                    }
                }
            }

            std::vector<std::vector<EClassId>> childCombinations;
            std::vector<EClassId> currentCombination(enode.getChildren().size());

            std::function<void(uint64_t, bool)> generateCombos = [&](uint64_t pos, bool hasUnwrapped) {
                if (pos == enode.getChildren().size())
                {
                    if (hasUnwrapped)
                        childCombinations.push_back(currentCombination);
                    return;
                }
                for (uint64_t cIdx = 0; cIdx < candidateChildrenPerPos[pos].size(); ++cIdx)
                {
                    currentCombination[pos] = candidateChildrenPerPos[pos][cIdx];
                    generateCombos(pos + 1, hasUnwrapped || (cIdx > 0));
                }
            };

            generateCombos(0, false);

            for (const auto &newChildren : childCombinations)
            {
                ENode newENode(enode.getKernelId(), enode.getOpType(), enode.getOpName(), newChildren, enode.getShape(),
                               enode.getStrides(), enode.getDType(), enode.getMemSpace(), enode.getEngines(),
                               enode.getContentHash());
                egraph.addENode(e_class_id, newENode);
            }
        }
    }
};
