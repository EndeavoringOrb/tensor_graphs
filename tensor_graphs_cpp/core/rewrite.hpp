#pragma once
#include <algorithm>
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

struct RuleCtx
{
    EGraph &egraph;
    const std::unordered_set<EClassId> &protectedEClasses;
    TGStore *repo;
    CostModel *costModel = nullptr;
    // Dirty/needed regions of the bucket being saturated (partial buckets).
    const BucketRegions *regions = nullptr;
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
    return ctx.regions->coversLive(cls.logical_id, box);
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
    const EClass a = egraph.getEClass(contigClass);
    const EClass b = egraph.getEClass(restricted);
    if (a.shape != b.shape || a.dtype != b.dtype)
        Error::throw_err("[mergeRestricted] restricted computation does not match its slice: " + toString(a.shape) +
                         " vs " + toString(b.shape));
    if (a.strides != b.strides || a.mem_space != b.mem_space)
        return;
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
        inNodes.push_back(in);
        input_mem_spaces.push_back(childCls.mem_space);
    }

    // Kernel matching depends only on the op and the input/output layouts and
    // memory spaces; memoize it per thread (no locking, no pattern graphs).
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
    thread_local std::unordered_map<std::string, OpMatches> opMatchMemo;
    std::string key;
    {
        auto add = [&key](const void *data, size_t size) {
            key.append(reinterpret_cast<const char *>(&size), sizeof(size));
            key.append(static_cast<const char *>(data), size);
        };
        const uint32_t opValue = static_cast<uint32_t>(op), dtypeValue = static_cast<uint32_t>(dtype);
        add(&opValue, sizeof(opValue));
        add(&dtypeValue, sizeof(dtypeValue));
        add(shape.data(), shape.size() * sizeof(uint32_t));
        add(strides.data(), strides.size() * sizeof(uint64_t));
        add(&mem_space, sizeof(mem_space));
        for (size_t i = 0; i < inNodes.size(); ++i)
        {
            const uint32_t d = static_cast<uint32_t>(inNodes[i].dtype);
            add(&d, sizeof(d));
            add(inNodes[i].getShape().data(), inNodes[i].getShape().size() * sizeof(uint32_t));
            add(inNodes[i].strides.data(), inNodes[i].strides.size() * sizeof(uint64_t));
            add(&input_mem_spaces[i], sizeof(MemSpace));
        }
    }
    auto memo = opMatchMemo.find(key);
    if (memo == opMatchMemo.end())
    {
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
        memo = opMatchMemo.emplace(std::move(key), std::move(result)).first;
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
               << "  Inputs (" << inNodes.size() << "):\n";
            for (uint64_t i = 0; i < inNodes.size(); ++i)
            {
                ss << "    Input #" << i << ": "
                   << "dtype=" << toString(inNodes[i].dtype) << ", shape=" << toString(inNodes[i].getShape())
                   << ", strides=" << toString(inNodes[i].strides) << ", mem_space=" << toString(input_mem_spaces[i])
                   << "\n";
            }

            Error::throw_err(ss.str());
        }
        for (const MatchedKernel &matched : memo->second.kernels)
        {
            const auto &kernel = KernelRegistry::get().getKernel(matched.uid);

            std::vector<EClassId> adapted_children = children;
            for (size_t input_idx = 0; input_idx < adapted_children.size(); ++input_idx)
            {
                if (kernel.requiresContiguous.empty())
                    continue;

                const size_t rule_idx = std::min(input_idx, kernel.requiresContiguous.size() - 1);
                if (!kernel.requiresContiguous[rule_idx] || isContiguous(inNodes[input_idx]))
                    continue;

                const EClassId source_class = egraph.findConst(adapted_children[input_idx]);
                const auto source = egraph.getEClass(source_class);
                EClassId contiguous_class = addOpToEGraph(
                    egraph, OpType::CONTIGUOUS, {source_class}, source.shape,
                    calcContiguousStrides(source.shape), source.dtype, source.mem_space, EClassId(), loc,
                    origin.empty() ? "addOpToEGraph.make_contiguous" : (origin + " -> make_contiguous"));
                adapted_children[input_idx] = contiguous_class;
            }

            ENode n(matched.uid, op, kernel.opName, adapted_children, shape,
                    kernel.is_view ? strides : calcContiguousStrides(shape), dtype, mem_space, matched.engines,
                    contentHash, 0, origin);
            if (cls == EClassId())
                cls = egraph.lookupENode(n);
            if (cls == EClassId())
                cls = egraph.addEClass(shape, strides, dtype, mem_space);
            cls = egraph.addENode(cls, n);
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
    EClass cls = egraph.getEClass(canon);
    if (cls.mem_space == target_mem_space)
        return canon;

    if (!isContiguous(cls))
    {
        canon = addOpToEGraph(egraph, OpType::CONTIGUOUS, {canon}, cls.shape,
                              calcContiguousStrides(cls.shape), cls.dtype, cls.mem_space,
                              EClassId(), SourceLocation::current(),
                              debugOrigin.empty() ? "copyTo.make_contiguous" : (debugOrigin + " -> make_contiguous"));
        cls = egraph.getEClass(canon);
    }

    return addOpToEGraph(egraph, OpType::COPY_TO, {canon}, cls.shape, cls.strides, cls.dtype, target_mem_space,
                         EClassId(), SourceLocation::current(), debugOrigin);
}

inline EClassId createCacheInputNode(EGraph &egraph, EClassId sourceClassId,
                                     const std::string &debugOrigin = "")
{
    EClassId canonSrcClass = egraph.findConst(sourceClassId);
    const EClass srcClass = egraph.getEClass(canonSrcClass);

    LogicalId srcLogicalId = srcClass.logical_id;

    EClassId op_cache =
        egraph.addEClass(srcClass.shape, srcClass.strides, srcClass.dtype, srcClass.mem_space, srcLogicalId);

    ENode cacheNode(KernelId{0}, OpType::CACHE, "", {}, srcClass.shape, srcClass.strides, srcClass.dtype,
                    srcClass.mem_space, {}, toString(srcLogicalId), 0, debugOrigin);
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

    static std::unordered_map<LogicalId, EClassId> toMap(const Pattern &pattern, const Binding &b)
    {
        std::unordered_map<LogicalId, EClassId> out;
        for (size_t i = 0; i < pattern.variables.size(); ++i)
            if (b.slots[i] != EClassId{})
                out[pattern.variables[i]] = b.slots[i];
        return out;
    }

    struct MatchResult
    {
        const Pattern *pattern;
        std::unordered_map<LogicalId, EClassId> binding;
        std::vector<EClassId> variadicConcatTensorEClasses;
    };

    struct PatternSet
    {
        std::unordered_map<OpType, std::vector<Pattern>> singleNodePatternsByOp;
        std::unordered_map<OpType, std::vector<Pattern>> multiNodePatternsByOp;
    };
    // Immutable after construction and shared by clones, so the Pattern
    // addresses in completed_applications stay valid across clones.
    std::shared_ptr<const PatternSet> patterns;
    std::vector<MatchResult> activeMatches;
    std::unordered_set<uint32_t> single_node_visited;

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
        EClassId destination;
        std::vector<EClassId> inputs;
        // Staged constant data of each input (null if none). Equal data is
        // one class with one staging buffer, so the pointer identifies it.
        std::vector<const void *> constants;

        bool operator==(const ApplicationKey &other) const
        {
            return root_idx == other.root_idx && pattern == other.pattern && destination == other.destination &&
                   inputs == other.inputs && constants == other.constants;
        }
    };

    struct ApplicationHash
    {
        size_t operator()(const ApplicationKey &key) const
        {
            uint64_t hash_value = key.root_idx;
            tg_hash::hashCombine(hash_value, std::hash<const Pattern *>{}(key.pattern));
            tg_hash::hashCombine(hash_value, key.destination.value);
            for (EClassId input_id : key.inputs)
                tg_hash::hashCombine(hash_value, input_id.value);
            // Constant bytes participate in equality; unchanged class IDs make a
            // good bucket key without hashing potentially large buffers again.
            return static_cast<size_t>(hash_value);
        }
    };

    // Scoped to one saturation, whose kernel registry, hardware and cost records
    // are fixed. E-class tensor metadata is invariant; merges change canonical
    // IDs, and late constants change the exact snapshot below. Matching still
    // runs every sweep, so new bindings at arbitrary pattern depth are observed.
    std::unordered_set<ApplicationKey, ApplicationHash> completed_applications;
    uint64_t application_cache_hits = 0;

    // Candidate selection per signature (see selectFusedCandidates). Kept per
    // thread (no locking); the key includes the cost model it was made with.
    struct FusedCandidate;
    using SelectionMemo = std::unordered_map<std::string, std::shared_ptr<const std::vector<FusedCandidate>>>;
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
                   << completed_applications.size() << " completed applications";
    }

    FusionRule(bool disableFusion = false)
    {
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

            if (disableFusion)
            {
                continue;
            }

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
                singleNodePatternsByOp[pattern.rootOpType].push_back(std::move(pattern));
            }
            else
            {
                multiNodePatternsByOp[pattern.rootOpType].push_back(std::move(pattern));
            }
        }
        patterns = std::move(pattern_set);
        for (const auto &[op, list] : patterns->multiNodePatternsByOp)
            for (const auto &pattern : list)
                max_pattern_depth = std::max(max_pattern_depth, pattern.depth);
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        activeMatches.clear();
        const EGraph &egraph = ctx.egraph;
        const ENode &eNode = egraph.getENode(ENodeId{eNodeIdx});

        if (single_node_visited.insert(eNodeIdx).second)
        {
            auto it_single = patterns->singleNodePatternsByOp.find(eNode.getOpType());
            if (it_single != patterns->singleNodePatternsByOp.end())
            {
                for (const auto &pattern : it_single->second)
                {
                    Binding binding(pattern.variables.size());
                    if (matchPatternNode(ENodeId{eNodeIdx}, egraph, pattern.rootId, pattern, binding, ctx.protectedEClasses))
                    {
                        MatchResult mr;
                        mr.pattern = &pattern;
                        mr.binding = toMap(pattern, binding);

                        if (eNode.getOpType() == OpType::CONCAT && eNode.getChildren().size() > 2)
                        {
                            for (uint64_t i = 1; i < eNode.getChildren().size(); ++i)
                            {
                                mr.variadicConcatTensorEClasses.push_back(egraph.findConst(eNode.getChildren()[i]));
                            }
                        }

                        activeMatches.push_back(std::move(mr));
                    }
                }
            }
        }

        auto it_multi = patterns->multiNodePatternsByOp.find(eNode.getOpType());
        // Distance (in levels) from this enode down to the nearest class
        // that changed in the previous sweep; patterns shallower than that
        // read only unchanged classes.
        uint32_t changedDistance = 0;
        const bool filtered = sweep_filter && eNodeIdx < sweep_filter->size();
        if (filtered)
        {
            changedDistance = (*sweep_filter)[eNodeIdx];
            if (changedDistance == 0)
                it_multi = patterns->multiNodePatternsByOp.end();
        }
        if (it_multi != patterns->multiNodePatternsByOp.end())
        {
            for (const auto &pattern : it_multi->second)
            {
                if (filtered && pattern.depth < changedDistance)
                    continue;
                Binding binding(pattern.variables.size());
                if (matchPatternNode(ENodeId{eNodeIdx}, egraph, pattern.rootId, pattern, binding, ctx.protectedEClasses))
                {
                    MatchResult mr;
                    mr.pattern = &pattern;
                    mr.binding = toMap(pattern, binding);

                    if (eNode.getOpType() == OpType::CONCAT && eNode.getChildren().size() > 2)
                    {
                        for (uint64_t i = 1; i < eNode.getChildren().size(); ++i)
                        {
                            mr.variadicConcatTensorEClasses.push_back(egraph.findConst(eNode.getChildren()[i]));
                        }
                    }

                    activeMatches.push_back(std::move(mr));
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
            const auto &binding = match.binding;

            ApplicationKey application_key;
            application_key.root_idx = eNodeIdx;
            application_key.pattern = &pattern;
            application_key.destination = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx}));
            if (!match.variadicConcatTensorEClasses.empty())
            {
                application_key.inputs.push_back(egraph.findConst(binding.at(pattern.variables.front())));
                for (EClassId input_id : match.variadicConcatTensorEClasses)
                    application_key.inputs.push_back(egraph.findConst(input_id));
            }
            else
            {
                for (LogicalId var : pattern.variables)
                    application_key.inputs.push_back(egraph.findConst(binding.at(var)));
            }
            for (EClassId input_id : application_key.inputs)
            {
                auto it = egraph.constantStaging.find(input_id);
                application_key.constants.push_back(it == egraph.constantStaging.end() ? nullptr
                                                                                       : it->second->data());
            }
            if (completed_applications.count(application_key))
            {
                ++application_cache_hits;
                continue;
            }

            std::vector<EClassId> inputs;
            std::vector<TensorNode> inputNodes;

            if (!match.variadicConcatTensorEClasses.empty())
            {
                // CONCAT's graph and kernel ABI is [axis, tensor0, tensor1, ...].
                // Keep that order when materializing a variadic fused match; otherwise
                // the first FLOAT32 tensor is interpreted as the INT32 axis at runtime.
                LogicalId axisVar = pattern.variables.front();
                EClassId axisEClass = binding.at(axisVar);
                inputs.push_back(axisEClass);
                const EClass &axisParent = egraph.getEClass(axisEClass);
                TensorNode axisInputNode;
                axisInputNode.opType = OpType::INPUT;
                axisInputNode.dtype = axisParent.dtype;
                axisInputNode.setShape(axisParent.shape);
                axisInputNode.strides = axisParent.strides;
                inputNodes.push_back(std::move(axisInputNode));

                for (EClassId tensorEClass : match.variadicConcatTensorEClasses)
                {
                    inputs.push_back(tensorEClass);
                    const EClass parent = egraph.getEClass(tensorEClass);
                    TensorNode inputNode;
                    inputNode.opType = OpType::INPUT;
                    inputNode.dtype = parent.dtype;
                    inputNode.setShape(parent.shape);
                    inputNode.strides = parent.strides;
                    inputNodes.push_back(std::move(inputNode));
                }
            }
            else
            {
                inputs.reserve(pattern.variables.size());
                inputNodes.reserve(pattern.variables.size());

                for (LogicalId var : pattern.variables)
                {
                    EClassId parentEClassId = binding.at(var);
                    const EClass &parent = egraph.getEClass(parentEClassId);
                    inputs.push_back(parentEClassId);

                    TensorNode inputNode;
                    inputNode.opType = OpType::INPUT;
                    inputNode.dtype = parent.dtype;
                    inputNode.setShape(parent.shape);
                    inputNode.strides = parent.strides;
                    inputNodes.push_back(std::move(inputNode));
                }
            }

            const EClass matchedClass = egraph.getEClass(egraph.getENodeEClass(ENodeId{eNodeIdx}));

            DType outDtype = matchedClass.dtype;
            std::vector<uint32_t> outShape = matchedClass.shape;
            std::vector<uint64_t> outStrides = matchedClass.strides;

            TensorNode outputNode;
            outputNode.opType = OpType::FUSED;
            outputNode.opName = pattern.opName;
            outputNode.dtype = outDtype;
            outputNode.setShape(outShape);
            outputNode.strides = outStrides;

            // Discovery must see reference expressions even when every current
            // implementation rejects this shape/layout. This does not add an
            // executable candidate or change normal compiler matching.
            if (ctx.costModel && ctx.costModel->referenceObserver)
            {
                std::vector<std::vector<uint8_t>> constantBytes;
                for (EClassId input_id : application_key.inputs)
                {
                    auto it = egraph.constantStaging.find(input_id);
                    constantBytes.push_back(it == egraph.constantStaging.end() ? std::vector<uint8_t>{}
                                                                               : *it->second);
                }
                ctx.costModel->referenceObserver(pattern.opName, inputNodes, outputNode, constantBytes);
            }

            bool ignoreInputMemSpaces = (pattern.rootOpType != OpType::COPY_TO);

            std::vector<MemSpace> candidateSpaces = {matchedClass.mem_space};
            for (const auto &avail_ms : System::get().getAvailableMemSpaces())
            {
                if (avail_ms.type != HandleType::STORAGE && !(avail_ms == matchedClass.mem_space))
                {
                    candidateSpaces.push_back(avail_ms);
                }
            }

            std::string signature;
            {
                const CostModel *model = ctx.costModel;
                appendSignature(signature, &model, sizeof(model));
                const ENode &old = egraph.getENode(ENodeId{eNodeIdx});
                appendSignature(signature, pattern.pattern_hash.data(), pattern.pattern_hash.size());
                const uint32_t opType = static_cast<uint32_t>(old.getOpType());
                appendSignature(signature, &opType, sizeof(opType));
                appendSignature(signature, old.getShape().data(), old.getShape().size() * sizeof(uint32_t));
                appendSignature(signature, old.getStrides().data(), old.getStrides().size() * sizeof(uint64_t));
                const uint32_t outD = static_cast<uint32_t>(old.getDType());
                appendSignature(signature, &outD, sizeof(outD));
                appendSignature(signature, outStrides.data(), outStrides.size() * sizeof(uint64_t));
                const MemSpace ms = matchedClass.mem_space;
                appendSignature(signature, &ms, sizeof(ms));
                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    const EClass &child = egraph.getEClass(egraph.findConst(inputs[i]));
                    appendSignature(signature, child.shape.data(), child.shape.size() * sizeof(uint32_t));
                    appendSignature(signature, child.strides.data(), child.strides.size() * sizeof(uint64_t));
                    const uint32_t d = static_cast<uint32_t>(child.dtype);
                    appendSignature(signature, &d, sizeof(d));
                    const MemSpace cms = child.mem_space;
                    appendSignature(signature, &cms, sizeof(cms));
                    auto staged = egraph.constantStaging.find(egraph.findConst(inputs[i]));
                    if (staged != egraph.constantStaging.end() && !isScatterControlInput(old.getOpType(), i))
                        appendSignature(signature, staged->second->data(), staged->second->size());
                    else
                        appendSignature(signature, nullptr, 0);
                }
            }

            for (const auto &target_ms : candidateSpaces)
            {
                std::string key = signature;
                appendSignature(key, &target_ms, sizeof(target_ms));
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

                std::vector<KernelId> kernelMatches = KernelPatternCache::get().find(
                    pattern.graph, pattern.rootId, inputNodes, outputNode, false, target_ms,
                    {}, {}, true, ignoreInputMemSpaces, true, true);

                std::vector<std::pair<const KernelEntry *, std::pair<std::vector<Engine>, std::vector<MemSpace>>>>
                    matched_kernels;
                for (KernelId uid : kernelMatches)
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
                                  << kernelMatches.size() << " matched=" << matched_kernels.size()
                                  << " selected=" << names << std::endl;
                    }
                materializeFusedCandidates(ctx, *selected, inputs, ENodeId{eNodeIdx});
            }
            completed_applications.insert(std::move(application_key));
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
                const EClass parent = egraph.getEClass(egraph.findConst(pid));

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
        const ENode oldENode = egraph.getENode(eNodeIdx);
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
                const EClass parent = egraph.getEClass(egraph.findConst(pid));

                bool needCopy = cand.child_need_copy[i];
                bool needContig = cand.child_need_contig[i];

                if (!needCopy && !needContig)
                {
                    adapted_children.push_back(pid);
                    continue;
                }

                EClassId currentPid = pid;
                EClass currentClass = parent;

                if (needContig)
                {
                    currentPid = addOpToEGraph(egraph, OpType::CONTIGUOUS, {currentPid}, currentClass.shape,
                                               calcContiguousStrides(currentClass.shape), currentClass.dtype,
                                               currentClass.mem_space);
                    currentClass = egraph.getEClass(egraph.findConst(currentPid));
                }

                if (needCopy)
                {
                    EClassId finalTargetClass = EClassId();
                    for (const auto &path : cand.child_mem_paths[i])
                    {
                        EClassId pathPid = currentPid;
                        EClass pathClass = currentClass;
                        for (uint64_t p_idx = 1; p_idx < path.size(); ++p_idx)
                        {
                            MemSpace next_ms = path[p_idx];
                            if (!isContiguous(pathClass))
                            {
                                pathPid = addOpToEGraph(egraph, OpType::CONTIGUOUS, {pathPid}, pathClass.shape,
                                                        calcContiguousStrides(pathClass.shape), pathClass.dtype,
                                                        pathClass.mem_space);
                                pathClass = egraph.getEClass(egraph.findConst(pathPid));
                            }
                            pathPid = addOpToEGraph(egraph, OpType::COPY_TO, {pathPid}, pathClass.shape,
                                                    pathClass.strides, pathClass.dtype, next_ms,
                                                    (p_idx == path.size() - 1) ? finalTargetClass : EClassId());
                            if (p_idx == path.size() - 1)
                            {
                                finalTargetClass = pathPid;
                            }
                            pathClass = egraph.getEClass(egraph.findConst(pathPid));
                        }
                    }
                    currentPid = finalTargetClass;
                }

                adapted_children.push_back(currentPid);
            }

            std::vector<uint64_t> strides =
                cand.kernel->is_view ? oldENode.getStrides() : calcContiguousStrides(oldENode.getShape());

            ENode enode(cand.kernel->uid, cand.kernel->opType, cand.kernel->opName, adapted_children,
                        oldENode.getShape(), strides, oldENode.getDType(), cand.target_mem_space, cand.mapped_engines,
                        oldENode.getContentHash(), 0, oldENode.getDebugOrigin());

            const EClass origCls = egraph.getEClass(egraph.findConst(e_class_id));
            MemSpace originalMemSpace = origCls.mem_space;
            if (cand.target_mem_space == originalMemSpace)
            {
                if (enode.getStrides() == origCls.strides)
                {
                    bool duplicate = false;
                    for (ENodeId existing : origCls.enodes)
                    {
                        const ENode &other = egraph.getENode(existing);
                        if (other.getKernelId() != enode.getKernelId() ||
                            other.getMemSpace() != enode.getMemSpace() ||
                            other.getStrides() != enode.getStrides() ||
                            other.getChildren().size() != enode.getChildren().size())
                            continue;
                        bool same_children = true;
                        for (size_t c = 0; c < enode.getChildren().size() && same_children; ++c)
                            same_children = egraph.findConst(other.getChildren()[c]) ==
                                            egraph.findConst(enode.getChildren()[c]);
                        if (same_children)
                        {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate)
                        egraph.addENode(e_class_id, enode);
                }
            }
            else
            {
                EClassId newEClass =
                    egraph.addEClass(enode.getShape(), enode.getStrides(), enode.getDType(), cand.target_mem_space);
                newEClass = egraph.addENode(newEClass, enode);

                EClassId srcForCopy = newEClass;
                const EClass srcCls = egraph.getEClass(egraph.findConst(srcForCopy));

                if (!isContiguous(srcCls))
                {
                    srcForCopy = addOpToEGraph(egraph, OpType::CONTIGUOUS, {srcForCopy}, srcCls.shape,
                                               calcContiguousStrides(srcCls.shape), srcCls.dtype, srcCls.mem_space);
                }

                const EClass copySrcCls = egraph.getEClass(egraph.findConst(srcForCopy));
                if (isContiguous(origCls))
                {
                    addOpToEGraph(egraph, OpType::COPY_TO, {srcForCopy}, copySrcCls.shape, copySrcCls.strides,
                                  copySrcCls.dtype, originalMemSpace, e_class_id);
                }
                else
                {
                    addOpToEGraph(egraph, OpType::COPY_TO, {srcForCopy}, copySrcCls.shape, copySrcCls.strides,
                                  copySrcCls.dtype, originalMemSpace);
                }
            }
        }
    }

    static bool matchPatternClass(EClassId eClassIdx, const EGraph &egraph, LogicalId patternId, const Pattern &pattern,
                                  Binding &binding, const std::unordered_set<EClassId> &protectedEClasses,
                                  bool ignoreConstantData = false)
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
            if (matchPatternNode(enodeId, egraph, patternId, pattern, binding, protectedEClasses, ignoreConstantData))
                return true;
            binding.undo(mark);
        }
        return false;
    }

    static bool matchPatternNode(ENodeId eNodeId, const EGraph &egraph, LogicalId patternId, const Pattern &pattern,
                                 Binding &binding,
                                 const std::unordered_set<EClassId> &protectedEClasses, bool ignoreConstantData = false)
    {
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
                                   protectedEClasses, false))
                return false;

            // The rest are tensors. The pattern has one tensor at index 1.
            bool firstTensor = true;
            for (uint64_t i = 1; i < eNode.getChildren().size(); ++i)
            {
                if (firstTensor)
                {
                    if (!matchPatternClass(eNode.getChildren()[i], egraph, pNode.child_ids[1], pattern, binding,
                                           protectedEClasses, true))
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
                                   protectedEClasses, childIgnoreConst))
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

    std::unordered_set<ENodeId> visited_enodes;

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

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (visited_enodes.count(ENodeId{eNodeIdx}))
            return false;

        const ENode &enode = ctx.egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::ADD || enode.getChildren().size() != 2)
            return false;

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
        visited_enodes.insert(ENodeId{eNodeIdx});

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

    std::unordered_set<uint32_t> visited_classes;

    std::string name() const override
    {
        return "ScatterPullUp";
    }

    static bool hasScatter(const EGraph &egraph, EClassId cid)
    {
        for (ENodeId id : egraph.getEClass(egraph.findConst(cid)).enodes)
            if (egraph.getENode(id).getOpType() == OpType::SCATTER)
                return true;
        return false;
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
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() == OpType::SCATTER || enode.getOpType() == OpType::INPUT ||
            enode.getOpType() == OpType::CACHE)
            return false;
        const EClassId cid = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx}));
        if (visited_classes.count(cid.value))
            return false;
        bool childUpdated = false;
        for (EClassId child : enode.getChildren())
            childUpdated = childUpdated || isUpdated(egraph, child, 4, ctx);
        if (!childUpdated)
            return false;
        Region box;
        return regionFor(egraph.getEClass(cid), ctx, box);
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const EClassId cid = egraph.findConst(egraph.getENodeEClass(ENodeId{eNodeIdx}));
        visited_classes.insert(cid.value);
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
        uint32_t contigIdx;
        uint32_t sliceIdx;
        uint32_t srcNodeIdx;
        bool operator==(const MatchKey &o) const
        {
            return contigIdx == o.contigIdx && sliceIdx == o.sliceIdx && srcNodeIdx == o.srcNodeIdx;
        }
    };
    struct MatchKeyHash
    {
        std::uint64_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.contigIdx) ^ (std::hash<uint32_t>{}(k.sliceIdx) << 1) ^
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

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::CONTIGUOUS || enode.getChildren().empty())
            return false;

        EClassId childClass = egraph.findConst(enode.getChildren()[0]);
        for (ENodeId childNodeIdx : egraph.getEClass(childClass).enodes)
        {
            const ENode &childNode = egraph.getENode(childNodeIdx);
            if (childNode.getOpType() == OpType::SLICE && childNode.getChildren().size() == 4)
            {
                EClassId srcClass = egraph.findConst(childNode.getChildren()[0]);

                // Cheap test first: an unvisited elementwise source enode
                // without broadcast children.
                bool candidate = false;
                for (ENodeId srcNodeIdx : egraph.getEClass(srcClass).enodes)
                {
                    const ENode &opNode = egraph.getENode(srcNodeIdx);
                    OpType op = opNode.getOpType();
                    if (!isElementwise(op) || op == OpType::COPY_TO || op == OpType::CONTIGUOUS)
                        continue;
                    if (visited.count(MatchKey{eNodeIdx, childNodeIdx.value, srcNodeIdx.value}))
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
                    {
                        candidate = true;
                        break;
                    }
                }
                if (!candidate)
                    continue;

                if (egraph.constantStaging.count(egraph.findConst(childNode.getChildren()[1])) == 0 ||
                    egraph.constantStaging.count(egraph.findConst(childNode.getChildren()[2])) == 0 ||
                    egraph.constantStaging.count(egraph.findConst(childNode.getChildren()[3])) == 0)
                    continue;

                auto starts = egraph.getConstantInt32(egraph.findConst(childNode.getChildren()[1]));
                auto ends = egraph.getConstantInt32(egraph.findConst(childNode.getChildren()[2]));
                auto steps = egraph.getConstantInt32(egraph.findConst(childNode.getChildren()[3]));

                if (starts.empty() || ends.empty() || steps.empty())
                    continue;

                const auto &origShape = egraph.getEClass(srcClass).shape;
                bool isFull = true;
                if (starts.size() != origShape.size())
                    isFull = false;
                for (uint64_t d = 0; d < starts.size() && isFull; ++d)
                {
                    int32_t st = starts[d] < 0 ? starts[d] + origShape[d] : starts[d];
                    int32_t en = ends[d] < 0 ? ends[d] + origShape[d] : ends[d];
                    if (st != 0 || en != (int32_t)origShape[d] || steps[d] != 1)
                    {
                        isFull = false;
                    }
                }
                if (isFull)
                    continue;

                if (!allowPushDownOnProtected && isEClassProtected(srcClass, ctx.protectedEClasses, egraph))
                    continue;
                return true;
            }
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const ENode contigNode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        EClassId sliceClass = egraph.findConst(contigNode.getChildren()[0]);

        std::vector<ENodeId> sliceNodes;
        for (ENodeId childNodeIdx : egraph.getEClass(sliceClass).enodes)
        {
            const ENode &childNode = egraph.getENode(childNodeIdx);
            if (childNode.getOpType() == OpType::SLICE && childNode.getChildren().size() == 4)
            {
                sliceNodes.push_back(childNodeIdx);
            }
        }

        for (ENodeId sliceNodeIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENode(sliceNodeIdx);

            EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            EClassId startsId = sliceNode.getChildren()[1];
            EClassId endsId = sliceNode.getChildren()[2];
            EClassId stepsId = sliceNode.getChildren()[3];

            if (egraph.constantStaging.count(egraph.findConst(startsId)) == 0 ||
                egraph.constantStaging.count(egraph.findConst(endsId)) == 0 ||
                egraph.constantStaging.count(egraph.findConst(stepsId)) == 0)
                continue;

            auto starts = egraph.getConstantInt32(egraph.findConst(startsId));
            auto ends = egraph.getConstantInt32(egraph.findConst(endsId));
            auto steps = egraph.getConstantInt32(egraph.findConst(stepsId));
            if (starts.empty() || ends.empty() || steps.empty())
                continue;

            const std::vector<uint32_t> sliceShape = sliceNode.getShape();
            std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);

            std::vector<ENodeId> srcEnodes = egraph.getEClass(srcClass).enodes;

            for (ENodeId srcNodeIdx : srcEnodes)
            {
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

                MatchKey key{eNodeIdx, sliceNodeIdx.value, srcNodeIdx.value};
                if (!visited.insert(key).second)
                    continue;

                std::vector<EClassId> newChildren;
                for (uint64_t childIdx = 0; childIdx < opNode.getChildren().size(); ++childIdx)
                {
                    EClassId childId = opNode.getChildren()[childIdx];
                    EClassId canonChildId = egraph.findConst(childId);
                    const EClass childCls = egraph.getEClass(canonChildId);
                    // A box of a fill is a fill of the box's shape, so fused
                    // patterns over fill(...) still match the sliced op. Fill
                    // is a view (zero strides over its scalar value).
                    EClassId fillValue{};
                    for (ENodeId e : childCls.enodes)
                    {
                        const ENode &cn = egraph.getENode(e);
                        if (cn.getOpType() == OpType::FILL && cn.getChildren().size() == 2)
                            fillValue = cn.getChildren()[0];
                    }
                    if (fillValue != EClassId{})
                    {
                        newChildren.push_back(addOpToEGraph(
                            egraph, OpType::FILL,
                            {fillValue, egraph.addIntConst(std::vector<int32_t>(sliceShape.begin(), sliceShape.end()))},
                            sliceShape, std::vector<uint64_t>(sliceShape.size(), 0), childCls.dtype,
                            childCls.mem_space));
                        continue;
                    }
                    std::vector<uint64_t> childSliceStrides = childCls.strides;

                    for (uint64_t d = 0; d < starts.size() && d < childCls.shape.size(); ++d)
                    {
                        int32_t start = starts[d];
                        if (start < 0)
                            start += childCls.shape[d];
                        childSliceStrides[d] *= steps[d];
                    }

                    EClassId childSlice =
                        addOpToEGraph(egraph, OpType::SLICE, {canonChildId, startsId, endsId, stepsId}, sliceShape,
                                      childSliceStrides, childCls.dtype, childCls.mem_space);

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
                                                   sliceContigStrides, childCls.dtype, childCls.mem_space);
                    }
                    newChildren.push_back(childSlice);
                }

                EClassId opEClass = addOpToEGraph(egraph, op, newChildren, sliceShape, sliceContigStrides,
                                                  sliceNode.getDType(), sliceNode.getMemSpace());
                // Keep the standalone result even when a cached update is valid.
                // Extraction must choose an acyclic implementation of either form.
                mergeRestricted(egraph, egraph.getENodeEClass(ENodeId{eNodeIdx}), opEClass);
                srcClass = egraph.findConst(srcClass);
                if (!scatterFormValid(ctx, egraph.getEClass(srcClass),
                                      regionFromBounds(starts, ends, egraph.getEClass(srcClass).shape)))
                    continue;

                const EClass srcEClass = egraph.getEClass(srcClass);
                EClassId shapeId =
                    egraph.addIntConst(std::vector<int32_t>(srcEClass.shape.begin(), srcEClass.shape.end()));
                EClassId scatterClass =
                    addOpToEGraph(egraph, OpType::SCATTER, {opEClass, startsId, endsId, stepsId, shapeId},
                        srcEClass.shape, srcEClass.strides, srcEClass.dtype, srcEClass.mem_space,
                        EClassId(), SourceLocation::current(), "", scatterFormTag(egraph, srcClass));

                egraph.merge(srcClass, scatterClass);
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
        uint32_t eNodeIdx;
        uint32_t childNodeIdx;
        uint32_t srcNodeIdx;
        bool operator==(const MatchKey &o) const
        {
            return eNodeIdx == o.eNodeIdx && childNodeIdx == o.childNodeIdx && srcNodeIdx == o.srcNodeIdx;
        }
    };

    struct MatchKeyHash
    {
        std::uint64_t operator()(const MatchKey &k) const
        {
            return std::hash<uint32_t>{}(k.eNodeIdx) ^ (std::hash<uint32_t>{}(k.childNodeIdx) << 1) ^
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

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::CONTIGUOUS || enode.getChildren().empty())
            return false;

        if (debugEnvTgDebugRoots())
        {
            const EClass &cc = egraph.getEClass(egraph.findConst(enode.getChildren()[0]));
            if (!cc.shape.empty() && cc.shape.back() == 262144)
            {
                std::string ops;
                for (ENodeId c : cc.enodes)
                {
                    const ENode &cn = egraph.getENode(c);
                    ops += toString(cn.getOpType()) + "/";
                    if (cn.getOpType() == OpType::SLICE)
                    {
                        EClassId src = egraph.findConst(cn.getChildren()[0]);
                        for (ENodeId se : egraph.getEClass(src).enodes)
                            ops += "[src " + toString(egraph.getENode(se).getOpType()) + "]";
                        ops += " starts_staged=" + std::to_string(egraph.constantStaging.count(egraph.findConst(cn.getChildren()[1])));
                    }
                }
                LOG(INFO) << "[debug dot match] contig shape=" << toString(cc.shape) << " child ops=" << ops;
            }
        }
        EClassId childClass = egraph.findConst(enode.getChildren()[0]);
        for (ENodeId childNodeIdx : egraph.getEClass(childClass).enodes)
        {
            const ENode &childNode = egraph.getENode(childNodeIdx);
            if (childNode.getOpType() == OpType::SLICE && childNode.getChildren().size() == 4)
            {
                EClassId srcClass = egraph.findConst(childNode.getChildren()[0]);

                if (egraph.constantStaging.count(egraph.findConst(childNode.getChildren()[1])) == 0 ||
                    egraph.constantStaging.count(egraph.findConst(childNode.getChildren()[2])) == 0 ||
                    egraph.constantStaging.count(egraph.findConst(childNode.getChildren()[3])) == 0)
                    continue;

                auto starts = egraph.getConstantInt32(egraph.findConst(childNode.getChildren()[1]));
                auto ends = egraph.getConstantInt32(egraph.findConst(childNode.getChildren()[2]));
                auto steps = egraph.getConstantInt32(egraph.findConst(childNode.getChildren()[3]));

                if (starts.empty() || ends.empty() || steps.empty())
                    continue;

                const auto &origShape = egraph.getEClass(srcClass).shape;
                bool isFull = true;
                if (starts.size() != origShape.size())
                    isFull = false;
                for (uint64_t d = 0; d < starts.size() && isFull; ++d)
                {
                    int32_t st = starts[d] < 0 ? starts[d] + origShape[d] : starts[d];
                    int32_t en = ends[d] < 0 ? ends[d] + origShape[d] : ends[d];
                    if (st != 0 || en != (int32_t)origShape[d] || steps[d] != 1)
                    {
                        isFull = false;
                    }
                }
                if (isFull)
                    continue;

                if (!allowPushDownOnProtected && isEClassProtected(srcClass, ctx.protectedEClasses, egraph))
                    continue;

                for (ENodeId srcNodeIdx : egraph.getEClass(srcClass).enodes)
                {
                    const ENode &opNode = egraph.getENode(srcNodeIdx);
                    if (opNode.getOpType() == OpType::DOT)
                    {
                        MatchKey key{eNodeIdx, childNodeIdx.value, srcNodeIdx.value};
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
        const ENode contigNode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});

        EClassId sliceClass = egraph.findConst(contigNode.getChildren()[0]);

        std::vector<ENodeId> sliceNodes;
        for (ENodeId childNodeIdx : egraph.getEClass(sliceClass).enodes)
        {
            if (egraph.getENode(childNodeIdx).getOpType() == OpType::SLICE &&
                egraph.getENode(childNodeIdx).getChildren().size() == 4)
            {
                sliceNodes.push_back(childNodeIdx);
            }
        }

        for (ENodeId sliceNodeIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENode(sliceNodeIdx);

            EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            std::vector<ENodeId> srcEnodes = egraph.getEClass(srcClass).enodes;

            for (ENodeId srcNodeIdx : srcEnodes)
            {
                const ENode dotNode = egraph.getENode(srcNodeIdx);
                if (dotNode.getOpType() != OpType::DOT)
                    continue;

                MatchKey key{eNodeIdx, sliceNodeIdx.value, srcNodeIdx.value};
                if (!visited.insert(key).second)
                    continue;

                EClassId startsId = sliceNode.getChildren()[1];
                EClassId endsId = sliceNode.getChildren()[2];
                EClassId stepsId = sliceNode.getChildren()[3];

                if (egraph.constantStaging.count(egraph.findConst(startsId)) == 0 ||
                    egraph.constantStaging.count(egraph.findConst(endsId)) == 0 ||
                    egraph.constantStaging.count(egraph.findConst(stepsId)) == 0)
                    continue;

                auto starts = egraph.getConstantInt32(egraph.findConst(startsId));
                auto ends = egraph.getConstantInt32(egraph.findConst(endsId));
                auto steps = egraph.getConstantInt32(egraph.findConst(stepsId));

                if (starts.empty() || ends.empty() || steps.empty())
                    Error::throw_err("[SlicePushDownDot.apply] can't find constants for "
                                     "all slice args");

                bool validSteps = true;
                for (int32_t s : steps)
                {
                    if (s != 1)
                        validSteps = false;
                }
                if (!validSteps)
                    continue;

                std::vector<uint32_t> outClassShape = egraph.getEClass(srcClass).shape;
                uint32_t rank = outClassShape.size();
                if (rank != 2 && rank != 3 && rank != 4)
                    continue;

                while (starts.size() < rank)
                    starts.push_back(0);
                while (ends.size() < rank)
                    ends.push_back(outClassShape[ends.size()]);

                for (uint64_t d = 0; d < rank; ++d)
                {
                    if (starts[d] < 0)
                        starts[d] += outClassShape[d];
                    if (ends[d] < 0)
                        ends[d] += outClassShape[d];
                    starts[d] = std::max(0, starts[d]);
                    ends[d] = std::min((int32_t)outClassShape[d], std::max(starts[d], ends[d]));
                }

                const std::vector<uint32_t> sliceShape = sliceNode.getShape();
                std::vector<uint64_t> sliceContigStrides = calcContiguousStrides(sliceShape);

                EClassId aClassId = dotNode.getChildren()[0];
                EClassId bClassId = dotNode.getChildren()[1];

                uint32_t K = (rank == 2)   ? egraph.getEClass(egraph.findConst(aClassId)).shape[1]
                             : (rank == 3) ? egraph.getEClass(egraph.findConst(aClassId)).shape[2]
                                           : egraph.getEClass(egraph.findConst(aClassId)).shape[3];

                std::vector<int32_t> startsA, endsA, stepsA(rank, 1);
                std::vector<int32_t> startsB, endsB, stepsB(rank, 1);

                if (rank == 2)
                {
                    startsA = {starts[0], 0};
                    endsA = {ends[0], (int32_t)K};

                    startsB = {0, starts[1]};
                    endsB = {(int32_t)K, ends[1]};
                }
                else if (rank == 3)
                {
                    // Reduce only over the columns of A that can be nonzero
                    // for these rows (A is exactly zero elsewhere).
                    int32_t k0 = 0, k1 = (int32_t)K;
                    const EClass aCls = egraph.getEClass(egraph.findConst(aClassId));
                    if (ctx.regions && ctx.regions->facts && aCls.logical_id != LogicalId{})
                    {
                        auto it = ctx.regions->facts->support.find(aCls.logical_id);
                        if (it != ctx.regions->facts->support.end())
                        {
                            Region aBox;
                            aBox.region = {{(uint32_t)starts[0], (uint32_t)ends[0]},
                                           {(uint32_t)starts[1], (uint32_t)ends[1]},
                                           {0, K}};
                            if (auto range = bucket_regions::axisRange(it->second, aBox, 2, false))
                            {
                                k0 = (int32_t)range->start;
                                k1 = (int32_t)range->stop;
                            }
                        }
                    }
                    startsA = {starts[0], starts[1], k0};
                    endsA = {ends[0], ends[1], k1};

                    startsB = {starts[0], k0, starts[2]};
                    endsB = {ends[0], k1, ends[2]};
                }
                else if (rank == 4)
                {
                    startsA = {starts[0], starts[1], starts[2], 0};
                    endsA = {ends[0], ends[1], ends[2], (int32_t)K};
                    startsB = {starts[0], starts[1], 0, starts[3]};
                    endsB = {ends[0], ends[1], (int32_t)K, ends[3]};
                }
                else
                {
                    Error::throw_err("[SlicePushDownDot.apply] expected rank=2,3,4 got rank=" + std::to_string(rank));
                }

                for (size_t d = 0; d < rank; ++d)
                {
                    if (endsA[d] <= startsA[d] || endsB[d] <= startsB[d])
                        Error::throw_err("[SlicePushDownDot.apply] empty operand slice: starts=" + toString(starts) +
                                         " ends=" + toString(ends) + " out shape=" + toString(outClassShape) +
                                         " slice shape=" + toString(sliceNode.getShape()));
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
                    // A box covering the whole operand (e.g. the weight of a
                    // row-restricted projection) is the operand itself.
                    bool whole = st.size() == cls.shape.size();
                    for (size_t d = 0; whole && d < st.size(); ++d)
                        whole = st[d] == 0 && en[d] == static_cast<int32_t>(cls.shape[d]);
                    if (whole)
                        return canonId;
                    std::vector<uint64_t> sStrides = cls.strides;
                    DType cDtype = cls.dtype;

                    std::vector<uint32_t> sShape;
                    for (uint64_t d = 0; d < st.size(); ++d)
                        sShape.push_back(en[d] - st[d]);

                    for (uint64_t d = 0; d < st.size(); ++d)
                    {
                        sStrides[d] *= steps[d];
                    }

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
                if (!scatterFormValid(ctx, egraph.getEClass(srcClass), regionFromBounds(starts, ends, outClassShape)))
                    continue;

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
        uint32_t contigIdx;
        uint32_t sliceIdx;
        uint32_t srcNodeIdx;
        bool operator==(const MatchKey &o) const
        {
            return contigIdx == o.contigIdx && sliceIdx == o.sliceIdx && srcNodeIdx == o.srcNodeIdx;
        }
    };
    struct MatchKeyHash
    {
        std::uint64_t operator()(const MatchKey &k) const
        {
            return (static_cast<uint64_t>(k.contigIdx) * 0x9E3779B97F4A7C15ULL) ^
                   (static_cast<uint64_t>(k.sliceIdx) << 21) ^ (static_cast<uint64_t>(k.srcNodeIdx) << 42) ^
                   k.srcNodeIdx;
        }
    };
    std::unordered_set<MatchKey, MatchKeyHash> visited;
    // Regions of the bucket being saturated, set by apply.
    const BucketRegions *regions_ = nullptr;

    std::string name() const override
    {
        return "SlicePushDownStructural";
    }

    static bool isStructural(OpType op)
    {
        return op == OpType::GATHER || op == OpType::REPEAT || op == OpType::SUM || op == OpType::MAX ||
               op == OpType::RESHAPE || op == OpType::PERMUTE || op == OpType::CONCAT || op == OpType::SLICE ||
               op == OpType::CONTIGUOUS || op == OpType::FILL;
    }

    static bool constInts(const EGraph &egraph, EClassId cid, std::vector<int32_t> &out)
    {
        cid = egraph.findConst(cid);
        if (egraph.constantStaging.count(cid) == 0)
            return false;
        out = egraph.getConstantInt32(cid);
        return true;
    }

    // Box of a slice node in its source's coordinates; false unless it is a
    // proper (non-full) unit-step slice.
    static bool sliceBox(const EGraph &egraph, const ENode &sliceNode, const std::vector<uint32_t> &srcShape,
                         Region &box)
    {
        std::vector<int32_t> starts, ends, steps;
        if (!constInts(egraph, sliceNode.getChildren()[1], starts) ||
            !constInts(egraph, sliceNode.getChildren()[2], ends) || !constInts(egraph, sliceNode.getChildren()[3], steps))
            return false;
        if (starts.size() != srcShape.size() || ends.size() != srcShape.size() || steps.size() != srcShape.size())
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

    static std::vector<uint32_t> boxShape(const Region &box)
    {
        std::vector<uint32_t> shape;
        for (const Dim &d : box.region)
            shape.push_back(d.stop - d.start);
        return shape;
    }

    static Region fullBox(const std::vector<uint32_t> &shape)
    {
        Region r;
        for (uint32_t s : shape)
            r.region.push_back({0, s});
        return r;
    }

    // contiguous(slice(cls, box)), or cls itself when box covers it.
    static EClassId sliceOf(EGraph &egraph, EClassId cid, const Region &box)
    {
        cid = egraph.findConst(cid);
        const EClass cls = egraph.getEClass(cid);
        if (box.region.size() != cls.shape.size())
            Error::throw_err("[SlicePushDownStructural] slice rank mismatch");
        bool full = true;
        std::vector<int32_t> starts, ends, steps;
        for (size_t d = 0; d < box.region.size(); ++d)
        {
            if (box.region[d].start >= box.region[d].stop || box.region[d].stop > cls.shape[d])
                Error::throw_err("[SlicePushDownStructural] slice box outside its tensor");
            starts.push_back((int32_t)box.region[d].start);
            ends.push_back((int32_t)box.region[d].stop);
            steps.push_back(1);
            full = full && box.region[d].start == 0 && box.region[d].stop == cls.shape[d];
        }
        if (full)
            return cid;
        const std::vector<uint32_t> shape = boxShape(box);
        EClassId slice = addOpToEGraph(egraph, OpType::SLICE,
                                       {cid, egraph.addIntConst(starts), egraph.addIntConst(ends), egraph.addIntConst(steps)},
                                       shape, cls.strides, cls.dtype, cls.mem_space);
        return addOpToEGraph(egraph, OpType::CONTIGUOUS, {slice}, shape, calcContiguousStrides(shape), cls.dtype,
                             cls.mem_space);
    }

    // src's op computed only for `box`; an invalid id if the op cannot be
    // restricted to this box.
    EClassId buildRestricted(EGraph &egraph, const ENode op, const EClass src, const Region &box)
    {
        const std::vector<EClassId> children = op.getChildren();
        const std::vector<uint32_t> outShape = boxShape(box);
        auto childCls = [&](size_t i) { return egraph.getEClass(egraph.findConst(children[i])); };
        switch (op.getOpType())
        {
        case OpType::GATHER: {
            const EClass idx = childCls(1);
            const size_t idxRank = idx.shape.size();
            Region idxBox;
            for (size_t d = 0; d < idxRank; ++d)
                idxBox.region.push_back(box.region[d]);
            for (size_t d = idxRank; d < box.region.size(); ++d)
                if (box.region[d].start != 0 || box.region[d].stop != src.shape[d])
                    return EClassId{};
            EClassId idxSlice = sliceOf(egraph, children[1], idxBox);
            return addOpToEGraph(egraph, OpType::GATHER, {children[0], idxSlice}, outShape,
                                 calcContiguousStrides(outShape), src.dtype, src.mem_space);
        }
        case OpType::REPEAT: {
            std::vector<int32_t> axisV;
            if (!constInts(egraph, children[2], axisV) || axisV.size() != 1)
                return EClassId{};
            const EClass in = childCls(0);
            const int32_t axis = axisV[0] < 0 ? axisV[0] + (int32_t)in.shape.size() : axisV[0];
            Region inBox = box;
            if (in.shape[axis] == 1)
                inBox.region[axis] = {0, 1};
            else if (box.region[axis].start == 0 && box.region[axis].stop == src.shape[axis])
                inBox.region[axis] = {0, in.shape[axis]};
            else
                return EClassId{};
            EClassId inSlice = sliceOf(egraph, children[0], inBox);
            const uint32_t reps = in.shape[axis] == 1 ? outShape[axis] : src.shape[axis] / in.shape[axis];
            const std::vector<uint32_t> inShape = boxShape(inBox);
            std::vector<uint64_t> strides = egraph.getEClass(egraph.findConst(inSlice)).strides;
            for (size_t d = 0; d < outShape.size(); ++d)
                if (inShape[d] != outShape[d])
                    strides[d] = 0;
            return addOpToEGraph(egraph, OpType::REPEAT, {inSlice, egraph.addIntConst({(int32_t)reps}), children[2]},
                                 outShape, strides, src.dtype, src.mem_space);
        }
        case OpType::SUM:
        case OpType::MAX: {
            std::vector<int32_t> axisV;
            if (!constInts(egraph, children[1], axisV) || axisV.size() != 1)
                return EClassId{};
            const EClass in = childCls(0);
            const int32_t axis = axisV[0] < 0 ? axisV[0] + (int32_t)in.shape.size() : axisV[0];
            Region inBox = box;
            inBox.region[axis] = {0, in.shape[axis]};
            // Reduce only where the input matters: a sum over the elements
            // that can be nonzero, a max over the unmasked ones (masked ones
            // sit ~1e9 below every unmasked one of their row, and every row
            // must keep one).
            if (regions_ && regions_->facts && in.logical_id != LogicalId{})
            {
                const auto &valid =
                    op.getOpType() == OpType::SUM ? regions_->facts->support : regions_->facts->finite;
                auto it = valid.find(in.logical_id);
                if (it != valid.end())
                    if (auto range = bucket_regions::axisRange(it->second, inBox, axis,
                                                               op.getOpType() == OpType::MAX))
                        inBox.region[axis] = *range;
            }
            EClassId inSlice = sliceOf(egraph, children[0], inBox);
            return addOpToEGraph(egraph, op.getOpType(), {inSlice, children[1]}, outShape,
                                 calcContiguousStrides(outShape), src.dtype, src.mem_space);
        }
        case OpType::RESHAPE: {
            const EClass in = childCls(0);
            if (!isContiguous(in))
                return EClassId{};
            auto prod = [](const std::vector<uint32_t> &s, size_t a, size_t b) {
                uint64_t p = 1;
                for (size_t i = a; i < b; ++i)
                    p *= s[i];
                return p;
            };
            Region inBox = fullBox(in.shape);
            std::vector<int32_t> newShape(src.shape.begin(), src.shape.end());
            for (size_t d = 0; d < box.region.size(); ++d)
            {
                if (box.region[d].start == 0 && box.region[d].stop == src.shape[d])
                    continue;
                bool mapped = false;
                for (size_t e = 0; e < in.shape.size() && !mapped; ++e)
                {
                    if (in.shape[e] == src.shape[d] && prod(in.shape, 0, e) == prod(src.shape, 0, d) &&
                        prod(in.shape, e + 1, in.shape.size()) == prod(src.shape, d + 1, src.shape.size()))
                    {
                        inBox.region[e] = box.region[d];
                        mapped = true;
                    }
                }
                if (!mapped)
                    return EClassId{};
                newShape[d] = (int32_t)(box.region[d].stop - box.region[d].start);
            }
            EClassId inSlice = sliceOf(egraph, children[0], inBox);
            return addOpToEGraph(egraph, OpType::RESHAPE, {inSlice, egraph.addIntConst(newShape)}, outShape,
                                 calcContiguousStrides(outShape), src.dtype, src.mem_space);
        }
        case OpType::PERMUTE: {
            std::vector<int32_t> dims;
            if (!constInts(egraph, children[1], dims) || dims.size() != box.region.size())
                return EClassId{};
            const EClass in = childCls(0);
            Region inBox = fullBox(in.shape);
            for (size_t i = 0; i < dims.size(); ++i)
                inBox.region[dims[i]] = box.region[i];
            EClassId inSlice = sliceOf(egraph, children[0], inBox);
            const std::vector<uint64_t> inStrides = egraph.getEClass(egraph.findConst(inSlice)).strides;
            std::vector<uint64_t> strides(dims.size());
            for (size_t i = 0; i < dims.size(); ++i)
                strides[i] = inStrides[dims[i]];
            return addOpToEGraph(egraph, OpType::PERMUTE, {inSlice, children[1]}, outShape, strides, src.dtype,
                                 src.mem_space);
        }
        case OpType::CONCAT: {
            std::vector<int32_t> axisV;
            if (!constInts(egraph, children[0], axisV) || axisV.size() != 1)
                return EClassId{};
            const int32_t axis = axisV[0] < 0 ? axisV[0] + (int32_t)src.shape.size() : axisV[0];
            std::vector<EClassId> pieces = {children[0]};
            uint32_t offset = 0;
            for (size_t i = 1; i < children.size(); ++i)
            {
                const EClass part = childCls(i);
                const uint32_t lo = offset, hi = offset + part.shape[axis];
                offset = hi;
                const uint32_t st = std::max(lo, box.region[axis].start), en = std::min(hi, box.region[axis].stop);
                if (st >= en)
                    continue;
                Region partBox = box;
                partBox.region[axis] = {st - lo, en - lo};
                pieces.push_back(sliceOf(egraph, children[i], partBox));
            }
            if (pieces.size() == 2)
                return pieces[1];
            return addOpToEGraph(egraph, OpType::CONCAT, pieces, outShape, calcContiguousStrides(outShape), src.dtype,
                                 src.mem_space);
        }
        case OpType::SLICE: {
            std::vector<int32_t> starts, steps;
            if (!constInts(egraph, children[1], starts) || !constInts(egraph, children[3], steps))
                return EClassId{};
            const EClass in = childCls(0);
            if (starts.size() != in.shape.size() || steps.size() != in.shape.size())
                return EClassId{};
            Region inBox = box;
            for (size_t d = 0; d < starts.size(); ++d)
            {
                const int32_t st = starts[d] < 0 ? starts[d] + (int32_t)in.shape[d] : starts[d];
                if (steps[d] != 1 || st < 0)
                    return EClassId{};
                inBox.region[d].start += (uint32_t)st;
                inBox.region[d].stop += (uint32_t)st;
            }
            return sliceOf(egraph, children[0], inBox);
        }
        case OpType::CONTIGUOUS:
            // A box of a contiguous copy is a contiguous copy of the box.
            return sliceOf(egraph, children[0], box);
        case OpType::FILL:
            // A box of a fill is a fill of the box's shape (a view: zero
            // strides over its scalar value).
            return addOpToEGraph(egraph, OpType::FILL,
                                 {children[0], egraph.addIntConst(std::vector<int32_t>(outShape.begin(), outShape.end()))},
                                 outShape, std::vector<uint64_t>(outShape.size(), 0), src.dtype, src.mem_space);
        default:
            return EClassId{};
        }
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::CONTIGUOUS || enode.getChildren().empty())
            return false;
        EClassId sliceClass = egraph.findConst(enode.getChildren()[0]);
        for (ENodeId sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            const ENode &sliceNode = egraph.getENode(sliceIdx);
            if (sliceNode.getOpType() != OpType::SLICE || sliceNode.getChildren().size() != 4)
                continue;
            EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            if (isEClassProtected(srcClass, ctx.protectedEClasses, egraph))
                continue;
            for (ENodeId srcIdx : egraph.getEClass(srcClass).enodes)
            {
                if (!isStructural(egraph.getENode(srcIdx).getOpType()))
                    continue;
                if (!visited.count(MatchKey{eNodeIdx, sliceIdx.value, srcIdx.value}))
                    return true;
            }
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        regions_ = ctx.regions;
        const ENode contigNode = egraph.getENode(ENodeId{eNodeIdx});
        const EClassId sliceClass = egraph.findConst(contigNode.getChildren()[0]);
        const std::vector<ENodeId> sliceNodes = egraph.getEClass(sliceClass).enodes;
        for (ENodeId sliceIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENode(sliceIdx);
            if (sliceNode.getOpType() != OpType::SLICE || sliceNode.getChildren().size() != 4)
                continue;
            EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            const std::vector<ENodeId> srcNodes = egraph.getEClass(srcClass).enodes;
            for (ENodeId srcIdx : srcNodes)
            {
                const ENode op = egraph.getENode(srcIdx);
                if (!isStructural(op.getOpType()))
                    continue;
                if (!visited.insert(MatchKey{eNodeIdx, sliceIdx.value, srcIdx.value}).second)
                    continue;
                const EClass src = egraph.getEClass(egraph.findConst(srcClass));
                Region box;
                if (!sliceBox(egraph, sliceNode, src.shape, box))
                    continue;
                const EClassId restricted = buildRestricted(egraph, op, src, box);
                if (restricted == EClassId{})
                    continue;
                EClassId restrictedContig = egraph.findConst(restricted);
                const EClass rc = egraph.getEClass(restrictedContig);
                if (!isContiguous(rc))
                    restrictedContig = addOpToEGraph(egraph, OpType::CONTIGUOUS, {restrictedContig}, rc.shape,
                                                     calcContiguousStrides(rc.shape), rc.dtype, rc.mem_space);
                mergeRestricted(egraph, egraph.getENodeEClass(ENodeId{eNodeIdx}), restrictedContig);
                srcClass = egraph.findConst(srcClass);
                if (!isContiguous(src) || !scatterFormValid(ctx, src, box))
                    continue;
                std::vector<int32_t> starts, ends, steps;
                for (const Dim &d : box.region)
                {
                    starts.push_back((int32_t)d.start);
                    ends.push_back((int32_t)d.stop);
                    steps.push_back(1);
                }
                const EClassId shapeId =
                    egraph.addIntConst(std::vector<int32_t>(src.shape.begin(), src.shape.end()));
                const EClassId scatterClass = addOpToEGraph(egraph, OpType::SCATTER,
                    {restricted, egraph.addIntConst(starts), egraph.addIntConst(ends), egraph.addIntConst(steps), shapeId},
                    src.shape, src.strides, src.dtype, src.mem_space, EClassId(), SourceLocation::current(),
                    "", scatterFormTag(egraph, srcClass));
                egraph.merge(srcClass, scatterClass);
                srcClass = egraph.findConst(srcClass);
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

    std::unordered_set<std::string> visited;

    std::string name() const override
    {
        return "SlicePushDownFused";
    }

    static std::string key(uint32_t a, uint32_t b, uint32_t c)
    {
        std::string k;
        for (uint32_t v : {a, b, c})
            k.append(reinterpret_cast<const char *>(&v), sizeof(v));
        return k;
    }

    static bool candidate(const EGraph &egraph, ENodeId srcIdx)
    {
        const ENode &op = egraph.getENode(srcIdx);
        return op.getOpType() == OpType::FUSED && op.getKernelId().value != 0 &&
               KernelRegistry::get().hasKernel(op.getKernelId()) && fusedRowLocality(op.getOpName()).rowLocal;
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        const EGraph &egraph = ctx.egraph;
        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
        if (enode.getOpType() != OpType::CONTIGUOUS || enode.getChildren().empty())
            return false;
        const EClassId sliceClass = egraph.findConst(enode.getChildren()[0]);
        for (ENodeId sliceIdx : egraph.getEClass(sliceClass).enodes)
        {
            const ENode &sliceNode = egraph.getENode(sliceIdx);
            if (sliceNode.getOpType() != OpType::SLICE || sliceNode.getChildren().size() != 4)
                continue;
            const EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            for (ENodeId srcIdx : egraph.getEClass(srcClass).enodes)
                if (candidate(egraph, srcIdx) && !visited.count(key(eNodeIdx, sliceIdx.value, srcIdx.value)))
                    return true;
        }
        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        const ENode contigNode = egraph.getENode(ENodeId{eNodeIdx});
        const EClassId sliceClass = egraph.findConst(contigNode.getChildren()[0]);
        const std::vector<ENodeId> sliceNodes = egraph.getEClass(sliceClass).enodes;
        for (ENodeId sliceIdx : sliceNodes)
        {
            const ENode sliceNode = egraph.getENode(sliceIdx);
            if (sliceNode.getOpType() != OpType::SLICE || sliceNode.getChildren().size() != 4)
                continue;
            EClassId srcClass = egraph.findConst(sliceNode.getChildren()[0]);
            const std::vector<ENodeId> srcNodes = egraph.getEClass(srcClass).enodes;
            for (ENodeId srcIdx : srcNodes)
            {
                if (!candidate(egraph, srcIdx) || !visited.insert(key(eNodeIdx, sliceIdx.value, srcIdx.value)).second)
                    continue;
                const ENode op = egraph.getENode(srcIdx);
                const EClass src = egraph.getEClass(egraph.findConst(srcClass));
                Region box;
                if (!SlicePushDownStructural::sliceBox(egraph, sliceNode, src.shape, box))
                    continue;
                const size_t rank = src.shape.size();
                if (rank < 2 || box.region.back().start != 0 || box.region.back().stop != src.shape.back())
                    continue;
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
                        child = egraph.findConst(SlicePushDownStructural::sliceOf(egraph, child, childBox));
                    }
                    newChildren.push_back(child);
                    const EClass &c = egraph.getEClass(child);
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
                if (!isContiguous(src) || !scatterFormValid(ctx, src, box))
                    continue;
                std::vector<int32_t> starts, ends, steps;
                for (const Dim &d : box.region)
                {
                    starts.push_back((int32_t)d.start);
                    ends.push_back((int32_t)d.stop);
                    steps.push_back(1);
                }
                const EClassId shapeId = egraph.addIntConst(std::vector<int32_t>(src.shape.begin(), src.shape.end()));
                const EClassId scatterClass = addOpToEGraph(egraph, OpType::SCATTER,
                    {restricted, egraph.addIntConst(starts), egraph.addIntConst(ends), egraph.addIntConst(steps), shapeId},
                    src.shape, src.strides, src.dtype, src.mem_space, EClassId(), SourceLocation::current(),
                    "", scatterFormTag(egraph, srcClass));
                egraph.merge(srcClass, scatterClass);
                srcClass = egraph.findConst(srcClass);
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
    std::unordered_set<uint32_t> visited;

    std::string name() const override
    {
        return "DotSplitRule";
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (visited.count(eNodeIdx))
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
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

        if (aCls.mem_space != enode.getMemSpace() || bCls.mem_space != enode.getMemSpace())
            return false;

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
        visited.insert(eNodeIdx);

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

    std::unordered_set<uint32_t> visited;

    std::string name() const override
    {
        return "RemoveContiguous";
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (visited.count(eNodeIdx))
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});

        if (enode.getOpType() == OpType::CONTIGUOUS || enode.getOpType() == OpType::INPUT ||
            enode.getOpType() == OpType::CACHE)
            return false;

        // Check if the kernel explicitly requires contiguous inputs for any child
        if (!KernelRegistry::get().hasKernel(enode.getKernelId()))
            return false;
        const KernelEntry kernel = KernelRegistry::get().getKernel(enode.getKernelId());

        if (kernel.is_view)
            return false; // view kernels output strides depend on input strides so will break things when merge eclass

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
        visited.insert(eNodeIdx);
        EGraph &egraph = ctx.egraph;

        const ENode enode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId e_class_id = egraph.getENodeEClass(ENodeId{eNodeIdx});
        const auto children = enode.getChildren();
        const KernelEntry &kernel = KernelRegistry::get().getKernel(enode.getKernelId());

        std::vector<std::vector<EClassId>> candidateChildrenPerPos(children.size());

        for (uint64_t i = 0; i < children.size(); ++i)
        {
            EClassId childClsId = egraph.findConst(children[i]);
            candidateChildrenPerPos[i].push_back(childClsId);
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
                        candidateChildrenPerPos[i].push_back(unwrappedChildId);
                    }
                }
            }
        }

        std::vector<std::vector<EClassId>> childCombinations;
        std::vector<EClassId> currentCombination(children.size());

        std::function<void(uint64_t, bool)> generateCombos = [&](uint64_t pos, bool hasUnwrapped) {
            if (pos == children.size())
            {
                if (hasUnwrapped)
                {
                    childCombinations.push_back(currentCombination);
                }
                return;
            }
            for (uint64_t cIdx = 0; cIdx < candidateChildrenPerPos[pos].size(); ++cIdx)
            {
                currentCombination[pos] = candidateChildrenPerPos[pos][cIdx];
                generateCombos(pos + 1, hasUnwrapped || (cIdx > 0));
            }
        };

        generateCombos(0, false);

        const EClass outCls = egraph.getEClass(egraph.findConst(e_class_id));
        TensorNode outNode;
        outNode.opType = enode.getOpType();
        outNode.opName = enode.getOpName();
        outNode.dtype = outCls.dtype;
        outNode.setShape(outCls.shape);
        outNode.strides = outCls.strides;

        for (const auto &newChildren : childCombinations)
        {
            std::vector<TensorNode> inNodes;
            std::vector<MemSpace> inMemSpaces;
            for (EClassId cid : newChildren)
            {
                const EClass &cCls = egraph.getEClass(egraph.findConst(cid));
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
                egraph.addENode(e_class_id, newENode);
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
        return std::make_unique<RemoveCopyChains>(*this);
    }

    std::unordered_set<uint32_t> visited;

    std::string name() const override
    {
        return "RemoveCopyChains";
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

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (visited.count(eNodeIdx))
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});

        if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
            return false;

        const auto &children = enode.getChildren();
        for (uint64_t i = 0; i < children.size(); ++i)
        {
            EClassId childCanon = egraph.findConst(children[i]);
            std::vector<EClassId> origins = findAllCopyChainOrigins(childCanon, egraph);
            if (!origins.empty())
            {
                return true;
            }
        }

        return false;
    }

    void apply(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        EGraph &egraph = ctx.egraph;
        visited.insert(eNodeIdx);

        const ENode enode = egraph.getENode(ENodeId{eNodeIdx});
        EClassId eClassId = egraph.getENodeEClass(ENodeId{eNodeIdx});

        std::vector<std::vector<EClassId>> child_origin_options;
        child_origin_options.reserve(enode.getChildren().size());
        bool any_has_origin = false;

        for (uint64_t i = 0; i < enode.getChildren().size(); ++i)
        {
            EClassId childCanon = egraph.findConst(enode.getChildren()[i]);
            std::vector<EClassId> origins = findAllCopyChainOrigins(childCanon, egraph);
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
            egraph.addENode(eClassId, newENode);
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

    std::unordered_set<uint32_t> visited_enodes;

    // Every COPY_TO enode seen so far (enodes are append-only): only classes
    // holding one can be a storage weight, so duplicate lookups scan these
    // instead of every class.
    size_t scanned_enodes = 0;
    std::vector<uint32_t> copy_enodes;

    void refreshCopyIndex(const EGraph &egraph)
    {
        const auto &enodes = egraph.getENodes();
        if (enodes.size() < scanned_enodes)
            Error::throw_err("[ConsumerWeightReuseRule] egraph enodes shrank below the indexed count");
        for (; scanned_enodes < enodes.size(); ++scanned_enodes)
            if (enodes[scanned_enodes].getOpType() == OpType::COPY_TO)
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

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (visited_enodes.count(eNodeIdx))
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
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
        visited_enodes.insert(eNodeIdx);

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
        return std::make_unique<RemoveRedundantReshape>(*this);
    }

    std::unordered_set<uint32_t> visited;

    std::string name() const override
    {
        return "RemoveRedundantReshape";
    }

    bool match(uint32_t eNodeIdx, RuleCtx &ctx) override
    {
        if (visited.count(eNodeIdx))
            return false;

        const EGraph &egraph = ctx.egraph;
        if (eNodeIdx >= egraph.getENodes().size())
            return false;

        const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});

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
        visited.insert(eNodeIdx);
        EGraph &egraph = ctx.egraph;
        const ENode enode = egraph.getENode(ENodeId{eNodeIdx});
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
