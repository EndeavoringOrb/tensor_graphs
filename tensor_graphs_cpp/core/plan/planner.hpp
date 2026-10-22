// File: tensor_graphs_cpp/core/plan/planner.hpp
#pragma once

#include <algorithm>
#include <deque>
#include <set>
#include "core/common/sampling_profiler.hpp"
#include <chrono>
#include <iomanip>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/constants.hpp"
#include "core/common/malloc_tuning.hpp"
#include "core/cost_model.hpp"
#include "core/debug.hpp"
#include "core/egraph.hpp"
#include "core/plan/base_equalities.hpp"
#include "core/plan/bucket_regions.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/logging.hpp"
#include "core/misc.hpp"
#include "core/ops/ops.hpp"
#include "core/plan/brancher.hpp"
#include "core/plan/cycles.hpp"
#include "core/plan/egraph_diagnostics.hpp"
#include "core/plan/domain.hpp"
#include "core/plan/propagator.hpp"
#include "core/plan/search_engine.hpp"
#include "core/plan/search_node.hpp"
#include "core/plan/search_state.hpp"
#include "core/plan/selector.hpp"
#include "core/rewrite.hpp"
#include "core/settings.hpp"
#include "core/shape_propagator.hpp"
#include "core/timer.hpp"
#include "core/types.hpp"

using ExtractionResult = plan::ExtractionResult;


struct ENodeDominationContext
{
    const EGraph &egraph;
    const std::vector<ENodeInfo> &enodeInfos;
    const std::unordered_map<MemSpace, uint64_t> &mem_caps;
};

inline bool isENodeMemCapDominated(ENodeId enodeId, const ENodeDominationContext &ctx)
{
    const ENode &enode = ctx.egraph.getENode(enodeId);
    MemSpace ms = enode.getMemSpace();

    if (ms.type == HandleType::STORAGE || ctx.mem_caps.find(ms) == ctx.mem_caps.end())
        return false;

    uint64_t cap = ctx.mem_caps.at(ms);
    uint32_t out_align = plan::SearchState::getDefaultPageAlignment(ms);
    uint64_t out_size = (getStorageSpanBytes(enode.getShape(), enode.getStrides(), enode.getDType()) + out_align - 1) & ~static_cast<uint64_t>(out_align - 1);

    if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
    {
        return out_size > cap;
    }

    const ENodeInfo &info = ctx.enodeInfos[enodeId.value];
    bool can_be_inplace = false;
    if (info.is_view)
    {
        can_be_inplace = true;
    }
    else if (enode.getKernelId().value != 0 && KernelRegistry::get().hasKernel(enode.getKernelId()))
    {
        const auto &k_entry = KernelRegistry::get().getKernel(enode.getKernelId());
        for (uint32_t inplace_idx : k_entry.safe_inplace_idxs)
        {
            if (inplace_idx < enode.getChildren().size())
            {
                EClassId child = ctx.egraph.findConst(enode.getChildren()[inplace_idx]);
                const EClass &cls = ctx.egraph.getEClass(child);
                if (cls.mem_space == ms)
                {
                    uint32_t in_align = plan::SearchState::getDefaultPageAlignment(cls.mem_space);
                    uint64_t in_size = (getStorageSpanBytes(cls.shape, cls.strides, cls.dtype) + in_align - 1) & ~static_cast<uint64_t>(in_align - 1);
                    if (out_size <= in_size)
                    {
                        can_be_inplace = true;
                        break;
                    }
                }
            }
        }
    }

    uint64_t sum_inputs_in_ms = 0;
    std::array<EClassId, 8> seen_children;
    size_t num_seen = 0;
    for (EClassId child : enode.getChildren())
    {
        EClassId canon_child = ctx.egraph.findConst(child);
        bool already_seen = false;
        for (size_t s = 0; s < num_seen; ++s)
        {
            if (seen_children[s] == canon_child)
            {
                already_seen = true;
                break;
            }
        }
        if (!already_seen)
        {
            if (num_seen < seen_children.size())
                seen_children[num_seen++] = canon_child;
            const EClass &cls = ctx.egraph.getEClass(canon_child);
            if (cls.mem_space == ms)
            {
                uint32_t in_align = plan::SearchState::getDefaultPageAlignment(cls.mem_space);
                sum_inputs_in_ms += (getStorageSpanBytes(cls.shape, cls.strides, cls.dtype) + in_align - 1) & ~static_cast<uint64_t>(in_align - 1);
            }
        }
    }

    uint64_t required_mem = (can_be_inplace ? 0 : out_size) + sum_inputs_in_ms;
    return required_mem > cap;
}

inline bool isFasterEquivalentENodeDominated(ENodeId enodeId, const ENodeDominationContext &ctx)
{
    float costA = ctx.enodeInfos[enodeId.value].cost;
    if (costA == TGConstants::INF || std::isnan(costA))
        return false;

    const ENode &a = ctx.egraph.getENode(enodeId);
    EClassId e_class_id = ctx.egraph.getENodeEClass(enodeId);
    const EClass &cls = ctx.egraph.getEClass(ctx.egraph.findConst(e_class_id));
    const ENodeInfo &infoA = ctx.enodeInfos[enodeId.value];

    static const std::vector<uint32_t> empty_inplace;
    const std::vector<uint32_t> *a_inplace = &empty_inplace;
    if (a.getKernelId().value != 0 && KernelRegistry::get().hasKernel(a.getKernelId()))
    {
        a_inplace = &KernelRegistry::get().getKernel(a.getKernelId()).safe_inplace_idxs;
    }

    for (ENodeId otherId : cls.enodes)
    {
        if (otherId == enodeId)
            continue;

        float costB = ctx.enodeInfos[otherId.value].cost;
        if (costB == TGConstants::INF || std::isnan(costB))
            continue;

        const ENode &b = ctx.egraph.getENode(otherId);
        const ENodeInfo &infoB = ctx.enodeInfos[otherId.value];

        if (a.getChildren().size() != b.getChildren().size())
            continue;

        bool same_children = true;
        for (size_t c = 0; c < a.getChildren().size(); ++c)
        {
            if (ctx.egraph.findConst(a.getChildren()[c]) != ctx.egraph.findConst(b.getChildren()[c]))
            {
                same_children = false;
                break;
            }
        }
        if (!same_children)
            continue;

        if (a.getMemSpace() != b.getMemSpace())
            continue;
        if (a.getShape() != b.getShape())
            continue;
        if (a.getStrides() != b.getStrides())
            continue;
        if (a.getDType() != b.getDType())
            continue;
        if (a.getEngines() != b.getEngines())
            continue;
        if (infoA.is_view != infoB.is_view)
            continue;
        if (a.getContentHash() != b.getContentHash())
            continue;

        const std::vector<uint32_t> *b_inplace = &empty_inplace;
        if (b.getKernelId().value != 0 && KernelRegistry::get().hasKernel(b.getKernelId()))
        {
            b_inplace = &KernelRegistry::get().getKernel(b.getKernelId()).safe_inplace_idxs;
        }

        bool inplace_compatible = true;
        for (uint32_t in_idx : *a_inplace)
        {
            if (std::find(b_inplace->begin(), b_inplace->end(), in_idx) == b_inplace->end())
            {
                inplace_compatible = false;
                break;
            }
        }
        if (!inplace_compatible)
            continue;

        if (costB < costA - 1e-9f)
        {
            return true;
        }

        if (std::abs(costA - costB) <= 1e-9f)
        {
            if (b_inplace->size() > a_inplace->size())
            {
                return true;
            }
            if (b_inplace->size() == a_inplace->size() && otherId < enodeId)
            {
                return true;
            }
        }
    }

    return false;
}



inline std::unordered_map<MemSpace, uint64_t>
precomputeReducedMemCaps(const std::unordered_map<MemSpace, uint64_t> &mem_caps,
                         const std::unordered_map<BaseEClassId, ParallelBuffer> &preallocated)
{
    std::unordered_map<MemSpace, uint64_t> reduced_caps = mem_caps;
    std::unordered_map<MemSpace, uint64_t> reserved_per_ms;
    for (const auto &kv : preallocated)
    {
        uint64_t extent = static_cast<uint64_t>(kv.second.offset) + kv.second.size;
        reserved_per_ms[kv.second.mem_space] = std::max(reserved_per_ms[kv.second.mem_space], extent);
    }
    for (const auto &kv : reserved_per_ms)
    {
        auto cap_it = reduced_caps.find(kv.first);
        if (cap_it == reduced_caps.end())
            continue;
        cap_it->second = kv.second >= cap_it->second ? 0 : cap_it->second - kv.second;
    }
    return reduced_caps;
}

struct Planner
{
    CostModel &costModel;
    const Settings &settings;

    EGraph baseState;
    bool baseStateInitialized = false;
    // Rules after the last full-bucket saturation, and the e-graph that
    // saturation produced (set by the caller that keeps it alive).
    std::vector<std::shared_ptr<Rule>> fullSaturationRules_;
    const EGraph *fullSaturationState_ = nullptr;
    SliceMemoTable fullSaturationSliceMemo_;
    std::vector<uint32_t> fullSaturationBaseClassEnodeCounts_;
    std::vector<uint8_t> fullSaturationBaseClassStaged_;
    // Bucket-independent region facts of the graph planAll is planning.
    const RegionFacts *regionFacts_ = nullptr;
    std::vector<LogicalId> cached_topo_;
    LogicalId cached_topo_root_;
    const Graph *cached_topo_graph_ = nullptr;

    // Persistent scratch buffers to eliminate per-bucket vector heap allocations
    std::vector<uint8_t> logical_dirty_;
    std::vector<uint32_t> clean_dirty_child_count_;
    std::vector<uint32_t> clean_parent_head_;
    std::vector<uint32_t> clean_parent_next_;
    std::vector<uint32_t> clean_parent_node_;
    std::vector<uint32_t> clean_queue_;
    std::vector<uint8_t> clean_in_queue_;

    std::vector<uint32_t> saturate_ruleMatchCounts_;
    std::vector<uint8_t> saturate_sweepFilter_;
    std::vector<uint32_t> saturate_classSizeAtSweepStart_;
    std::vector<uint8_t> saturate_stagedAtSweepStart_;
    std::vector<uint32_t> saturate_parent_head_;
    std::vector<uint32_t> saturate_parent_next_;
    std::vector<uint32_t> saturate_parent_node_;
    std::vector<uint8_t> saturate_seenClass_;
    std::vector<uint32_t> saturate_frontier_;
    std::vector<uint32_t> saturate_next_frontier_;
    std::vector<uint32_t> saturate_filtered_base_enodes_;
    std::vector<uint8_t> saturate_rule_active_;
    std::array<std::vector<size_t>, static_cast<size_t>(OpType::FUSED) + 1> saturate_rules_by_op_;
    std::array<std::vector<size_t>, static_cast<size_t>(OpType::FUSED) + 1> saturate_active_rules_by_op_;

    std::vector<uint8_t> death_cascade_enode_valid_;
    std::vector<uint32_t> death_cascade_valid_enode_count_;
    std::vector<uint32_t> death_cascade_parents_head_;
    std::vector<uint32_t> death_cascade_parents_next_;
    std::vector<ENodeId> death_cascade_parents_node_;
    std::vector<EClassId> death_cascade_worklist_;

    struct CostMemo
    {
        std::shared_mutex mutex;
        std::unordered_map<std::string, float> costs;
    };
    std::shared_ptr<CostMemo> cost_memo_; // set only inside planAll, where the cost model is fixed

    Planner(CostModel &costModel, const Settings &settings = Settings::get_default())
        : costModel(costModel), settings(settings)
    {
    }

    void applyDominationRules(const EGraph &egraph, std::vector<ENodeInfo> &enodeInfos)
    {
        ENodeDominationContext ctx{egraph, enodeInfos, settings.mem_caps};
        for (uint32_t i = 0; i < egraph.getENodes().size(); ++i)
        {
            ENodeId enodeId{i};
            if (enodeInfos[i].cost == TGConstants::INF)
                continue;
            if (isENodeMemCapDominated(enodeId, ctx) || isFasterEquivalentENodeDominated(enodeId, ctx))
            {
                enodeInfos[i].cost = TGConstants::INF;
            }
        }
    }

    void preallocate(const Graph &graph, const EGraph &egraph,
                     std::unordered_map<BaseEClassId, ParallelBuffer> &out) const
    {
        out.clear();

        struct PreAllocEntry
        {
            BaseEClassId baseEClassId;
            MemSpace memSpace;
            std::vector<uint32_t> shape;
            std::vector<uint64_t> strides;
            DType dtype;
        };
        std::vector<PreAllocEntry> entries;

        MemSpace storage = MemSpace{0, HandleType::STORAGE};
        MemSpace ram = MemSpace{1, HandleType::CPP};

        auto add_input = [&](const TensorNode &node, LogicalId logicalId) {
            EClassId cid = egraph.findEClassByLogicalId(logicalId);
            if (cid == EClassId{})
                return;
            const EClass &cls = egraph.getEClass(cid);
            if (cls.base_eclass_id == BaseEClassId{} || cls.mem_space == storage)
                return;
            entries.push_back({cls.base_eclass_id, ram, node.getShape(), node.strides, node.dtype});
        };

        for (const auto &pair : graph.nodes)
        {
            const TensorNode &node = pair.second;
            if (node.opType != OpType::INPUT || !graph.input_data_types.count(node.id))
                continue;
            if (graph.input_data_types.at(node.id) == InputDataType::CONSTANT ||
                graph.input_data_types.at(node.id) == InputDataType::RUNTIME)
                add_input(node, node.id);
        }

        if (settings.resident_weights)
            for (const auto &pair : graph.nodes)
                if (isResidentWeightCopy(graph, pair.first))
                    add_input(pair.second, pair.first);

        std::sort(entries.begin(), entries.end(),
                  [](const PreAllocEntry &a, const PreAllocEntry &b) { return a.baseEClassId < b.baseEClassId; });
        entries.erase(std::unique(entries.begin(), entries.end(), [](const PreAllocEntry &a, const PreAllocEntry &b) {
                          return a.baseEClassId == b.baseEClassId;
                      }),
                      entries.end());

        std::unordered_map<MemSpace, uint64_t> cursor;
        BufferId nextId{0};
        for (const auto &e : entries)
        {
            if (e.memSpace == storage)
                continue;

            uint64_t size_bytes = getStorageSpanBytes(e.shape, e.strides, e.dtype);
            if (size_bytes == 0)
                continue;
            uint32_t align = plan::SearchState::getDefaultPageAlignment(e.memSpace);
            size_bytes = (size_bytes + align - 1) & ~static_cast<uint64_t>(align - 1);

            uint64_t offset = cursor[e.memSpace];
            cursor[e.memSpace] = offset + size_bytes;

            ParallelBuffer buf;
            buf.id = nextId++;
            buf.mem_space = e.memSpace;
            buf.size = size_bytes;
            buf.start = 0;
            buf.end = std::numeric_limits<uint32_t>::max();
            buf.offset = static_cast<int64_t>(offset);
            out[e.baseEClassId] = std::move(buf);
        }
    }

    void inferShapes(const std::vector<LogicalId> &topo, Graph &graph)
    {
        ShapePropagator propagator;
        for (LogicalId nodeId : topo)
        {
            propagator.inferShape(nodeId, graph);
        }
    }

    // `seed`: rules that already saturated the e-graph this one was copied
    // from (same enode/class IDs). Their clones skip work already done there.
    // `saturated`: receives the rules after this saturation.
    void saturate(EGraph &egraph, const std::unordered_set<EClassId> &protectedEClasses,
                  bool injected, bool allowPushDownOnProtected = false, TGStore *repo = nullptr,
                  const std::vector<std::shared_ptr<Rule>> *seed = nullptr,
                  std::vector<std::shared_ptr<Rule>> *saturated = nullptr, const BucketRegions *regions = nullptr,
                  size_t initial_unchanged_classes = 0,
                  const std::vector<uint32_t> *initial_class_sizes = nullptr,
                  const std::vector<uint8_t> *initial_staged = nullptr,
                  const SliceMemoTable *seed_slice_memo = nullptr,
                  SliceMemoTable *saved_slice_memo = nullptr)
    {
        SliceMemoTable slice_memo_table;
        if (seed_slice_memo)
            slice_memo_table = *seed_slice_memo;
        RuleCtx ctx{egraph, protectedEClasses, repo, &costModel, regions};
        ctx.slice_memo = &slice_memo_table;
        if (seed && !seed->empty())
        {
            ctx.base_enode_watermark = (*seed)[0]->base_enode_watermark;
            ctx.base_class_watermark = static_cast<uint32_t>(initial_unchanged_classes);
            ctx.base_staged = initial_staged;
        }
        std::vector<std::unique_ptr<Rule>> rules;
        if (seed)
        {
            for (const auto &rule : *seed)
                rules.emplace_back(rule->clone());
            if (regions)
            {
                bool has_scatter_pull_up = false;
                for (const auto &r : rules)
                    if (dynamic_cast<ScatterPullUp *>(r.get()))
                        has_scatter_pull_up = true;
                if (!has_scatter_pull_up)
                    rules.emplace_back(makeProfiledRewriteRule<ScatterPullUp>());
            }
        }
        else
        {
            rules.emplace_back(makeProfiledRewriteRule<FusionRule>(settings.disable_fusion));
            rules.emplace_back(makeProfiledRewriteRule<DotSplitRule>());
            rules.emplace_back(makeProfiledRewriteRule<RemoveContiguous>());
            rules.emplace_back(makeProfiledRewriteRule<RemoveIdentityContiguous>());
            rules.emplace_back(makeProfiledRewriteRule<RemoveCopyChains>());
            rules.emplace_back(makeProfiledRewriteRule<ConsumerWeightReuseRule>());
            rules.emplace_back(makeProfiledRewriteRule<RemoveRedundantReshape>());
            rules.emplace_back(makeProfiledRewriteRule<SlicePushDownElementwise>(allowPushDownOnProtected));
            rules.emplace_back(makeProfiledRewriteRule<SlicePushDownDot>(allowPushDownOnProtected));
            rules.emplace_back(makeProfiledRewriteRule<SlicePushDownStructural>());
            rules.emplace_back(makeProfiledRewriteRule<SlicePushDownFused>());
            if (regions)
                rules.emplace_back(makeProfiledRewriteRule<ScatterPullUp>());
        }
        // Diagnostics: TG_DISABLE_RULES=Name1,Name2 drops those rules.
        if (const char *disabled = std::getenv("TG_DISABLE_RULES"))
        {
            const std::string list = std::string(",") + disabled + ",";
            rules.erase(std::remove_if(rules.begin(), rules.end(),
                                       [&](const std::unique_ptr<Rule> &rule) {
                                           return list.find("," + rule->name() + ",") != std::string::npos;
                                       }),
                        rules.end());
        }

        auto &rules_by_op = saturate_rules_by_op_;
        for (auto &v : rules_by_op)
            v.clear();
        for (size_t op_val = 0; op_val <= static_cast<size_t>(OpType::FUSED); ++op_val)
        {
            const OpType op = static_cast<OpType>(op_val);
            for (size_t rule_idx = 0; rule_idx < rules.size(); ++rule_idx)
            {
                if (rules[rule_idx]->canMatchOp(op))
                    rules_by_op[op_val].push_back(rule_idx);
            }
        }

        std::vector<uint32_t> &ruleMatchCounts = saturate_ruleMatchCounts_;
        ruleMatchCounts.assign(rules.size(), 0);
        // Merge trace diagnostics name the applying rule.
        const bool trace_rule_names = mergeTraceConfig().enabled;
        std::vector<std::string> rule_names;
        if (trace_rule_names)
            for (const auto &rule : rules)
                rule_names.push_back(rule->name());
        uint64_t iterations = 0;
        bool changed = true;
        uint32_t nMatches = 0;
        ProgressTimer timer(0, "saturating");
        const bool time_rules = std::getenv("TG_TIME_RULES") != nullptr;
        std::vector<double> rule_match_seconds(rules.size(), 0.0), rule_apply_seconds(rules.size(), 0.0);
        double rebuild_seconds = 0.0, filter_seconds = 0.0, sweep_seconds = 0.0;

        // FusionRule sweep worklist (see FusionRule::sweep_filter): after each
        // sweep, mark the enodes within its pattern depth above any class
        // that changed during that sweep.
        FusionRule *fusion = nullptr;
        for (auto &rule : rules)
            if (auto *f = dynamic_cast<FusionRule *>(rule.get()))
                fusion = f;
        std::vector<uint8_t> &sweepFilter = saturate_sweepFilter_;
        std::vector<uint32_t> &classSizeAtSweepStart = saturate_classSizeAtSweepStart_;
        std::vector<uint8_t> &stagedAtSweepStart = saturate_stagedAtSweepStart_;
        const std::vector<uint32_t> *currClassSize = &classSizeAtSweepStart;
        const std::vector<uint8_t> *currStaged = &stagedAtSweepStart;
        size_t classesAtSweepStart = 0;
        auto snapshotSweepStart = [&]() {
            classesAtSweepStart = egraph.getNumClasses();
            classSizeAtSweepStart.resize(classesAtSweepStart);
            stagedAtSweepStart.assign(classesAtSweepStart, 0);
            for (size_t c = 0; c < classesAtSweepStart; ++c)
                classSizeAtSweepStart[c] = static_cast<uint32_t>(egraph.getRawEClass(static_cast<uint32_t>(c)).enodes.size());
            for (const auto &[cid, data] : egraph.constantStaging)
                if (cid.value < classesAtSweepStart)
                    stagedAtSweepStart[cid.value] = 1;
            currClassSize = &classSizeAtSweepStart;
            currStaged = &stagedAtSweepStart;
        };

        std::vector<uint32_t> &parent_head = saturate_parent_head_;
        std::vector<uint32_t> &parent_next = saturate_parent_next_;
        std::vector<uint32_t> &parent_node = saturate_parent_node_;
        std::vector<uint8_t> &seenClass = saturate_seenClass_;
        std::vector<uint32_t> &frontier = saturate_frontier_;
        std::vector<uint32_t> &next_frontier = saturate_next_frontier_;
        std::vector<uint32_t> &filtered_base_enodes = saturate_filtered_base_enodes_;

        auto buildSweepFilter = [&]() {
            if (!fusion)
                return;
            fusion->sweep_filter = nullptr;
            ctx.sweep_filter = nullptr;
            const size_t num_classes = egraph.getNumClasses();
            const size_t num_enodes = egraph.getNumENodes();

            parent_head.assign(num_classes, UINT32_MAX);
            parent_next.clear();
            parent_node.clear();
            parent_next.reserve(num_enodes * 2);
            parent_node.reserve(num_enodes * 2);

            for (uint32_t e = 0; e < num_enodes; ++e)
            {
                for (EClassId child : egraph.getENode(ENodeId{e}).getChildren())
                {
                    uint32_t c = egraph.findConst(child).value;
                    if (c < num_classes)
                    {
                        uint32_t edge_idx = static_cast<uint32_t>(parent_next.size());
                        parent_next.push_back(parent_head[c]);
                        parent_node.push_back(e);
                        parent_head[c] = edge_idx;
                    }
                }
            }

            seenClass.assign(num_classes, 0);
            frontier.clear();
            for (size_t c = 0; c < num_classes; ++c)
            {
                if (c >= classesAtSweepStart || egraph.getRawEClass(static_cast<uint32_t>(c)).enodes.size() != (*currClassSize)[c])
                {
                    const uint32_t canon = egraph.findConst(EClassId{static_cast<uint32_t>(c)}).value;
                    if (!seenClass[canon])
                    {
                        seenClass[canon] = 1;
                        frontier.push_back(canon);
                    }
                }
            }
            for (const auto &[cid, data] : egraph.constantStaging)
            {
                if (cid.value < classesAtSweepStart && !(*currStaged)[cid.value])
                {
                    const uint32_t canon = egraph.findConst(cid).value;
                    if (canon < num_classes && !seenClass[canon])
                    {
                        seenClass[canon] = 1;
                        frontier.push_back(canon);
                    }
                }
            }
            sweepFilter.assign(num_enodes, 0);
            filtered_base_enodes.clear();
            for (uint32_t depth = 0; depth < fusion->max_pattern_depth && !frontier.empty(); ++depth)
            {
                next_frontier.clear();
                for (uint32_t c : frontier)
                {
                    for (uint32_t idx = parent_head[c]; idx != UINT32_MAX; idx = parent_next[idx])
                    {
                        uint32_t e = parent_node[idx];
                        if (sweepFilter[e])
                            continue;
                        sweepFilter[e] = static_cast<uint8_t>(std::min<uint32_t>(depth + 1, 255));
                        if (ctx.base_enode_watermark > 0 && e < ctx.base_enode_watermark)
                        {
                            filtered_base_enodes.push_back(e);
                        }
                        const uint32_t owner = egraph.findConst(egraph.getENodeEClass(ENodeId{e})).value;
                        if (!seenClass[owner])
                        {
                            seenClass[owner] = 1;
                            next_frontier.push_back(owner);
                        }
                    }
                }
                frontier = std::move(next_frontier);
            }
            std::sort(filtered_base_enodes.begin(), filtered_base_enodes.end());
            fusion->sweep_filter = &sweepFilter;
            ctx.sweep_filter = &sweepFilter;
        };

        if (seed && initial_unchanged_classes > 0 && initial_class_sizes)
        {
            classesAtSweepStart = initial_unchanged_classes;
            currClassSize = initial_class_sizes;
            if (initial_staged)
                currStaged = initial_staged;
            else
            {
                stagedAtSweepStart.assign(classesAtSweepStart, 0);
                currStaged = &stagedAtSweepStart;
            }
            buildSweepFilter();
        }
        else
        {
            if (fusion)
                fusion->sweep_filter = nullptr;
            ctx.sweep_filter = nullptr;
        }
        size_t last_sweep_enodes_bound = ctx.base_enode_watermark;
        std::vector<uint8_t> &rule_active = saturate_rule_active_;
        rule_active.assign(rules.size(), 1);
        auto &active_rules_by_op = saturate_active_rules_by_op_;
        while (changed)
        {
            iterations++;
            for (size_t r = 0; r < rules.size(); ++r)
                rule_active[r] = rules[r]->prepareSweep(ctx) ? 1 : 0;
            for (size_t op_val = 0; op_val <= static_cast<size_t>(OpType::FUSED); ++op_val)
            {
                active_rules_by_op[op_val].clear();
                for (size_t rule_idx : rules_by_op[op_val])
                {
                    if (rule_active[rule_idx])
                        active_rules_by_op[op_val].push_back(rule_idx);
                }
            }
            snapshotSweepStart();
            uint32_t preUniqueNodes = egraph.getNumUniqueENodes();
            uint32_t preMatches = nMatches;
            auto last_progress = std::chrono::steady_clock::now();
            const auto sweep_start = last_progress;

            auto run_node = [&](uint32_t eNodeIdx) {
                if (time_rules && (eNodeIdx & 0x3FF) == 0 &&
                    std::chrono::steady_clock::now() - last_progress > std::chrono::seconds(5))
                {
                    last_progress = std::chrono::steady_clock::now();
                    std::stringstream ss;
                    ss << "[saturate progress] sweep=" << iterations << " enode=" << eNodeIdx << "/"
                       << egraph.getNumENodes() << " classes=" << egraph.getNumClasses();
                    for (size_t r = 0; r < rules.size(); ++r)
                        if (ruleMatchCounts[r] > 0)
                            ss << " " << rules[r]->name() << "=" << ruleMatchCounts[r];
                    LOG(INFO) << ss.str();
                }
                const ENode &enode = egraph.getENode(ENodeId{eNodeIdx});
                const size_t op_val = static_cast<size_t>(enode.getOpType());
                if (op_val >= active_rules_by_op.size())
                    return;
                const auto &candidate_rules = active_rules_by_op[op_val];
                for (size_t rule_idx : candidate_rules)
                {
                    const auto &rule = rules[rule_idx];
                    // Clock reads cost a kernel call on some platforms and
                    // contend across saturation threads: only when timing.
                    const auto match_start =
                        time_rules ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    bool matched = rule->match(eNodeIdx, ctx);
                    if (time_rules)
                        rule_match_seconds[rule_idx] +=
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - match_start).count();
                    if (!matched)
                        continue;
                    if (trace_rule_names)
                        tg_active_rule = rule_names[rule_idx].c_str();
                    if (!time_rules)
                    {
                        rule->apply(eNodeIdx, ctx);
                        tg_active_rule = nullptr;
                        changed = true;
                        ruleMatchCounts[rule_idx]++;
                        nMatches++;
                        continue;
                    }
                    const auto apply_start = std::chrono::steady_clock::now();
                    const ENode applied = egraph.getENode(ENodeId{eNodeIdx});
                    rule->apply(eNodeIdx, ctx);
                    tg_active_rule = nullptr;
                    {
                        const double seconds =
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - apply_start).count();
                        rule_apply_seconds[rule_idx] += seconds;
                        if (seconds > 1.0)
                            LOG(INFO) << "[saturate slow apply] rule=" << rule->name() << " " << seconds
                                      << "s enode=" << eNodeIdx << " op=" << toString(applied.getOpType()) << ":"
                                      << applied.getOpName() << " shape=" << toString(applied.getShape());
                    }
                    changed = true;
                    ruleMatchCounts[rule_idx]++;
                    nMatches++;
                }
            };

            const size_t sweep_enodes_bound = egraph.getNumENodes();
            if (ctx.sweep_filter && ctx.base_enode_watermark > 0)
            {
                for (uint32_t eNodeIdx : filtered_base_enodes)
                    run_node(eNodeIdx);
                for (uint32_t eNodeIdx = ctx.base_enode_watermark; eNodeIdx < egraph.getNumENodes(); ++eNodeIdx)
                {
                    if (eNodeIdx >= last_sweep_enodes_bound || (eNodeIdx < ctx.sweep_filter->size() && (*ctx.sweep_filter)[eNodeIdx]))
                        run_node(eNodeIdx);
                }
            }
            else
            {
                for (uint32_t eNodeIdx = 0; eNodeIdx < egraph.getNumENodes(); ++eNodeIdx)
                    run_node(eNodeIdx);
            }
            last_sweep_enodes_bound = egraph.getNumENodes();
            sweep_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - sweep_start).count();
            if (nMatches == preMatches)
            {
                changed = false;
                std::stringstream ss;
                ss << "\n--- Saturation Summary (" << iterations << " iterations) ---" << std::endl;
                for (size_t r = 0; r < rules.size(); ++r)
                {
                    if (ruleMatchCounts[r] > 0)
                        ss << "  " << rules[r]->name() << ": " << ruleMatchCounts[r] << " matches\n";
                }
                ss << "Total Matches: " << nMatches;
                LOG(INFO) << ss.str();
                timer.tick();
                break;
            }
            const auto rebuild_start = std::chrono::steady_clock::now();
            egraph.rebuild();
            rebuild_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - rebuild_start).count();
            uint32_t postUniqueNodes = egraph.getNumUniqueENodes();
            changed = preUniqueNodes != postUniqueNodes;
            if (time_rules)
            {
                std::stringstream ss;
                ss << "\n--- Saturation Summary (" << iterations << " iterations) ---" << std::endl;
                for (size_t r = 0; r < rules.size(); ++r)
                {
                    if (ruleMatchCounts[r] > 0)
                        ss << "  " << rules[r]->name() << ": " << ruleMatchCounts[r] << " matches\n";
                }
                ss << "Total Matches: " << nMatches;
                LOG(INFO) << ss.str();
                timer.tick();
            }
            if (time_rules && fusion)
            {
                for (size_t rule_idx = 0; rule_idx < rules.size(); ++rule_idx)
                    if (rules[rule_idx].get() == fusion)
                        LOG(INFO) << "[saturate sweep] sweep=" << iterations << " enodes=" << egraph.getNumENodes()
                                  << " fusion match total=" << rule_match_seconds[rule_idx] << "s";
            }
            const auto filter_start = std::chrono::steady_clock::now();
            buildSweepFilter();
            filter_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - filter_start).count();
        }
        if (fusion)
            fusion->sweep_filter = nullptr;
        ctx.sweep_filter = nullptr;
        if (time_rules)
            std::cout << "[saturate phases] sweeps=" << iterations << " sweep=" << sweep_seconds
                      << "s rebuild=" << rebuild_seconds << "s filter=" << filter_seconds << "s" << std::endl;
        if (time_rules)
            for (size_t rule_idx = 0; rule_idx < rules.size(); ++rule_idx)
                std::cout << "[saturate timing] rule=" << rules[rule_idx]->name()
                          << " match=" << rule_match_seconds[rule_idx] << "s apply=" << rule_apply_seconds[rule_idx]
                          << "s" << std::endl;
        printRewriteProfileSummary();
        if (saved_slice_memo)
            *saved_slice_memo = std::move(slice_memo_table);
        if (saturated)
        {
            saturated->clear();
            for (auto &rule : rules)
                saturated->push_back(std::shared_ptr<Rule>(std::move(rule)));
        }
    }

    uint32_t deathCascade(EGraph &egraph)
    {
        uint32_t numClasses = static_cast<uint32_t>(egraph.getNumClasses());
        uint32_t numEnodes = static_cast<uint32_t>(egraph.getNumENodes());
        std::vector<uint8_t> &enode_valid = death_cascade_enode_valid_;
        enode_valid.assign(numEnodes, 0);
        std::vector<uint32_t> &valid_enode_count = death_cascade_valid_enode_count_;
        valid_enode_count.assign(numClasses, 0);
        std::vector<uint32_t> &parents_head = death_cascade_parents_head_;
        parents_head.assign(numClasses, UINT32_MAX);
        std::vector<uint32_t> &parents_next = death_cascade_parents_next_;
        parents_next.clear();
        parents_next.reserve(numEnodes * 2);
        std::vector<ENodeId> &parents_node = death_cascade_parents_node_;
        parents_node.clear();
        parents_node.reserve(numEnodes * 2);

        for (uint32_t i = 0; i < numClasses; ++i)
        {
            EClassId e_class_id = egraph.find(EClassId{i});
            if (e_class_id != EClassId{i})
                continue;

            const EClass &cls = egraph.getEClass(e_class_id);
            valid_enode_count[e_class_id.value] = static_cast<uint32_t>(cls.enodes.size());

            for (ENodeId enodeId : cls.enodes)
            {
                enode_valid[enodeId.value] = 1;
                const ENode &enode = egraph.getENode(enodeId);
                for (EClassId child : enode.getChildren())
                {
                    EClassId canon_child = egraph.findConst(child);
                    if (canon_child.value < numClasses)
                    {
                        uint32_t edge_idx = static_cast<uint32_t>(parents_next.size());
                        parents_next.push_back(parents_head[canon_child.value]);
                        parents_node.push_back(enodeId);
                        parents_head[canon_child.value] = edge_idx;
                    }
                }
            }
        }

        std::vector<EClassId> &dead_worklist = death_cascade_worklist_;
        dead_worklist.clear();
        for (uint32_t i = 0; i < numClasses; ++i)
        {
            EClassId e_class_id = egraph.find(EClassId{i});
            if (e_class_id != EClassId{i})
                continue;

            if (valid_enode_count[e_class_id.value] == 0)
            {
                dead_worklist.push_back(e_class_id);
            }
        }

        uint32_t cascadePruned = 0;
        while (!dead_worklist.empty())
        {
            EClassId dead_cls = dead_worklist.back();
            dead_worklist.pop_back();

            for (uint32_t edge = parents_head[dead_cls.value]; edge != UINT32_MAX; edge = parents_next[edge])
            {
                ENodeId parent_enode_id = parents_node[edge];
                if (enode_valid[parent_enode_id.value])
                {
                    enode_valid[parent_enode_id.value] = 0;
                    cascadePruned++;

                    EClassId parent_cls = egraph.findConst(egraph.getENodeEClass(parent_enode_id));
                    if (valid_enode_count[parent_cls.value] > 0)
                    {
                        valid_enode_count[parent_cls.value]--;
                        if (valid_enode_count[parent_cls.value] == 0)
                        {
                            dead_worklist.push_back(parent_cls);
                        }
                    }
                }
            }
        }

        // Commit the cascade to class membership; counting invalid expressions
        // alone leaves their references to empty classes in the graph.
        if (cascadePruned > 0)
        {
            for (uint32_t i = 0; i < numClasses; ++i)
            {
                const EClassId class_id{i};
                if (egraph.findConst(class_id) != class_id)
                    continue;
                EClass &cls = egraph.getEClass(class_id);
                const auto end = std::remove_if(cls.enodes.begin(), cls.enodes.end(), [&](ENodeId node_id) {
                    return !enode_valid[node_id.value];
                });
                cls.enodes.erase(end, cls.enodes.end());
            }
        }

        return cascadePruned;
    }

    void pruneEGraph(EGraph &egraph, const std::vector<ENodeInfo> &enodeInfos)
    {
        uint32_t totalPruned = 0;
        for (uint32_t i = 0; i < egraph.getNumClasses(); ++i)
        {
            EClassId e_class_id = egraph.find(EClassId{i});
            if (e_class_id != EClassId{i})
                continue;

            EClass &cls = egraph.getEClass(e_class_id);
            auto it = std::remove_if(cls.enodes.begin(), cls.enodes.end(), [&](ENodeId enodeId) {
                if (enodeId.value >= enodeInfos.size())
                    Error::throw_err("[Planner.pruneEGraph] ENode " + std::to_string(enodeId.value) +
                                     " has no ENodeInfo (" + std::to_string(enodeInfos.size()) + " infos)");
                return enodeInfos[enodeId.value].cost == TGConstants::INF;
            });
            totalPruned += static_cast<uint32_t>(cls.enodes.end() - it);
            cls.enodes.erase(it, cls.enodes.end());
        }

        totalPruned += deathCascade(egraph);

        if (totalPruned > 0)
        {
            LOG(DEBUG) << "[Planner.pruneEGraph] Pruned " << totalPruned
                       << " infinite-cost enodes (including dead-class cascade) from the search space.";
        }
    }

    // Compute both the old subtree DP estimates and the optimistic DAG
    // estimate used by the search brancher.  The latter intentionally uses a
    // single best summary per e-class, so it is a heuristic rather than an
    // exact e-graph extraction.  It is nevertheless optimistic because it
    // ignores scheduling precedence and memory constraints.
    void computeENodeHeuristicCosts(const EGraph &egraph, std::vector<ENodeInfo> &enodeInfos) const
    {
        const uint32_t num_classes = static_cast<uint32_t>(egraph.getNumClasses());
        const uint32_t num_enodes = static_cast<uint32_t>(egraph.getNumENodes());

        // Storage sizes, computed once per class and per enode (the fixed
        // point below re-evaluates enodes many times).
        std::vector<float> class_bytes(num_classes, 0.0f);
        for (uint32_t c = 0; c < num_classes; ++c)
        {
            if (egraph.findConst(EClassId{c}).value != c)
                continue;
            const EClass &cls = egraph.getEClass(EClassId{c});
            class_bytes[c] = static_cast<float>(getStorageSpanBytes(cls.shape, cls.strides, cls.dtype));
        }
        std::vector<float> enode_bytes(num_enodes, 0.0f);
        for (uint32_t e = 0; e < num_enodes; ++e)
        {
            const ENode &enode = egraph.getENode(ENodeId{e});
            enode_bytes[e] = static_cast<float>(getStorageSpanBytes(enode.getShape(), enode.getStrides(), enode.getDType()));
        }
        std::vector<float> eclass_dp_cost(num_classes, TGConstants::INF);
        std::vector<float> eclass_dp_cp_cost(num_classes, TGConstants::INF);
        std::vector<float> eclass_dp_mem(num_classes, TGConstants::INF);

        // Leaf values seed the ordinary DP pass.  All other values are
        // propagated below so cycles without a leaf remain infinite.
        for (uint32_t i = 0; i < num_classes; ++i)
        {
            EClassId cid = egraph.findConst(EClassId{i});
            if (cid.value != i)
                continue;

            for (ENodeId enode_id : egraph.getEClass(cid).enodes)
            {
                if (enode_id.value >= enodeInfos.size())
                    continue;
                const ENode &enode = egraph.getENode(enode_id);
                if (enode.getOpType() != OpType::INPUT && enode.getOpType() != OpType::CACHE)
                    continue;

                const float node_size = static_cast<float>(getStorageSpanBytes(enode.getShape(), enode.getStrides(), enode.getDType()));
                eclass_dp_cost[i] = 0.0f;
                eclass_dp_cp_cost[i] = 0.0f;
                eclass_dp_mem[i] = std::min(eclass_dp_mem[i], node_size);
                enodeInfos[enode_id.value].dp_cost = 0.0f;
                enodeInfos[enode_id.value].dp_cp_cost = 0.0f;
                enodeInfos[enode_id.value].dp_mem = node_size;
            }

        }

        // Least fixed point of the cost/critical-path/memory equations (cycles
        // without a leaf stay INF), by Knuth's generalization of Dijkstra: each
        // equation is superior (an enode's value is at least each child's,
        // costs being non-negative), so the class with the smallest tentative
        // value is final. Every enode is evaluated once per equation, from its
        // children's final values, which is the value the fixed point assigns
        // it; sums run over the children in their original order.
        std::vector<uint32_t> owner_of(num_enodes, UINT32_MAX);
        std::vector<uint32_t> distinct_children(num_enodes, 0);
        std::vector<std::vector<uint32_t>> class_users(num_classes);
        std::vector<uint8_t> can_be_inplace(num_enodes, 0);
        {
            std::vector<uint32_t> distinct;
            for (uint32_t i = 0; i < num_enodes; ++i)
            {
                const float cost = enodeInfos[i].cost;
                if (cost == TGConstants::INF || std::isnan(cost))
                    continue;
                if (cost < 0.0f)
                    Error::throw_err("[Planner.computeENodeHeuristicCosts] negative cost for enode " +
                                     std::to_string(i));
                const ENode &enode = egraph.getENode(ENodeId{i});
                const uint32_t owner = egraph.findConst(egraph.getENodeEClass(ENodeId{i})).value;
                if (owner >= num_classes)
                    continue;
                distinct.clear();
                bool children_known = true;
                for (EClassId child : enode.getChildren())
                {
                    const uint32_t c = egraph.findConst(child).value;
                    if (c >= num_classes)
                        children_known = false;
                    distinct.push_back(c);
                }
                if (!children_known)
                    continue; // never ready, as a child outside the graph has no value
                std::sort(distinct.begin(), distinct.end());
                distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
                owner_of[i] = owner;
                distinct_children[i] = static_cast<uint32_t>(distinct.size());
                for (uint32_t c : distinct)
                    class_users[c].push_back(i);

                bool inplace = enodeInfos[i].is_view;
                if (!inplace && !enode.getChildren().empty() && enode.getKernelId().value != 0 &&
                    KernelRegistry::get().hasKernel(enode.getKernelId()))
                {
                    const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
                    for (uint32_t inplace_idx : kernel.safe_inplace_idxs)
                    {
                        if (inplace_idx >= enode.getChildren().size())
                            continue;
                        EClassId child = egraph.findConst(enode.getChildren()[inplace_idx]);
                        const EClass &child_class = egraph.getEClass(child);
                        if (child_class.mem_space == enode.getMemSpace() && enode_bytes[i] <= class_bytes[child.value])
                        {
                            inplace = true;
                            break;
                        }
                    }
                }
                can_be_inplace[i] = inplace;
            }
        }
        auto solveLeastFixedPoint = [&](std::vector<float> &class_value, auto &&enode_value, auto &&store) {
            std::vector<uint32_t> pending = distinct_children;
            std::vector<uint8_t> finalized(num_classes, 0);
            using Item = std::pair<float, uint32_t>;
            std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
            for (uint32_t c = 0; c < num_classes; ++c)
                if (class_value[c] < TGConstants::INF)
                    heap.push({class_value[c], c});
            auto relax = [&](uint32_t i) {
                const float value = enode_value(i);
                store(i, value);
                const uint32_t owner = owner_of[i];
                if (value < class_value[owner])
                {
                    class_value[owner] = value;
                    heap.push({value, owner});
                }
            };
            for (uint32_t i = 0; i < num_enodes; ++i)
                if (owner_of[i] != UINT32_MAX && distinct_children[i] == 0)
                    relax(i);
            while (!heap.empty())
            {
                const auto [value, c] = heap.top();
                heap.pop();
                if (finalized[c] || value > class_value[c])
                    continue;
                finalized[c] = 1;
                for (uint32_t user : class_users[c])
                    if (--pending[user] == 0)
                        relax(user);
            }
        };
        solveLeastFixedPoint(
            eclass_dp_cost,
            [&](uint32_t i) {
                float total = enodeInfos[i].cost;
                float sum_child_cost = 0.0f;
                for (EClassId child : egraph.getENode(ENodeId{i}).getChildren())
                    sum_child_cost += eclass_dp_cost[egraph.findConst(child).value];
                return total + sum_child_cost;
            },
            [&](uint32_t i, float value) { enodeInfos[i].dp_cost = std::min(enodeInfos[i].dp_cost, value); });
        solveLeastFixedPoint(
            eclass_dp_cp_cost,
            [&](uint32_t i) {
                float max_child_cp_cost = 0.0f;
                for (EClassId child : egraph.getENode(ENodeId{i}).getChildren())
                    max_child_cp_cost = std::max(max_child_cp_cost, eclass_dp_cp_cost[egraph.findConst(child).value]);
                return enodeInfos[i].cost + max_child_cp_cost;
            },
            [&](uint32_t i, float value) { enodeInfos[i].dp_cp_cost = std::min(enodeInfos[i].dp_cp_cost, value); });
        struct ChildMem
        {
            float peak;
            float output;
        };
        std::vector<ChildMem> child_mems;
        solveLeastFixedPoint(
            eclass_dp_mem,
            [&](uint32_t i) {
                const auto &children = egraph.getENode(ENodeId{i}).getChildren();
                if (children.empty())
                    return enode_bytes[i];
                child_mems.clear();
                float sum_child_sizes = 0.0f;
                for (EClassId child : children)
                {
                    const uint32_t c = egraph.findConst(child).value;
                    child_mems.push_back({eclass_dp_mem[c], class_bytes[c]});
                    sum_child_sizes += class_bytes[c];
                }
                std::sort(child_mems.begin(), child_mems.end(), [](const ChildMem &a, const ChildMem &b) {
                    return (a.peak - a.output) > (b.peak - b.output);
                });
                float peak_child_eval = 0.0f;
                float accumulated_outputs = 0.0f;
                for (const ChildMem &child_mem : child_mems)
                {
                    peak_child_eval = std::max(peak_child_eval, accumulated_outputs + child_mem.peak);
                    accumulated_outputs += child_mem.output;
                }
                const float op_memory = sum_child_sizes + (can_be_inplace[i] ? 0.0f : enode_bytes[i]);
                return std::max(peak_child_eval, op_memory);
            },
            [&](uint32_t i, float value) { enodeInfos[i].dp_mem = std::min(enodeInfos[i].dp_mem, value); });

        struct OptimisticSummary
        {
            float cost = TGConstants::INF;
            ENodeId chosen_enode = ENodeId{UINT32_MAX};
            std::vector<uint64_t> covered_bits;
            // Sorted class-bit indices for sparse child-summary traversal.
            std::vector<uint32_t> covered_class_bits;
            // Chosen enode per covered_class_bits entry, kept parallel and
            // sorted so child merges never touch a num_canonical-sized array.
            std::vector<ENodeId> covered_enodes;
            // Dense per-engine work accumulator indexed by engine ordinal.
            std::vector<float> engine_work;
            // Engine ordinals with nonzero work, so clears and makespan scans
            // skip the untouched majority.
            std::vector<uint32_t> used_engines;
            bool valid = false;
        };

        struct CandidateCostCache
        {
            std::vector<uint64_t> child_versions;
            // Full last-built candidate summary, kept so an accepted
            // candidate can be adopted without a rebuild. Empty on misses.
            OptimisticSummary summary;
            float cost = TGConstants::INF;
            bool valid = false;
            bool initialized = false;
        };

        std::vector<EClassId> canonical_classes;
        std::vector<uint32_t> class_to_bit(num_classes, UINT32_MAX);
        for (uint32_t i = 0; i < num_classes; ++i)
        {

            EClassId cid = egraph.findConst(EClassId{i});
            if (cid.value != i)
                continue;
            class_to_bit[i] = static_cast<uint32_t>(canonical_classes.size());
            canonical_classes.push_back(cid);
        }

        const uint32_t num_canonical = static_cast<uint32_t>(canonical_classes.size());
        if (num_canonical > settings.optimistic_dag_max_classes)
            return; // optimistic_dag_cost stays INF: branch ordering uses dp_cost
        const uint32_t bit_words = (num_canonical + 63) >> 6;
        const ENodeId invalid_enode{UINT32_MAX};
        auto bit_test = [](const std::vector<uint64_t> &bits, uint32_t bit) {
            return (bits[bit >> 6] & (1ULL << (bit & 63))) != 0;
        };
        auto add_bits = [](std::vector<uint64_t> &bits, uint32_t bit) {
            bits[bit >> 6] |= 1ULL << (bit & 63);
        };
        // Precompute per-enode engine-work data once. Candidate construction
        // accumulates this on every merged bit, so the hot path must not
        // revisit egraph/enodeInfos or hash Engine keys.
        struct EngineKeyLess
        {
            bool operator()(const Engine &a, const Engine &b) const { return a < b; }
        };
        std::map<Engine, uint32_t, EngineKeyLess> engine_ordinal;
        struct ENodeWork
        {
            float cost = TGConstants::INF;
            uint32_t begin = 0;
            uint32_t count = 0;
        };
        std::vector<ENodeWork> enode_work(num_enodes);
        std::vector<uint32_t> enode_engine_ids;
        for (uint32_t i = 0; i < num_enodes; ++i)
        {
            const float cost = enodeInfos[i].cost;
            if (cost == TGConstants::INF || std::isnan(cost))
                continue;
            ENodeWork &work = enode_work[i];
            work.cost = cost;
            work.begin = static_cast<uint32_t>(enode_engine_ids.size());
            for (const Engine &engine : egraph.getENode(ENodeId{i}).getEngines())
            {
                auto inserted = engine_ordinal.emplace(
                    engine, static_cast<uint32_t>(engine_ordinal.size()));
                enode_engine_ids.push_back(inserted.first->second);
                ++work.count;
            }
        }
        const uint32_t num_engines = static_cast<uint32_t>(engine_ordinal.size());
        auto add_enode_work = [&](OptimisticSummary &summary, ENodeId enode_id) {
            if (enode_id.value >= num_enodes)
                return;
            const ENodeWork &work = enode_work[enode_id.value];
            if (work.cost == TGConstants::INF)
                return;
            if (summary.engine_work.size() < num_engines)
                summary.engine_work.resize(num_engines, 0.0f);
            for (uint32_t k = 0; k < work.count; ++k)
            {
                const uint32_t ordinal = enode_engine_ids[work.begin + k];
                float &accumulated = summary.engine_work[ordinal];
                if (accumulated == 0.0f)
                    summary.used_engines.push_back(ordinal);
                accumulated += work.cost;
            }
        };
        auto makespan = [](const OptimisticSummary &summary) {
            float result = 0.0f;
            for (uint32_t ordinal : summary.used_engines)
                result = std::max(result, summary.engine_work[ordinal]);
            return result;
        };

        std::vector<OptimisticSummary> optimistic(num_classes);

        // Canonicalize and deduplicate each enode's children once. Candidate
        // construction revisits the same enodes many times during propagation.
        std::vector<std::vector<EClassId>> canonical_children(num_enodes);
        for (uint32_t i = 0; i < num_enodes; ++i)
        {
            const auto &children = egraph.getENode(ENodeId{i}).getChildren();
            auto &canonical = canonical_children[i];
            canonical.reserve(children.size());
            for (EClassId child : children)
                canonical.push_back(egraph.findConst(child));
            std::sort(canonical.begin(), canonical.end());
            canonical.erase(std::unique(canonical.begin(), canonical.end()), canonical.end());
        }

        std::vector<std::vector<EClassId>> parent_map(num_classes);
        for (EClassId cid : canonical_classes)
        {
            for (ENodeId enode_id : egraph.getEClass(cid).enodes)
            {
                for (EClassId canonical_child : canonical_children[enode_id.value])
                {
                    if (canonical_child.value < num_classes)
                    {
                        parent_map[canonical_child.value].push_back(cid);
                    }
                }
            }
            
        }
        for (auto &parents : parent_map)
        {
            std::sort(parents.begin(), parents.end());
            parents.erase(std::unique(parents.begin(), parents.end()), parents.end());
        }

        // Each enode candidate depends on its fixed cost and its children's
        // current summaries. Versioning those summaries lets us reuse the
        // candidate's scalar result when a worklist revisit changed unrelated
        // classes.
        std::vector<CandidateCostCache> candidate_cost_cache(num_enodes);
        std::vector<uint64_t> summary_versions(num_classes, 0);
        auto candidate_cache_is_current = [&](ENodeId enode_id) {
            const CandidateCostCache &cache = candidate_cost_cache[enode_id.value];
            const auto &children = canonical_children[enode_id.value];
            if (!cache.initialized || cache.child_versions.size() != children.size())
                return false;
            for (size_t i = 0; i < children.size(); ++i)
            {
                if (cache.child_versions[i] != summary_versions[children[i].value])
                    return false;
            }
            return true;
        };
        auto update_candidate_cache = [&](ENodeId enode_id, bool valid, float cost,
                                          OptimisticSummary &candidate) {
            CandidateCostCache &cache = candidate_cost_cache[enode_id.value];
            const auto &children = canonical_children[enode_id.value];
            cache.child_versions.resize(children.size());
            for (size_t i = 0; i < children.size(); ++i)
                cache.child_versions[i] = summary_versions[children[i].value];
            cache.cost = cost;
            cache.valid = valid;
            // Move the built summary into the cache so an accepted candidate
            // can be adopted later without rebuilding it.
            if (valid)
                std::swap(cache.summary, candidate);
            else
                cache.summary.covered_class_bits.clear();
            cache.initialized = true;
        };

        auto build_candidate = [&](EClassId cid, ENodeId enode_id, OptimisticSummary &candidate) {
            if (enode_id.value >= enodeInfos.size() || enodeInfos[enode_id.value].cost == TGConstants::INF ||
                std::isnan(enodeInfos[enode_id.value].cost))
                return false;

            const uint32_t class_bit = class_to_bit[cid.value];
            if (class_bit == UINT32_MAX)
                return false;

            candidate.cost = TGConstants::INF;
            candidate.chosen_enode = invalid_enode;
            candidate.valid = false;
            candidate.covered_bits.assign(bit_words, 0);
            if (candidate.engine_work.size() != num_engines)
                candidate.engine_work.assign(num_engines, 0.0f);
            else
            {
                for (uint32_t ordinal : candidate.used_engines)
                    candidate.engine_work[ordinal] = 0.0f;
            }
            candidate.used_engines.clear();
            add_bits(candidate.covered_bits, class_bit);
            add_enode_work(candidate, enode_id);

            for (EClassId child : canonical_children[enode_id.value])
            {
                if (child.value >= num_classes || !optimistic[child.value].valid)
                {
                    return false;
                }
                const OptimisticSummary &child_summary = optimistic[child.value];
                if (class_to_bit[child.value] == UINT32_MAX || bit_test(child_summary.covered_bits, class_bit))
                {
                    return false;
                }

                // covered_class_bits is sorted, so this visits the same new
                // bits in the same ascending order as the former word scan,
                // while skipping words and bits absent from the child.
                // covered_enodes is parallel to it, so covered bits never
                // carry the invalid_enode sentinel.
                size_t capacity_hint = candidate.covered_class_bits.size();
                if (!child_summary.covered_class_bits.empty())
                {
                    capacity_hint += child_summary.covered_class_bits.back();
                }
                candidate.covered_class_bits.reserve(capacity_hint);
                candidate.covered_enodes.reserve(capacity_hint);
                for (size_t k = 0; k < child_summary.covered_class_bits.size(); ++k)
                {
                    const uint32_t selected_bit = child_summary.covered_class_bits[k];
                    if (bit_test(candidate.covered_bits, selected_bit))
                        continue;

                    const ENodeId selected_enode = child_summary.covered_enodes[k];
                    candidate.covered_class_bits.push_back(selected_bit);
                    candidate.covered_enodes.push_back(selected_enode);
                    add_enode_work(candidate, selected_enode);
                    add_bits(candidate.covered_bits, selected_bit);
                }
            }

            // The candidate's own class_bit is the smallest covered bit, so
            // the child appends above keep covered_class_bits sorted without
            // rescanning the bitset.
            candidate.covered_class_bits.insert(candidate.covered_class_bits.begin(), class_bit);
            candidate.covered_enodes.insert(candidate.covered_enodes.begin(), enode_id);

            candidate.cost = makespan(candidate);
            candidate.chosen_enode = enode_id;
            candidate.valid = true;
            return true;
        };

        std::vector<EClassId> worklist = canonical_classes;
        std::vector<EClassId> next_worklist;
        std::vector<bool> in_queue(num_classes, true);
        OptimisticSummary candidate_scratch;
        ProgressTimer timer(0, "calculating optimistic cost");
        while (!worklist.empty())
        {
            const size_t round_classes = worklist.size();
            for (EClassId cid : worklist)
            {
                in_queue[cid.value] = false;
                float best_cost = TGConstants::INF;
                ENodeId best_enode = invalid_enode;
                bool best_valid = false;
                for (ENodeId enode_id : egraph.getEClass(cid).enodes)
                {
                    CandidateCostCache &cache = candidate_cost_cache[enode_id.value];
                    if (!candidate_cache_is_current(enode_id))
                    {
                        const bool valid = build_candidate(cid, enode_id, candidate_scratch);
                        update_candidate_cache(enode_id, valid, valid ? candidate_scratch.cost : TGConstants::INF,
                                               candidate_scratch);
                    }
                    enodeInfos[enode_id.value].optimistic_dag_cost = cache.valid ? cache.cost : TGConstants::INF;
                    if (!cache.valid)
                        continue;
                    if (!best_valid || cache.cost < best_cost - 1e-6f ||
                        (std::abs(cache.cost - best_cost) <= 1e-6f && enode_id < best_enode))
                    {
                        best_valid = true;
                        best_cost = cache.cost;
                        best_enode = enode_id;
                    }

                }

                OptimisticSummary &current = optimistic[cid.value];
                if (best_valid &&
                    (!current.valid || best_cost < current.cost - 1e-6f ||
                     (std::abs(best_cost - current.cost) <= 1e-6f && best_enode < current.chosen_enode)))
                {
                    // Adopt the candidate summary the cache already holds
                    // instead of building it a second time. The old summary
                    // (or the emptied cache slot) becomes scratch storage.
                    CandidateCostCache &best_cache = candidate_cost_cache[best_enode.value];
                    if (!best_cache.valid || best_cache.cost != best_cost)
                        Error::throw_err("Optimistic candidate cache did not reproduce its selected candidate");
                    std::swap(current, best_cache.summary);
                    if (candidate_scratch.covered_class_bits.empty() &&
                        candidate_scratch.covered_bits.capacity() < best_cache.summary.covered_bits.capacity())
                        std::swap(candidate_scratch, best_cache.summary);
                    ++summary_versions[cid.value];
                    for (EClassId parent : parent_map[cid.value])
                    {
                        if (!in_queue[parent.value])
                        {
                            in_queue[parent.value] = true;
                            next_worklist.push_back(parent);
                        }
                    }
                }
            }
            worklist.clear();
            std::swap(worklist, next_worklist);
            timer.tick();
        }

        // Recompute every enode after the fixed point.  During propagation an
        // enode can have been visited before one of its children improved.
        for (EClassId cid : canonical_classes)
        {
            for (ENodeId enode_id : egraph.getEClass(cid).enodes)
            {
                CandidateCostCache &cache = candidate_cost_cache[enode_id.value];
                if (!candidate_cache_is_current(enode_id))
                {
                    const bool valid = build_candidate(cid, enode_id, candidate_scratch);
                    update_candidate_cache(enode_id, valid, valid ? candidate_scratch.cost : TGConstants::INF,
                                           candidate_scratch);
                }
                enodeInfos[enode_id.value].optimistic_dag_cost = cache.valid ? cache.cost : TGConstants::INF;
            }
        }
    }

    // estimateCost is a pure function of its arguments; the buckets of one
    // planAll share most e-nodes, so remember each distinct query.
    float memoizedCost(KernelId kernel_id, const std::vector<uint32_t> &out_shape,
                       const std::vector<uint64_t> &out_strides, DType out_dtype,
                       const std::vector<std::vector<uint32_t>> &in_shapes,
                       const std::vector<std::vector<uint64_t>> &in_strides, const std::vector<DType> &in_dtypes,
                       const std::vector<std::vector<uint8_t>> &in_constants)
    {
        if (!cost_memo_ || costModel.callObserver) // observers expect every call
            return costModel.estimateCost(kernel_id, out_shape, out_strides, out_dtype, in_shapes, in_strides,
                                          in_dtypes, in_constants);
        std::string key;
        auto put = [&](const void *data, size_t bytes) { key.append(static_cast<const char *>(data), bytes); };
        auto putVec = [&](const auto &v) {
            const uint64_t n = v.size();
            put(&n, sizeof(n));
            if (n)
                put(v.data(), n * sizeof(v[0]));
        };
        put(&kernel_id.value, sizeof(kernel_id.value));
        putVec(out_shape);
        putVec(out_strides);
        put(&out_dtype, sizeof(out_dtype));
        const uint64_t ni = in_shapes.size();
        put(&ni, sizeof(ni));
        for (size_t i = 0; i < in_shapes.size(); ++i)
        {
            putVec(in_shapes[i]);
            putVec(in_strides[i]);
            put(&in_dtypes[i], sizeof(in_dtypes[i]));
            if (isScatterKernelControlInput(kernel_id, i) || i >= in_constants.size())
            {
                static const std::vector<uint8_t> emptyVec;
                putVec(emptyVec);
            }
            else
            {
                putVec(in_constants[i]);
            }
        }
        {
            std::shared_lock<std::shared_mutex> lock(cost_memo_->mutex);
            auto it = cost_memo_->costs.find(key);
            if (it != cost_memo_->costs.end())
                return it->second;
        }
        const float cost = costModel.estimateCost(kernel_id, out_shape, out_strides, out_dtype, in_shapes, in_strides,
                                                  in_dtypes, in_constants);
        std::unique_lock<std::shared_mutex> lock(cost_memo_->mutex);
        cost_memo_->costs.emplace(std::move(key), cost);
        return cost;
    }

    std::vector<ENodeInfo> computeENodeInfos(const EGraph &egraph, bool compute_heuristics = true)
    {
        std::vector<ENodeInfo> enodeInfos(egraph.getNumENodes());

        // Enodes are independent; planAll's cost memo is shared under a lock
        // (reduction workers already query it concurrently). Without the memo,
        // or with a cost-call observer (expects calls in order), stay serial.
        const uint32_t num_enodes = static_cast<uint32_t>(egraph.getNumENodes());
        const uint32_t num_chunks = (!cost_memo_ || costModel.callObserver)
                                        ? 1u
                                        : std::min<uint32_t>(64u, std::max(1u, num_enodes / 4096));
        auto computeChunk = [&](uint32_t chunk) {
        for (uint32_t i = chunk * num_enodes / num_chunks; i < (chunk + 1) * num_enodes / num_chunks; ++i)
        {
            const ENode &enode = egraph.getENode(ENodeId{i});
            ENodeInfo info;
            info.is_view = false;
            info.dp_cost = TGConstants::INF;

            if (enode.getKernelId() != KernelId{0})
            {
                const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
                info.is_view = kernel.is_view;
            }

            if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
            {
                info.cost = 0.0f;
            }
            else if (enode.getKernelId() != KernelId{0})
            {
                std::vector<std::vector<uint32_t>> inShapes;
                std::vector<std::vector<uint64_t>> inStrides;
                std::vector<DType> inDTypes;
                std::vector<std::vector<uint8_t>> inConstants;

                inShapes.reserve(enode.getChildren().size());
                inStrides.reserve(enode.getChildren().size());
                inDTypes.reserve(enode.getChildren().size());
                inConstants.reserve(enode.getChildren().size());

                for (uint64_t j = 0; j < enode.getChildren().size(); j++)
                {
                    EClassId childEClassId = enode.getChildren()[j];
                    const EClass &childCls = egraph.getEClass(egraph.findConst(childEClassId));
                    inShapes.push_back(childCls.shape);
                    inStrides.push_back(childCls.strides);
                    inDTypes.push_back(childCls.dtype);
                    EClassId canonChild = egraph.findConst(childEClassId);
                    if (egraph.constantStaging.count(canonChild))
                    {
                        inConstants.push_back(*egraph.constantStaging.at(canonChild));
                    }
                    else
                    {
                        inConstants.push_back({});
                    }
                }

                info.cost = memoizedCost(enode.getKernelId(), enode.getShape(), enode.getStrides(), enode.getDType(),
                                         inShapes, inStrides, inDTypes, inConstants);
            }
            else
            {
                info.cost = TGConstants::INF;
            }

            if (settings.cpu_only && enode.getOpType() != OpType::INPUT &&
                enode.getOpType() != OpType::CACHE)
            {
                const auto &engines = enode.getEngines();
                const bool cpu_only = engines.empty() ||
                    std::all_of(engines.begin(), engines.end(), [](const Engine &engine) {
                        return engine.type == EngineType::CPU;
                    });
                if (!cpu_only)
                    info.cost = TGConstants::INF;
            }

            enodeInfos[i] = info;
        }
        };
        if (num_chunks == 1)
            computeChunk(0);
        else
            ThreadPool::get().parallel_for(num_chunks, computeChunk);

        applyDominationRules(egraph, enodeInfos);

        if (compute_heuristics)
            computeENodeHeuristicCosts(egraph, enodeInfos);

        return enodeInfos;
    }

  public:
    // copy_to(weight from storage): with Settings::resident_weights its class
    // is a preallocated input holding the weight, written at session load.
    static bool isResidentWeightCopy(const Graph &graph, LogicalId id)
    {
        const TensorNode &node = graph.getNode(id);
        if (node.opType != OpType::COPY_TO || node.child_ids.size() != 1)
            return false;
        const TensorNode &src = graph.getNode(node.child_ids[0]);
        return src.opType == OpType::INPUT && graph.input_data_types.count(src.id) &&
               graph.getInputDataType(src.id) == InputDataType::STORAGE;
    }

    void initBaseEGraph(LogicalId rootId, Graph &graph, const std::vector<LogicalId> &topo, TGStore *repo = nullptr,
                        bool doSaturate = true)
    {
        if (KernelRegistry::get().nKernels() == 0)
        {
            Error::throw_err("KernelRegistry has 0 registered kernels!");
        }
        if (baseStateInitialized)
            return;

        inferShapes(topo, graph);

        MemSpace storage = MemSpace{0, HandleType::STORAGE};
        MemSpace ram = MemSpace{1, HandleType::CPP};
        Engine cpu = Engine{0, EngineType::CPU};

        for (LogicalId nodeId : topo)
        {
            TensorNode &node = graph.getNode(nodeId);
            MemSpace mem_space = ram;
            if (node.opType == OpType::INPUT && graph.getInputDataType(nodeId) == InputDataType::STORAGE)
            {
                mem_space = storage;
            }
            EClassId e_class_id = baseState.addEClass(node.getShape(), node.strides, node.dtype, mem_space, nodeId);
            if (graph.constantStaging.count(nodeId))
            {
                baseState.constantStaging[e_class_id] = graph.constantStaging.at(nodeId);
                uint64_t dataHash = tg_hash::computeConstantHash(node.getShape(), node.strides, node.dtype,
                                                                 *graph.constantStaging.at(nodeId));
                baseState.constantHashIndex[dataHash].push_back(e_class_id);
            }
        }
        baseState.root_class_id = baseState.findConst(baseState.findEClassByLogicalId(rootId));

        std::unordered_map<EClassId, EClassId> contiguousInputs;
        for (LogicalId nodeId : topo)
        {
            const TensorNode &node = graph.getNode(nodeId);
            EClassId e_class_id = baseState.findEClassByLogicalId(nodeId);

            if (node.opType == OpType::INPUT)
            {
                std::vector<EClassId> children;
                for (LogicalId pid : node.child_ids)
                    children.push_back(baseState.findEClassByLogicalId(pid));

                std::string contentHash = node.contentHash;
                if (graph.getInputDataType(nodeId) == InputDataType::RUNTIME)
                    contentHash = toString(nodeId);

                ENode enode =
                    ENode(KernelId{0}, node.opType, node.opName, children, node.getShape(), node.strides, node.dtype,
                          graph.getInputDataType(nodeId) == InputDataType::STORAGE ? storage : ram, {cpu}, contentHash,
                          0, node.debugOrigin);
                baseState.addENode(e_class_id, enode);
                continue;
            }

            std::vector<TensorNode> inputs;
            std::vector<MemSpace> input_mem_spaces;
            for (LogicalId pid : node.child_ids)
            {
                inputs.push_back(graph.getNode(pid));
                EClassId pid_eclass = baseState.findEClassByLogicalId(pid);
                input_mem_spaces.push_back(baseState.getEClass(pid_eclass).mem_space);
            }

            bool ignore_in_ms = (node.opType != OpType::COPY_TO);
            std::vector<KernelId> refs =
                KernelRegistry::get().findMatchingKernels(node.opType, node.opName, inputs, node, true, ram,
                                                          input_mem_spaces, {cpu}, false, ignore_in_ms, false, true);

            if (refs.empty())
            {
                std::stringstream details;
                details << "[Planner.initBaseEGraph] couldn't find any kernels to init EClass "
                        << toString(e_class_id) << "\n"
                        << "No registered kernel matched this node under the planner's current constraints.\n"
                        << "Node details:\n"
                        << "  Logical ID: " << nodeId.value << "\n"
                        << "  Op type:    " << toString(node.opType) << "\n"
                        << "  Op name:    " << (node.opName.empty() ? "N/A" : node.opName) << "\n"
                        << "  Debug origin: " << (node.debugOrigin.empty() ? "N/A" : node.debugOrigin) << "\n"
                        << "  Output dtype: " << toString(node.dtype) << "\n"
                        << "  Output shape: " << toString(node.getShape()) << "\n"
                        << "  Output strides: " << toString(node.strides) << "\n"
                        << "  Inputs (" << inputs.size() << "):\n";

                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    details << "    [" << i << "] Logical ID: " << node.child_ids[i].value << "\n"
                            << "        Op type: " << toString(inputs[i].opType) << "\n"
                            << "        Op name: " << (inputs[i].opName.empty() ? "N/A" : inputs[i].opName)
                            << "\n"
                            << "        Debug origin: "
                            << (inputs[i].debugOrigin.empty() ? "N/A" : inputs[i].debugOrigin) << "\n"
                            << "        Dtype: " << toString(inputs[i].dtype) << "\n"
                            << "        Shape: " << toString(inputs[i].getShape()) << "\n"
                            << "        Strides: " << toString(inputs[i].strides) << "\n"
                            << "        Memory space: " << toString(input_mem_spaces[i]) << "\n";
                }

                const auto &registered_kernels = KernelRegistry::get().getAllKernels();
                size_t same_op_kernel_count = 0;
                for (const auto &[kernel_id, kernel] : registered_kernels)
                {
                    if (kernel.opType != node.opType || kernel.opName != node.opName)
                        continue;

                    ++same_op_kernel_count;
                    details << "Registered kernel candidate " << toString(kernel_id) << ":\n"
                            << "  Name: " << kernel.getName() << "\n"
                            << "  Input count: " << kernel.min_num_inputs << ".." << kernel.max_num_inputs << "\n"
                            << "  Reference kernel: " << (kernel.isReference ? "true" : "false") << "\n"
                            << "  Output memory space: " << toString(kernel.output_mem_space) << "\n"
                            << "  Required engines: [";
                    for (size_t i = 0; i < kernel.engines.size(); ++i)
                    {
                        if (i > 0)
                            details << ", ";
                        details << toString(kernel.engines[i]);
                    }
                    details << "]\n  Dtypes: [";
                    for (size_t i = 0; i < kernel.dtypes.size(); ++i)
                    {
                        if (i > 0)
                            details << ", ";
                        details << toString(kernel.dtypes[i]);
                    }
                    details << "]\n  Required input memory spaces: [";
                    for (size_t i = 0; i < kernel.input_mem_spaces.size(); ++i)
                    {
                        if (i > 0)
                            details << ", ";
                        details << toString(kernel.input_mem_spaces[i]);
                    }
                    details << "]\n  Requires contiguous inputs: [";
                    for (size_t i = 0; i < kernel.requiresContiguous.size(); ++i)
                    {
                        if (i > 0)
                            details << ", ";
                        details << (kernel.requiresContiguous[i] ? "true" : "false");
                    }
                    details << "]\n";
                }
                if (same_op_kernel_count == 0)
                    details << "Registered kernel candidates for this exact op type/name: none\n";

                details << "Kernel matching constraints:\n"
                        << "  Output memory space: " << toString(ram) << "\n"
                        << "  Required engines: [" << toString(cpu) << "]\n"
                        << "  Reference kernels only: true\n"
                        << "  Ignore output memory space: false\n"
                        << "  Ignore input memory spaces: " << (ignore_in_ms ? "true" : "false") << "\n"
                        << "  Ignore engines: false\n"
                        << "  Ignore input contiguity: true\n"
                        << "Full node graph context:\n"
                        << toString(node, graph);
                Error::throw_err(details.str());
            }

            for (KernelId uid : refs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                std::vector<EClassId> children;
                for (LogicalId pid : node.child_ids)
                {
                    children.push_back(baseState.findEClassByLogicalId(pid));
                }

                for (size_t input_idx = 0; input_idx < children.size(); ++input_idx)
                {
                    if (kernel.requiresContiguous.empty())
                        continue;

                    const size_t rule_idx = std::min(input_idx, kernel.requiresContiguous.size() - 1);
                    if (!kernel.requiresContiguous[rule_idx] || isContiguous(inputs[input_idx]))
                        continue;

                    const EClassId source_class = baseState.findConst(children[input_idx]);
                    auto contiguous_it = contiguousInputs.find(source_class);
                    if (contiguous_it == contiguousInputs.end())
                    {
                        const auto source = baseState.getEClass(source_class);
                        EClassId contiguous_class = addOpToEGraph(
                            baseState, OpType::CONTIGUOUS, {source_class}, source.shape,
                            calcContiguousStrides(source.shape), source.dtype, ram, EClassId(), SourceLocation::current(),
                            "Planner.initBaseEGraph input for " + toString(node.opType) + " logical ID " +
                                std::to_string(nodeId.value));
                        contiguous_it = contiguousInputs.emplace(source_class, contiguous_class).first;
                    }

                    children[input_idx] = contiguous_it->second;
                }

                ENode enode = ENode(uid, node.opType, node.opName, children, node.getShape(), node.strides, node.dtype,
                                    ram, {cpu}, "", 0, node.debugOrigin);
                baseState.addENode(e_class_id, enode);
            }
            // Resident weight: the in-memory copy is also a preallocated
            // input (written at session load).
            if (settings.resident_weights && isResidentWeightCopy(graph, nodeId))
                baseState.addENode(e_class_id,
                                   ENode(KernelId{0}, OpType::INPUT, "", {}, node.getShape(), node.strides, node.dtype,
                                         ram, {cpu}, "resident:" + graph.getNode(node.child_ids[0]).contentHash, 0,
                                         node.debugOrigin));
        }

        baseState.rebuild();
        baseStateInitialized = true;
    }

    bool injectPartialPath(EGraph &egraph, const Graph &graph, LogicalId logicalId,
                           const std::vector<Region> &dirtyRegions)
    {
        EClassId E_L = egraph.findEClassByLogicalId(logicalId);
        if (dirtyRegions.empty() || E_L == EClassId{})
            return false;

        const auto lStrides = egraph.getEClass(E_L).strides;
        const DType lDType = egraph.getEClass(E_L).dtype;
        const MemSpace lMemSpace = egraph.getEClass(E_L).mem_space;
        // Runtime inputs already contain current values after writeInput.
        // Expose real views of them; a fabricated partial INPUT would have
        // no binding and could scatter uninitialized memory into that input.
        for (const Region &reg : normalizeRegions(dirtyRegions))
        {
            const size_t rank = reg.region.size();
            std::vector<uint32_t> shape;
            shape.reserve(rank);
            std::vector<int32_t> starts, ends, steps;
            starts.reserve(rank);
            ends.reserve(rank);
            steps.reserve(rank);
            for (const Dim &dim : reg.region)
            {
                starts.push_back(static_cast<int32_t>(dim.start));
                ends.push_back(static_cast<int32_t>(dim.stop));
                steps.push_back(1);
                shape.push_back(dim.stop - dim.start);
            }
            const EClassId slice = addOpToEGraph(egraph, OpType::SLICE,
                {E_L, egraph.addIntConst(starts), egraph.addIntConst(ends), egraph.addIntConst(steps)},
                shape, lStrides, lDType, lMemSpace);
            addOpToEGraph(egraph, OpType::CONTIGUOUS, {slice}, shape,
                calcContiguousStrides(shape), lDType, lMemSpace);
        }
        return true;
    }

    bool injectInputPartialPaths(EGraph &egraph, const Graph &graph,
                                 const std::unordered_map<LogicalId, std::vector<Region>> &dirtyOutputRegions)
    {
        bool injected = false;
        for (const auto &kv : dirtyOutputRegions)
        {
            LogicalId nodeId = kv.first;
            if (!graph.hasNode(nodeId) || egraph.findEClassByLogicalId(nodeId) == EClassId{})
                continue;
            const TensorNode &node = graph.getNode(nodeId);
            if (node.opType == OpType::INPUT && graph.constantStaging.count(nodeId) == 0 && !kv.second.empty())
            {
                injected = injectPartialPath(egraph, graph, nodeId, kv.second) ||
                           injected;
            }
        }
        return injected;
    }

    bool injectOutputPartialPaths(EGraph &egraph, const Graph &graph, LogicalId rootId,
                                  const std::vector<Region> &outputNeeded)
    {
        bool injected = false;
        if (!outputNeeded.empty() && egraph.findEClassByLogicalId(rootId) != EClassId{})
        {
            injected = injectPartialPath(egraph, graph, rootId, outputNeeded);
        }
        return injected;
    }

    EGraph saturateBucket(const LogicalId rootId, const Graph &graph, const Bucket &bucket,
                          bool doSaturate = true,
                          TGStore *repo = nullptr, const EGraph *startingState = nullptr,
                          const std::vector<const Bucket *> *bucket_group = nullptr)
    {
        using BucketClock = std::chrono::steady_clock;
        const auto bucket_start = BucketClock::now();
        auto last_report = bucket_start;
        static const bool report_timing = (std::getenv("TG_TIME_BUCKETS") != nullptr);
        auto report_bucket_timing = [&](const std::string &stage, const std::string &details = std::string()) {
            if (!report_timing)
                return;
            const auto now = BucketClock::now();
            std::cout << "[saturateBucket] " << (startingState ? "partial" : "full") << " bucket " << stage
                      << " | total=" << std::chrono::duration<double, std::milli>(now - bucket_start).count()
                      << " ms | since previous="
                      << std::chrono::duration<double, std::milli>(now - last_report).count() << " ms";
            if (!details.empty())
                std::cout << " | " << details;
            std::cout << std::endl;
            last_report = now;
        };

        if (report_timing)
            report_bucket_timing("start", "doSaturate=" + std::string(doSaturate ? "true" : "false"));
        EGraph egraph;
        auto graph_setup_start = BucketClock::now();
        if (cached_topo_graph_ != &graph || cached_topo_root_ != rootId || cached_topo_.empty())
        {
            cached_topo_ = topologicalSort({rootId}, graph);
            cached_topo_root_ = rootId;
            cached_topo_graph_ = &graph;
        }
        const std::vector<LogicalId> &topo = cached_topo_;
        const std::vector<const Bucket *> single_bucket{&bucket};
        const std::vector<const Bucket *> &group = bucket_group ? *bucket_group : single_bucket;
        bool base_state_has_base_ids = false;

        if (startingState)
        {
            egraph = *startingState;
            egraph.allow_base_equalities = true;
        }
        else
        {
            Graph tempGraph = graph;
            initBaseEGraph(rootId, tempGraph, topo, repo, false);
            egraph = baseState;
        }
        if (report_timing)
            report_bucket_timing("prepared egraph state", "topological nodes=" + std::to_string(topo.size()) +
                                                                ", classes=" +
                                                                std::to_string(egraph.getNumClasses()) +
                                                                ", enodes=" + std::to_string(egraph.getNumENodes()) +
                                                                ", elapsed=" + std::to_string(
                                                                    std::chrono::duration<double, std::milli>(BucketClock::now() - graph_setup_start).count()) +
                                                                " ms");

        auto base_ids_start = BucketClock::now();
        if (startingState != nullptr)
        {
            base_state_has_base_ids = true;
        }
        else
        {
            for (size_t i = 0; i < egraph.getNumClasses(); ++i)
            {
                EClassId cid{static_cast<uint32_t>(i)};
                if (egraph.findConst(cid) == cid && egraph.getEClass(cid).base_eclass_id != BaseEClassId{})
                {
                    base_state_has_base_ids = true;
                    break;
                }
            }
        }
        if (report_timing)
            report_bucket_timing("checked base IDs", "elapsed=" + std::to_string(
                                                           std::chrono::duration<double, std::milli>(BucketClock::now() - base_ids_start).count()) +
                                                           " ms");

        auto dirty_start = BucketClock::now();
        uint32_t max_logical_id = 0;
        for (LogicalId nodeId : topo)
            max_logical_id = std::max(max_logical_id, nodeId.value);
        std::vector<uint8_t> &logicalDirty = logical_dirty_;
        logicalDirty.assign(max_logical_id + 1, 0);
        for (LogicalId nodeId : topo)
        {
            bool dirty = false;
            for (const Bucket *group_bucket : group)
                dirty = dirty || (group_bucket->inputDirtyRegions.count(nodeId) &&
                                  !group_bucket->inputDirtyRegions.at(nodeId).empty());
            if (!dirty)
            {
                for (LogicalId child : graph.getNode(nodeId).child_ids)
                {
                    if (child.value < logicalDirty.size() && logicalDirty[child.value])
                    {
                        dirty = true;
                        break;
                    }
                }
            }
            logicalDirty[nodeId.value] = dirty ? 1 : 0;
        }
        if (report_timing)
            report_bucket_timing("propagated dirty logical nodes", "nodes=" + std::to_string(topo.size()) +
                                                                         ", elapsed=" + std::to_string(
                                                                             std::chrono::duration<double, std::milli>(BucketClock::now() - dirty_start).count()) +
                                                                         " ms");

        std::unordered_set<EClassId> protectedEClasses;

        auto isFullRegion = [](const Region &region, const std::vector<uint32_t> &shape) {
            if (region.region.size() != shape.size())
                return false;
            for (size_t d = 0; d < shape.size(); ++d)
            {
                if (region.region[d].start != 0 || region.region[d].stop != shape[d])
                    return false;
            }
            return true;
        };

        bool hasPartialInputRegion = false;
        for (const Bucket *group_bucket : group)
        {
            for (const auto &inputPair : group_bucket->inputDirtyRegions)
            {
                if (!graph.hasNode(inputPair.first))
                    continue;
                const auto &shape = graph.getNode(inputPair.first).getShape();
                for (const Region &region : inputPair.second)
                {
                    if (!isFullRegion(region, shape))
                    {
                        hasPartialInputRegion = true;
                        break;
                    }
                }
                if (hasPartialInputRegion)
                    break;
            }
            if (hasPartialInputRegion)
                break;
        }
        bool hasPartialOutputRegion = false;
        const auto &outputShape = graph.getNode(rootId).getShape();
        for (const Bucket *group_bucket : group)
        {
            for (const Region &region : group_bucket->outputNeededRegion)
            {
                if (!isFullRegion(region, outputShape))
                {
                    hasPartialOutputRegion = true;
                    break;
                }
            }
            if (hasPartialOutputRegion)
                break;
        }

        auto inject_start = BucketClock::now();
        bool dirtyInjected = false;
        bool neededInjected = false;
        for (const Bucket *group_bucket : group)
        {
            dirtyInjected = (hasPartialInputRegion &&
                             injectInputPartialPaths(egraph, graph, group_bucket->inputDirtyRegions)) || dirtyInjected;
            neededInjected = (hasPartialOutputRegion &&
                             injectOutputPartialPaths(egraph, graph, rootId,
                                                      group_bucket->outputNeededRegion)) || neededInjected;
        }
        if (report_timing)
            report_bucket_timing("injected partial paths", "dirty=" + std::string(dirtyInjected ? "true" : "false") +
                                                                 ", needed=" +
                                                                 (neededInjected ? "true" : "false") +
                                                                 ", elapsed=" + std::to_string(
                                                                     std::chrono::duration<double, std::milli>(BucketClock::now() - inject_start).count()) +
                                                                 " ms");

        const size_t base_class_count = egraph.getNumClasses();
        const bool seeded = startingState && startingState == fullSaturationState_ && !fullSaturationRules_.empty();
        std::vector<uint32_t> local_base_class_enode_counts;
        std::vector<uint8_t> local_base_class_staged;
        const std::vector<uint32_t> *base_class_enode_counts_ptr = nullptr;
        const std::vector<uint8_t> *base_class_staged_ptr = nullptr;
        if (seeded)
        {
            base_class_enode_counts_ptr = &fullSaturationBaseClassEnodeCounts_;
            base_class_staged_ptr = &fullSaturationBaseClassStaged_;
        }
        else if (startingState && (dirtyInjected || neededInjected))
        {
            local_base_class_enode_counts.resize(base_class_count);
            for (size_t c = 0; c < base_class_count; ++c)
                local_base_class_enode_counts[c] = static_cast<uint32_t>(egraph.getRawEClass(static_cast<uint32_t>(c)).enodes.size());
            local_base_class_staged.assign(base_class_count, 0);
            for (const auto &[cid, data] : egraph.constantStaging)
                if (cid.value < base_class_count)
                    local_base_class_staged[cid.value] = 1;
            base_class_enode_counts_ptr = &local_base_class_enode_counts;
            base_class_staged_ptr = &local_base_class_staged;
        }

        auto saturation_start = BucketClock::now();
        const size_t enodes_before_saturation = egraph.getNumENodes();
        if (doSaturate && settings.do_saturate)
        {
            // A partial bucket starts from the saturated full state: continue
            // from the full saturation's rules when this planner ran it.
            BucketRegions regions;
            if (seeded && !dirtyInjected && !neededInjected)
            {
                // The base egraph was already saturated to fixpoint with fullSaturationRules_.
                // With zero injected partial paths or merges, zero rewrites can match.
            }
            else
            {
                if (dirtyInjected || neededInjected)
                {
                    regions.facts = regionFacts_;
                    for (const Bucket *group_bucket : group)
                    {
                        const BucketRegions current_regions = computeBucketRegions(
                            graph, rootId, *group_bucket, topo, repo, regionFacts_, &settings.padded_inputs);
                        for (const auto &[id, dirty_regions] : current_regions.dirty)
                            regions.dirty[id] = bucket_regions::unite(regions.dirtyOf(id), dirty_regions);
                        for (const auto &[id, needed_regions] : current_regions.needed)
                            regions.needed[id] = bucket_regions::unite(regions.neededOf(id), needed_regions);
                    }
                }
                if (std::getenv("TG_DEBUG_ROOTS"))
                    LOG(INFO) << "[debug regions] root dirty=" << encodeRegionList(regions.dirtyOf(rootId))
                              << " needed=" << encodeRegionList(regions.neededOf(rootId))
                              << " live=" << encodeRegionList(regions.liveOf(rootId))
                              << " root class logical=" << egraph.getEClass(egraph.findConst(egraph.findEClassByLogicalId(rootId))).logical_id.value;
                saturate(egraph, protectedEClasses, dirtyInjected || neededInjected, false, repo,
                         seeded ? &fullSaturationRules_ : nullptr, startingState ? nullptr : &fullSaturationRules_,
                         (dirtyInjected || neededInjected) ? &regions : nullptr,
                         seeded ? base_class_count : 0,
                         base_class_enode_counts_ptr,
                         base_class_staged_ptr,
                         seeded ? &fullSaturationSliceMemo_ : nullptr,
                         startingState ? nullptr : &fullSaturationSliceMemo_);
                if (!startingState)
                {
                    const uint32_t watermark = static_cast<uint32_t>(egraph.getNumENodes());
                    for (auto &r : fullSaturationRules_)
                    {
                        r->base_enode_watermark = watermark;
                        r->clearVisited();
                    }
                    for (auto &it : fullSaturationSliceMemo_)
                    {
                        it.second = egraph.findConst(it.second);
                    }
                    egraph.freezeHashcons();
                    egraph.freezeENodes();
                }
            }
            if (std::getenv("TG_DEBUG_MISSING") && (dirtyInjected || neededInjected))
            {
                std::map<std::string, uint32_t> missing, present;
                for (LogicalId id : topo)
                {
                    const std::vector<Region> live = regions.liveOf(id);
                    if (live.empty())
                        continue;
                    const EClassId cid = egraph.findEClassByLogicalId(id);
                    if (cid == EClassId{})
                        continue;
                    const EClass &cls = egraph.getEClass(egraph.findConst(cid));
                    if (bucket_regions::isFull(bucket_regions::boundingBox(live), cls.shape))
                        continue;
                    bool hasScatter = false;
                    for (ENodeId e : cls.enodes)
                        hasScatter = hasScatter || egraph.getENode(e).getOpType() == OpType::SCATTER;
                    const std::string key = toString(graph.getNode(id).opType);
                    (hasScatter ? present : missing)[key]++;
                }
                if (const char *chain = std::getenv("TG_DEBUG_CHAIN"))
                {
                    // Per-node chain for logical ids below the given bound.
                    const uint32_t bound = static_cast<uint32_t>(std::stoul(chain));
                    for (LogicalId id : topo)
                    {
                        if (id.value >= bound)
                            continue;
                        const TensorNode &node = graph.getNode(id);
                        const std::vector<Region> live = regions.liveOf(id);
                        const EClassId cid = egraph.findEClassByLogicalId(id);
                        std::string forms;
                        if (cid != EClassId{})
                        {
                            const EClass &cls = egraph.getEClass(egraph.findConst(cid));
                            std::set<std::string> ops;
                            for (ENodeId e : cls.enodes)
                                ops.insert(toString(egraph.getENode(e).getOpType()));
                            for (const auto &o : ops)
                                forms += o + ",";
                            // Restricted computations behind scatter forms.
                            auto classOps = [&](EClassId c) {
                                std::set<std::string> cops;
                                for (ENodeId ce : egraph.getEClass(egraph.findConst(c)).enodes)
                                    cops.insert(toString(egraph.getENode(ce).getOpType()));
                                std::string r;
                                for (const auto &o : cops)
                                    r += o + "/";
                                return r;
                            };
                            for (ENodeId e : cls.enodes)
                            {
                                const ENode &en = egraph.getENode(e);
                                if (en.getOpType() != OpType::SCATTER)
                                    continue;
                                const EClass &rc = egraph.getEClass(egraph.findConst(en.getChildren()[0]));
                                forms += " | R" + toString(rc.shape) + ":";
                                for (ENodeId re : rc.enodes)
                                {
                                    const ENode &rn = egraph.getENode(re);
                                    if (rn.getOpType() == OpType::FUSED)
                                        continue;
                                    forms += " " + toString(rn.getOpType()) + "(";
                                    for (EClassId ch : rn.getChildren())
                                        forms += classOps(ch) + ";";
                                    forms += ")";
                                }
                            }
                        }
                        std::string children;
                        for (LogicalId c : node.child_ids)
                            children += std::to_string(c.value) + ",";
                        LOG(INFO) << "[debug chain] " << id.value << " " << toString(node.opType) << " "
                                  << toString(node.getShape()) << " live="
                                  << (live.empty() ? std::string("-") : encodeRegion(bucket_regions::boundingBox(live)))
                                  << " children=" << children << " forms=" << forms;
                    }
                }
                std::stringstream ss;
                ss << "[debug missing scatter] missing:";
                for (const auto &[k, v] : missing)
                    ss << " " << k << "=" << v;
                ss << " | present:";
                for (const auto &[k, v] : present)
                    ss << " " << k << "=" << v;
                LOG(INFO) << ss.str();
            }
        }
        if (std::getenv("TG_SATURATION_HISTOGRAM"))
        {
            std::map<std::string, uint32_t> histogram;
            std::map<std::string, std::unordered_set<uint32_t>> distinct_classes;
            for (size_t i = enodes_before_saturation; i < egraph.getNumENodes(); ++i)
            {
                const ENode &enode = egraph.getENode(ENodeId{static_cast<uint32_t>(i)});
                std::string key = toString(enode.getOpType()) + ":" + enode.getOpName() + " " + toString(enode.getShape()) +
                                  " ms=" + toString(enode.getMemSpace());
                histogram[key]++;
                distinct_classes[key].insert(egraph.find(egraph.getENodeEClass(ENodeId{static_cast<uint32_t>(i)})).value);
            }
            std::vector<std::pair<uint32_t, std::string>> sorted;
            for (const auto &[key, count] : histogram)
                sorted.push_back({count, key});
            std::sort(sorted.rbegin(), sorted.rend());
            const bool all = std::string(std::getenv("TG_SATURATION_HISTOGRAM")) == "all";
            for (size_t i = 0; i < sorted.size() && (all || i < 60); ++i)
                LOG(INFO) << "[saturation histogram] " << sorted[i].first << " classes="
                          << distinct_classes[sorted[i].second].size() << " " << sorted[i].second;
        }
        if (report_timing)
            report_bucket_timing("finished rewrite saturation", "elapsed=" + std::to_string(
                                                                       std::chrono::duration<double, std::milli>(BucketClock::now() - saturation_start).count()) +
                                                                       " ms, classes=" +
                                                                       std::to_string(egraph.getNumClasses()) +
                                                                       ", enodes=" +
                                                                       std::to_string(egraph.getNumENodes()));

        const uint32_t maxClasses = static_cast<uint32_t>(egraph.getNumClasses());
        auto clean_init_start = BucketClock::now();
        std::vector<uint8_t> clean(maxClasses, 0);
        for (LogicalId id : topo)
        {
            if (id.value < logicalDirty.size() && !logicalDirty[id.value])
            {
                EClassId cid = egraph.findEClassByLogicalId(id);
                if (cid != EClassId{})
                {
                    uint32_t canon = egraph.findConst(cid).value;
                    if (canon < maxClasses)
                        clean[canon] = 1;
                }
            }
        }
        for (const auto &[cid, _] : egraph.constantStaging)
        {
            uint32_t canon = egraph.findConst(cid).value;
            if (canon < maxClasses)
                clean[canon] = 1;
        }
        if (report_timing)
            report_bucket_timing("seeded clean classes", "classes=" + std::to_string(maxClasses) +
                                                            ", elapsed=" + std::to_string(
                                                                std::chrono::duration<double, std::milli>(BucketClock::now() - clean_init_start).count()) +
                                                            " ms");
        uint32_t clean_round = 1;
        uint64_t clean_class_checks = 0;
        auto clean_propagation_start = BucketClock::now();
        const uint32_t num_classes = maxClasses;
        const uint32_t num_enodes = static_cast<uint32_t>(egraph.getNumENodes());
        std::vector<uint32_t> &dirty_child_count = clean_dirty_child_count_;
        dirty_child_count.assign(num_enodes, 0);
        std::vector<uint32_t> &parent_head = clean_parent_head_;
        parent_head.assign(num_classes, UINT32_MAX);
        std::vector<uint32_t> &parent_next = clean_parent_next_;
        parent_next.clear();
        parent_next.reserve(num_enodes * 2);
        std::vector<uint32_t> &parent_node = clean_parent_node_;
        parent_node.clear();
        parent_node.reserve(num_enodes * 2);

        for (uint32_t e = 0; e < num_enodes; ++e)
        {
            const ENode &enode = egraph.getENode(ENodeId{e});
            if (enode.getOpType() == OpType::INPUT)
                continue;
            std::array<uint32_t, 8> unique_children;
            size_t num_unique = 0;
            for (EClassId cid : enode.getChildren())
            {
                uint32_t c_val = egraph.findConst(cid).value;
                bool found = false;
                for (size_t i = 0; i < num_unique; ++i)
                {
                    if (unique_children[i] == c_val)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found && num_unique < unique_children.size())
                    unique_children[num_unique++] = c_val;
            }
            dirty_child_count[e] = static_cast<uint32_t>(num_unique);
            for (size_t i = 0; i < num_unique; ++i)
            {
                uint32_t c_val = unique_children[i];
                if (c_val < num_classes)
                {
                    uint32_t edge_idx = static_cast<uint32_t>(parent_next.size());
                    parent_next.push_back(parent_head[c_val]);
                    parent_node.push_back(e);
                    parent_head[c_val] = edge_idx;
                }
            }
        }

        std::vector<uint32_t> &queue = clean_queue_;
        queue.clear();
        queue.reserve(num_classes / 4);
        std::vector<uint8_t> &in_queue = clean_in_queue_;
        in_queue.assign(num_classes, 0);
        for (uint32_t i = 0; i < num_classes; ++i)
        {
            if (clean[i] && egraph.findConst(EClassId{i}).value == i)
            {
                in_queue[i] = 1;
                queue.push_back(i);
            }
        }

        for (uint32_t e = 0; e < num_enodes; ++e)
        {
            const ENode &enode = egraph.getENode(ENodeId{e});
            if (enode.getOpType() == OpType::INPUT)
                continue;
            if (dirty_child_count[e] == 0)
            {
                const uint32_t owner_val = egraph.findConst(egraph.getENodeEClass(ENodeId{e})).value;
                if (owner_val < num_classes && !clean[owner_val])
                {
                    clean[owner_val] = 1;
                    if (!in_queue[owner_val])
                    {
                        in_queue[owner_val] = 1;
                        queue.push_back(owner_val);
                    }
                }
            }
        }

        while (!queue.empty())
        {
            uint32_t c = queue.back();
            queue.pop_back();
            for (uint32_t idx = parent_head[c]; idx != UINT32_MAX; idx = parent_next[idx])
            {
                uint32_t e = parent_node[idx];
                ++clean_class_checks;
                if (dirty_child_count[e] == 0)
                    continue;
                dirty_child_count[e]--;
                if (dirty_child_count[e] == 0)
                {
                    const uint32_t owner_val = egraph.findConst(egraph.getENodeEClass(ENodeId{e})).value;
                    if (owner_val < num_classes && !clean[owner_val])
                    {
                        clean[owner_val] = 1;
                        if (!in_queue[owner_val])
                        {
                            in_queue[owner_val] = 1;
                            queue.push_back(owner_val);
                        }
                    }
                }
            }
        }
        const double clean_propagation_ms = std::chrono::duration<double, std::milli>(
                                                BucketClock::now() - clean_propagation_start)
                                                .count();
        auto collect_clean_start = BucketClock::now();
        egraph.clean_classes = std::move(clean);
        if (!startingState)
        {
            if (!base_state_has_base_ids)
                egraph.populateBaseEClassIds();
            const size_t base_count = egraph.getNumClasses();
            fullSaturationBaseClassEnodeCounts_.resize(base_count);
            for (size_t c = 0; c < base_count; ++c)
                fullSaturationBaseClassEnodeCounts_[c] = static_cast<uint32_t>(egraph.getRawEClass(static_cast<uint32_t>(c)).enodes.size());
            fullSaturationBaseClassStaged_.assign(base_count, 0);
            for (const auto &[cid, data] : egraph.constantStaging)
                if (cid.value < base_count)
                    fullSaturationBaseClassStaged_[cid.value] = 1;
            egraph.freezeClasses();
            egraph.freezeHashcons();
            egraph.freezeENodes();
            egraph.freezeLogicalIndex();
        }
        if (report_timing)
            report_bucket_timing("finished bucket postprocessing", "clean rounds=" + std::to_string(clean_round) +
                                                                         ", class checks=" +
                                                                         std::to_string(clean_class_checks) +
                                                                         ", propagation elapsed=" +
                                                                         std::to_string(clean_propagation_ms) + " ms" +
                                                                         ", collection/remap elapsed=" +
                                                                         std::to_string(
                                                                             std::chrono::duration<double, std::milli>(BucketClock::now() - collect_clean_start).count()) +
                                                                         " ms");
        egraph.allow_base_equalities = false;
        return egraph;
    }

    CompiledGraph buildCompiledGraph(LogicalId rootId, const Graph &graph, const EGraph &egraph,
                                     const ExtractionResult &extraction,
                                     const std::vector<ENodeInfo> &enodeInfos)
    {
        CompiledGraph compiled;
        std::unordered_map<BufferId, const ParallelBuffer *> buf_map;
        buf_map.reserve(extraction.buffers.size());
        for (const ParallelBuffer &buf : extraction.buffers)
            buf_map[buf.id] = &buf;

        for (EClassId eclass_id : extraction.order)
        {
            const ENode &enode =
                egraph.getENode(egraph.getEClass(eclass_id).enodes[extraction.selection_map.at(eclass_id)]);

            LogicalId logical_id;
            const EClass &cls = egraph.getEClass(eclass_id);
            if (cls.logical_id != LogicalId{})
            {
                logical_id = cls.logical_id;
            }
            else
            {
                EClassId base_eclass = resolve_view_alias(eclass_id, egraph, extraction.selection_map, enodeInfos);
                const EClass &base_cls = egraph.getEClass(base_eclass);
                if (base_cls.logical_id != LogicalId{})
                {
                    logical_id = base_cls.logical_id;
                }
            }

            if (logical_id != LogicalId{UINT32_MAX} && logical_id.value != UINT32_MAX)
            {
                compiled.eclass_to_logical[eclass_id] = logical_id;
                compiled.logical_to_eclass[logical_id] = eclass_id;
            }

            OpInstruction inst;
            inst.eclass_id = eclass_id;
            inst.logical_id = logical_id;
            BaseEClassId base_id = cls.base_eclass_id;
            if (base_id == BaseEClassId{})
            {
                base_id = egraph.getEClass(egraph.findConst(eclass_id)).base_eclass_id;
            }
            Debug::BaseRefVerifier::get().registerBase(eclass_id, base_id);
            inst.kernel_id = enode.getKernelId();

            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = egraph.findConst(child);
                inst.children.push_back(canon_child);

                if (!compiled.eclass_to_logical.count(canon_child))
                {
                    const EClass &child_cls = egraph.getEClass(canon_child);
                    if (child_cls.logical_id != LogicalId{})
                    {
                        compiled.eclass_to_logical[canon_child] = child_cls.logical_id;
                        compiled.logical_to_eclass[child_cls.logical_id] = canon_child;
                    }
                    else
                    {
                        EClassId base_child =
                            resolve_view_alias(canon_child, egraph, extraction.selection_map, enodeInfos);
                        const EClass &base_child_cls = egraph.getEClass(base_child);
                        if (base_child_cls.logical_id != LogicalId{})
                        {
                            compiled.eclass_to_logical[canon_child] = base_child_cls.logical_id;
                            compiled.logical_to_eclass[base_child_cls.logical_id] = canon_child;
                        }
                    }
                }
            }
            inst.inBuffers.resize(inst.children.size());

            auto out_buf_it = extraction.eclass_to_buf.find(eclass_id);
            if (out_buf_it != extraction.eclass_to_buf.end())
            {
                auto it = buf_map.find(out_buf_it->second);
                if (it != buf_map.end())
                    inst.outBuffer = *it->second;
            }

            for (uint32_t j = 0; j < inst.children.size(); j++)
            {
                auto in_buf_it = extraction.eclass_to_buf.find(inst.children[j]);
                if (in_buf_it != extraction.eclass_to_buf.end())
                {
                    auto it = buf_map.find(in_buf_it->second);
                    if (it != buf_map.end())
                        inst.inBuffers[j] = *it->second;
                }
            }

            if (logical_id != LogicalId{UINT32_MAX} && logical_id.value != UINT32_MAX && graph.hasNode(logical_id))
            {
                inst.debugOrigin = graph.getNode(logical_id).debugOrigin;
            }

            if (enode.getOpType() == OpType::INPUT && enode.getMemSpace().type == HandleType::CPP &&
                egraph.constantStaging.count(eclass_id))
            {
                compiled.constantStaging[eclass_id] = egraph.constantStaging.at(eclass_id);
            }

            bool is_view = false;
            const KernelEntry *kernel_ptr = nullptr;
            if (enode.getKernelId().value != 0)
            {
                kernel_ptr = &KernelRegistry::get().getKernel(enode.getKernelId());
                is_view = kernel_ptr->is_view;
            }

            uint64_t final_offset_bytes = (inst.outBuffer.offset >= 0) ? static_cast<uint64_t>(inst.outBuffer.offset) : 0ULL;
            std::vector<uint64_t> final_strides = enode.getStrides();

            if (is_view && kernel_ptr && kernel_ptr->inferView)
            {
                Graph tempGraph;
                std::vector<TensorNode> dummyInputNodes;

                for (uint32_t i = 0; i < inst.children.size(); i++)
                {
                    EClassId child_id = inst.children[i];
                    if (compiled.nodeViews.count(child_id))
                    {
                        const TensorView &childView = compiled.nodeViews.at(child_id);
                        LogicalId fakeId = tempGraph.input(childView.getShape(), childView.dtype, childView.strides);

                        if (egraph.constantStaging.count(child_id))
                        {
                            tempGraph.constantStaging[fakeId] = egraph.constantStaging.at(child_id);
                        }
                        else if (compiled.constantStaging.count(child_id))
                        {
                            tempGraph.constantStaging[fakeId] = compiled.constantStaging.at(child_id);
                        }
                        else if (egraph.getEClass(child_id).logical_id != LogicalId{} &&
                                 graph.constantStaging.count(egraph.getEClass(child_id).logical_id))
                        {
                            tempGraph.constantStaging[fakeId] = graph.constantStaging.at(egraph.getEClass(child_id).logical_id);
                        }

                        dummyInputNodes.push_back(tempGraph.getNode(fakeId));
                    }
                }

                if (!inst.children.empty() && compiled.nodeViews.count(inst.children[0]))
                {
                    final_offset_bytes = compiled.nodeViews.at(inst.children[0]).offset;
                }

                TensorView dummyOutView(enode.getShape(), final_offset_bytes, enode.getStrides(), enode.getDType());
                kernel_ptr->inferView(dummyInputNodes, dummyOutView, tempGraph);

                final_offset_bytes = dummyOutView.offset;
                final_strides = dummyOutView.strides;
            }

            compiled.nodeViews[eclass_id] =
                TensorView(enode.getShape(), final_offset_bytes, final_strides, enode.getDType());

            if (kernel_ptr)
            {
                std::vector<TensorNode> dummyInputs(inst.children.size());
                std::vector<MemSpace> in_mem_spaces(inst.children.size());
                for (size_t i = 0; i < inst.children.size(); ++i)
                {
                    if (compiled.nodeViews.count(inst.children[i]))
                    {
                        const auto &view = compiled.nodeViews.at(inst.children[i]);
                        dummyInputs[i].setShape(view.getShape());
                        dummyInputs[i].strides = view.strides;
                        dummyInputs[i].dtype = view.dtype;
                    }
                    in_mem_spaces[i] = inst.inBuffers[i].mem_space;
                }
                TensorNode dummyOutput;
                const auto &outView = compiled.nodeViews.at(eclass_id);
                dummyOutput.setShape(outView.getShape());
                dummyOutput.strides = outView.strides;
                dummyOutput.dtype = outView.dtype;

                kernel_ptr->matches(dummyInputs, dummyOutput, inst.outBuffer.mem_space, in_mem_spaces, {}, false, false,
                                    true, true, &inst.engines);
            }

            if (inst.engines.empty())
            {
                if (inst.outBuffer.mem_space.type == HandleType::CUDA)
                    inst.engines.push_back(Engine{inst.outBuffer.mem_space.idx, EngineType::CUDA_GPU});
                else
                    inst.engines.push_back(Engine{0, EngineType::CPU});
            }

            if (enode.getOpType() != OpType::INPUT && enode.getOpType() != OpType::CACHE && !is_view)
            {
                if (kernel_ptr && !kernel_ptr->requiresContiguous.empty())
                {
                    for (size_t c_idx = 0; c_idx < inst.children.size(); ++c_idx)
                    {
                        size_t rule_idx = std::min(c_idx, kernel_ptr->requiresContiguous.size() - 1);
                        if (kernel_ptr->requiresContiguous[rule_idx])
                        {
                            const auto &child_view = compiled.nodeViews.at(inst.children[c_idx]);
                            if (!isContiguous(child_view))
                            {
                                std::stringstream ss;
                                ss << "[Planner] Instruction assembly contiguity violation: kernel "
                                   << kernel_ptr->getName() << " (eclass " << eclass_id.value
                                   << ") input #" << c_idx << " (eclass " << inst.children[c_idx].value
                                   << ") is not contiguous, but kernel requires contiguous input!"
                                   << " shape=" << toString(child_view.getShape())
                                   << " strides=" << toString(child_view.strides);
                                Error::throw_err(ss.str());
                            }
                        }
                    }
                }
                compiled.instructions.push_back(inst);
            }
        }

        compiled.nodeCosts = extraction.eclass_to_cost;
        for (const auto &pair : extraction.selection_map)
        {
            EClassId cid = pair.first;
            uint32_t en_idx = pair.second;
            ENodeId en_id = egraph.getEClass(cid).enodes[en_idx];
            float en_cost = (en_id.value < enodeInfos.size()) ? enodeInfos[en_id.value].cost : 0.0f;
            compiled.nodeCosts[cid] = en_cost;
        }

        // Views of classes the plan never reads (saturation alternatives that
        // were not selected) are not stored: every lookup is by an
        // instruction's output or child, a constant, or a logical id.
        std::unordered_map<BufferId, uint64_t> buffer_offsets;
        for (const auto &buf : extraction.buffers)
            buffer_offsets.emplace(buf.id, buf.offset >= 0 ? static_cast<uint64_t>(buf.offset) : 0ULL);
        auto addView = [&](EClassId cid) {
            if (compiled.nodeViews.count(cid))
                return;
            const EClass &cls = egraph.getEClass(cid);
            uint64_t offset = 0;
            auto out_buf_it = extraction.eclass_to_buf.find(cid);
            if (out_buf_it != extraction.eclass_to_buf.end())
            {
                auto offset_it = buffer_offsets.find(out_buf_it->second);
                if (offset_it != buffer_offsets.end())
                    offset = offset_it->second;
            }
            compiled.nodeViews[cid] = TensorView(cls.shape, offset, cls.strides, cls.dtype);
        };
        for (size_t i = 0; i < egraph.getNumClasses(); ++i)
        {
            EClassId cid{static_cast<uint32_t>(i)};
            if (egraph.findConst(cid) != cid)
                continue;
            const EClass &cls = egraph.getEClass(cid);
            if (cls.logical_id != LogicalId{})
            {
                compiled.logical_to_eclass[cls.logical_id] = cls.id;
                if (!compiled.eclass_to_logical.count(cls.id))
                {
                    compiled.eclass_to_logical[cls.id] = cls.logical_id;
                }
                addView(cls.id);
            }

            if (cls.base_eclass_id != BaseEClassId{})
            {
                Debug::BaseRefVerifier::get().registerBase(cls.id, cls.base_eclass_id);
            }
        }
        for (const auto &[cid, logical_id] : compiled.eclass_to_logical)
            if (egraph.findConst(cid) == cid)
                addView(cid);
        for (const auto &[cid, data] : compiled.constantStaging)
            if (egraph.findConst(cid) == cid)
                addView(cid);
        for (const OpInstruction &inst : compiled.instructions)
        {
            if (egraph.findConst(inst.eclass_id) == inst.eclass_id)
                addView(inst.eclass_id);
            for (EClassId child : inst.children)
                if (egraph.findConst(child) == child)
                    addView(child);
        }

        return compiled;
    }

    // =========================================================================
    // Joint Multi-Bucket Planning: 1. Saturate -> 2. Search
    // =========================================================================
    std::pair<std::vector<CompiledGraph>, std::unordered_set<BaseEClassId>>
    planAll(LogicalId rootId, const Graph &graph, const std::vector<Bucket> &buckets,
            const std::vector<float> &bucket_weights, bool doSaturate = true, TGStore *repo = nullptr,
            float minCompileSeconds = 0.0f, std::shared_ptr<plan::Brancher> brancher = nullptr)
    {
        struct CostMemoScope
        {
            Planner &planner;
            explicit CostMemoScope(Planner &p) : planner(p) { planner.cost_memo_ = std::make_shared<CostMemo>(); }
            ~CostMemoScope() { planner.cost_memo_.reset(); }
        } cost_memo_scope(*this);

#if defined(__GLIBC__)
        if (!tg_malloc_tuning::kApplied)
            Error::throw_err("[Planner.planAll] glibc rejected the allocator settings in malloc_tuning.hpp");
#endif
        LOG(INFO) << "[Planner.planAll] Planning " << buckets.size() << " bucket(s) with weights:";
        // A bucket identical to an earlier one takes its saturated and reduced
        // e-graph instead of building the same one again.
        std::vector<uint32_t> identical_to(buckets.size());
        std::vector<uint32_t> shared_state_to(buckets.size());
        for (uint32_t b = 0; b < buckets.size(); ++b)
        {
            identical_to[b] = b;
            shared_state_to[b] = b;
            std::string duplicates;
            for (uint32_t prior = 0; prior < b; ++prior)
            {
                if (buckets[b] == buckets[prior])
                {
                    duplicates = ", identical to bucket " + std::to_string(prior);
                    identical_to[b] = identical_to[prior];
                    break;
                }
            }
            const float weight = b < bucket_weights.size() ? bucket_weights[b] : 1.0f;
            LOG(INFO) << "[Planner.planAll]   bucket " << b << ": weight=" << weight << duplicates
                      << " | " << buckets[b];
        }

        // 1. SATURATE
        const auto plan_all_start = std::chrono::steady_clock::now();
        auto planAllSeconds = [&]() {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - plan_all_start).count();
        };
        LOG(INFO) << "[Planner.planAll timeline] start epoch=" << std::fixed << std::setprecision(3)
                  << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
                  << std::defaultfloat;
        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);
        Graph temp_graph = graph;
        initBaseEGraph(rootId, temp_graph, topo, repo, false);

        uint32_t full_idx = 0;
        for (uint32_t i = 0; i < buckets.size(); ++i)
        {
            if (buckets[i].weight == 0.0f)
            {
                full_idx = i;
                break;
            }
        }

        EGraph full_state = saturateBucket(rootId, graph, buckets[full_idx], doSaturate, repo);
        fullSaturationState_ = &full_state;
        LOG(INFO) << "[Planner.planAll timeline] full saturation done at " << planAllSeconds() << "s";
        const RegionFacts region_facts = computeRegionFacts(graph, topo, repo);
        regionFacts_ = &region_facts;
        struct FullStateScope
        {
            Planner &planner;
            ~FullStateScope()
            {
                planner.fullSaturationState_ = nullptr;
                planner.regionFacts_ = nullptr;
                planner.fullSaturationBaseClassEnodeCounts_.clear();
                planner.fullSaturationBaseClassStaged_.clear();
            }
        } full_state_scope{*this};
        std::vector<EGraph> bucket_states(buckets.size());
        bucket_states[full_idx] = full_state;

        // Partial buckets are split into groups of consecutive buckets; each
        // group shares one superset e-graph: every member's partial paths are
        // injected before one seeded rewrite closure, and the union of member
        // regions lets the rules retain alternatives useful to any member.
        // The groups' closures are independent and run in parallel.
        std::vector<uint32_t> partial_indices;
        for (uint32_t b = 0; b < buckets.size(); ++b)
            if (b != full_idx)
                partial_indices.push_back(b);
        if (!partial_indices.empty())
        {
            const uint32_t num_partial = static_cast<uint32_t>(partial_indices.size());
            // By default buckets that need the same output region share a
            // group: they compute the same rows and differ only in which
            // inputs are dirty, so most of their closure is common. The
            // closure cost of a group grows with the number of distinct
            // regions in it, so each region gets its own group, and regions
            // are merged consecutively only when there are more of them than
            // hardware threads. TG_PARTIAL_GROUPS overrides with that many
            // groups of consecutive buckets (a diagnostic).
            std::vector<uint32_t> region_of(num_partial);
            std::vector<const std::vector<Region> *> distinct_regions;
            for (uint32_t i = 0; i < num_partial; ++i)
            {
                const std::vector<Region> &region = buckets[partial_indices[i]].outputNeededRegion;
                uint32_t k = 0;
                while (k < distinct_regions.size() && !(*distinct_regions[k] == region))
                    ++k;
                if (k == distinct_regions.size())
                    distinct_regions.push_back(&region);
                region_of[i] = k;
            }
            const uint32_t num_regions = static_cast<uint32_t>(distinct_regions.size());
            const uint32_t hw_threads = std::max(1u, std::thread::hardware_concurrency());
            uint32_t num_groups = std::min(num_regions, hw_threads);
            const char *groups_text = std::getenv("TG_PARTIAL_GROUPS");
            if (groups_text)
                num_groups = static_cast<uint32_t>(std::stoul(groups_text));
            if (num_groups == 0 || num_groups > num_partial)
                Error::throw_err("[Planner.planAll] partial bucket groups must be in [1, " +
                                 std::to_string(num_partial) + "], got " + std::to_string(num_groups));
            std::vector<std::vector<uint32_t>> groups(num_groups);
            if (std::getenv("TG_PARTIAL_GROUPS_ROUND_ROBIN"))
                for (uint32_t i = 0; i < num_partial; ++i)
                    groups[i % num_groups].push_back(partial_indices[i]);
            else if (groups_text)
                for (uint32_t g = 0; g < num_groups; ++g)
                    for (uint32_t i = g * num_partial / num_groups; i < (g + 1) * num_partial / num_groups; ++i)
                        groups[g].push_back(partial_indices[i]);
            else
                for (uint32_t i = 0; i < num_partial; ++i)
                    groups[static_cast<uint64_t>(region_of[i]) * num_groups / num_regions].push_back(partial_indices[i]);
            LOG(INFO) << "[Planner.planAll] " << num_partial << " partial bucket(s), " << num_regions
                      << " distinct output region(s), " << num_groups << " saturation group(s)";
            // Build full_state's lazy lookup indices before it is read from
            // several threads.
            full_state.findEClassByLogicalId(rootId);
            for (size_t c = 0; c < full_state.getNumClasses(); ++c)
            {
                const EClass &cls = full_state.getEClass(EClassId{static_cast<uint32_t>(c)});
                if (cls.base_eclass_id != BaseEClassId{})
                {
                    full_state.findEClassByBaseId(cls.base_eclass_id);
                    break;
                }
            }
            // A group's state is kept by a member that is not an alias of an
            // identical earlier bucket (aliases are redirected below).
            std::vector<uint32_t> group_owner(num_groups, UINT32_MAX);
            for (uint32_t g = 0; g < num_groups; ++g)
            {
                for (uint32_t b : groups[g])
                    if (identical_to[b] == b)
                    {
                        group_owner[g] = b;
                        break;
                    }
                if (group_owner[g] == UINT32_MAX)
                    Error::throw_err("[Planner.planAll] partial bucket group " + std::to_string(g) +
                                     " has only aliases of identical buckets");
            }
            std::vector<std::thread> threads;
            std::vector<std::exception_ptr> errors(num_groups);
            for (uint32_t g = 0; g < num_groups; ++g)
                threads.emplace_back([&, g]() {
                    try
                    {
                        Planner worker_planner(costModel, settings);
                        worker_planner.cost_memo_ = cost_memo_;
                        worker_planner.fullSaturationRules_ = fullSaturationRules_;
                        worker_planner.fullSaturationState_ = fullSaturationState_;
                        worker_planner.fullSaturationSliceMemo_ = fullSaturationSliceMemo_;
                        worker_planner.fullSaturationBaseClassEnodeCounts_ = fullSaturationBaseClassEnodeCounts_;
                        worker_planner.fullSaturationBaseClassStaged_ = fullSaturationBaseClassStaged_;
                        worker_planner.regionFacts_ = regionFacts_;
                        std::vector<const Bucket *> members;
                        for (uint32_t b : groups[g])
                            members.push_back(&buckets[b]);
                        // Keep one owner per group until reduction finishes;
                        // copying a union graph per bucket duplicates
                        // hundreds of thousands of nodes.
                        const auto group_start = std::chrono::steady_clock::now();
                        bucket_states[group_owner[g]] = worker_planner.saturateBucket(
                            rootId, graph, *members.front(), settings.saturate_partial_buckets, repo, &full_state,
                            &members);
                        LOG(INFO) << "[Planner.planAll] saturation group " << g << " (" << members.size()
                                  << " bucket(s), owner " << group_owner[g] << ") closed in "
                                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - group_start).count()
                                  << "s with " << bucket_states[group_owner[g]].getNumENodes() << " enodes";
                        worker_planner.fullSaturationState_ = nullptr;
                        worker_planner.regionFacts_ = nullptr;
                    }
                    catch (...)
                    {
                        errors[g] = std::current_exception();
                    }
                });
            for (auto &thread : threads)
                thread.join();
            for (auto &error : errors)
                if (error)
                    std::rethrow_exception(error);
            for (uint32_t g = 0; g < num_groups; ++g)
                for (uint32_t b : groups[g])
                    shared_state_to[b] = group_owner[g];
        }

        for (uint32_t b = 0; b < bucket_states.size(); ++b)
            if (identical_to[b] != b && b != full_idx)
                shared_state_to[b] = shared_state_to[identical_to[b]];
        for (uint32_t b = 0; b < bucket_states.size(); ++b)
            egraph_diagnostics::logClassSizeHistogram(bucket_states[shared_state_to[b]],
                                                      "bucket=" + std::to_string(b) + " stage=saturated");
        std::vector<EGraph *> unique_bucket_states{&full_state, &bucket_states[full_idx]};
        for (uint32_t b = 0; b < buckets.size(); ++b)
            if (shared_state_to[b] == b && b != full_idx)
                unique_bucket_states.push_back(&bucket_states[b]);
        LOG(INFO) << "[Planner.planAll timeline] partial saturation done at " << planAllSeconds() << "s";
        reconcileBaseEqualities(unique_bucket_states);
        for (uint32_t b = 0; b < bucket_states.size(); ++b)
            egraph_diagnostics::logClassSizeHistogram(bucket_states[shared_state_to[b]],
                                                      "bucket=" + std::to_string(b) + " stage=reconciled");
        LOG(INFO) << "[Planner.planAll timeline] saturation done at " << planAllSeconds() << "s";
#ifdef _WIN32
        if (SamplingProfiler::enabled())
        {
            SamplingProfiler::get().start();
            SamplingProfiler::get().registerCurrentThread();
        }
#endif
        // Cache candidate discovery across saturated egraphs.  Cache choices
        // only help weighted partial buckets; the full bucket is
        // the initialization path and must compute every value itself.
        std::vector<CacheCandidate> candidates;
        bool has_weighted_partial_bucket = false;
        for (uint32_t b = 0; b < buckets.size(); ++b)
        {
            if (b != full_idx && bucket_weights[b] > 0.0f)
            {
                has_weighted_partial_bucket = true;
                break;
            }
        }
        if (!settings.disable_caching && has_weighted_partial_bucket)
        {
            std::unordered_map<LogicalId, uint32_t> user_counts;
            for (const auto &pair : graph.nodes)
            {
                for (LogicalId child_id : pair.second.child_ids)
                    user_counts[child_id]++;
            }

            for (size_t i = 0; i < full_state.getNumClasses(); ++i)
            {
                EClassId cid{static_cast<uint32_t>(i)};
                if (full_state.findConst(cid) != cid)
                    continue;
                const EClass &cls = full_state.getEClass(cid);
                if (cls.base_eclass_id == BaseEClassId{} ||
                    cls.mem_space.type == HandleType::STORAGE || getStorageSpanBytes(cls.shape, cls.strides, cls.dtype) == 0)
                    continue;

                LogicalId log_id = cls.logical_id;
                if (cls.id == full_state.findConst(full_state.findEClassByLogicalId(rootId)))
                {
                    // Root buckets need not maintain unrequested dirty rows.
                    // Persist the root only if every bucket refreshes them.
                    bool maintained = true;
                    for (uint32_t b = 0; b < buckets.size() && maintained; ++b)
                    {
                        if (b == full_idx || buckets[b].outputNeededRegion.empty())
                            continue;
                        const BucketRegions regions = computeBucketRegions(graph, rootId, buckets[b], topo, repo,
                            regionFacts_, &settings.padded_inputs);
                        maintained = coversRegionList(buckets[b].outputNeededRegion, regions.dirtyOf(rootId));
                    }
                    if (!maintained)
                        continue;
                }
                if (log_id != LogicalId{} && graph.hasNode(log_id) && graph.getNode(log_id).opType == OpType::INPUT)
                    continue;

                bool is_input_class = false;
                bool has_non_view = false;
                for (ENodeId eid : cls.enodes)
                {
                    const ENode &enode = full_state.getENode(eid);
                    if (enode.getOpType() == OpType::INPUT)
                    {
                        is_input_class = true;
                        break;
                    }
                    bool is_view = false;
                    if (enode.getKernelId().value != 0)
                    {
                        is_view = KernelRegistry::get().getKernel(enode.getKernelId()).is_view;
                    }
                    if (!is_view)
                    {
                        has_non_view = true;
                    }
                }
                if (is_input_class || !has_non_view)
                    continue;

                // Persisting a value across runs pays off where a bucket can
                // reuse it: it is clean there, or the bucket can update it in
                // place (a SCATTER form keeps the rest of the previous value).
                bool clean_in_any = false;
                for (uint32_t b = 0; b < bucket_states.size(); ++b)
                {
                    if (b == full_idx || bucket_weights[b] <= 0.0f)
                        continue;
                    EClassId bid = bucket_states[b].findEClassByBaseId(cls.base_eclass_id);
                    if (bid == EClassId{})
                        continue;
                    const EClass &bcls = bucket_states[b].getEClass(bid);
                    bool updatable = bucket_states[b].isClean(bid);
                    for (ENodeId enodeId : bcls.enodes)
                        updatable = updatable || bucket_states[b].getENode(enodeId).getOpType() == OpType::SCATTER;
                    if (updatable)
                    {
                        clean_in_any = true;
                        break;
                    }
                }
                if (clean_in_any)
                {
                    uint32_t n_users = (log_id != LogicalId{}) ? user_counts[log_id] : 0;
                    uint64_t max_cand_bytes = getStorageSpanBytes(cls.shape, cls.strides, cls.dtype);
                    for (uint32_t b = 0; b < bucket_states.size(); ++b)
                    {
                        EClassId bid = bucket_states[b].findEClassByBaseId(cls.base_eclass_id);
                        if (bid != EClassId{})
                        {
                            const EClass &b_cls = bucket_states[b].getEClass(bid);
                            max_cand_bytes = std::max(max_cand_bytes, getStorageSpanBytes(b_cls.shape, b_cls.strides, b_cls.dtype));
                        }
                    }
                    candidates.push_back(
                        {cls.base_eclass_id, max_cand_bytes, cls.dtype, cls.mem_space, n_users});
                }
            }
            std::stable_sort(candidates.begin(), candidates.end(),
                             [](const CacheCandidate &a, const CacheCandidate &b) { return a.num_users > b.num_users; });
        }

        // Add CACHE enodes to bucket egraphs for clean candidates
        LOG(INFO) << "[Planner.planAll timeline] cache candidates done at " << planAllSeconds() << "s";
        Engine cpu = Engine{0, EngineType::CPU};
        auto addCacheENodes = [&](EGraph &bstate) {
            for (const auto &cand : candidates)
            {
                EClassId cid = bstate.findEClassByBaseId(cand.base_eclass_id);
                if (cid == EClassId{} || !bstate.isClean(cid))
                    continue;
                const EClass &cls = bstate.getEClass(cid);
                bool has_cache = false;
                for (ENodeId en_id : cls.enodes)
                {
                    if (bstate.getENode(en_id).getOpType() == OpType::CACHE)
                    {
                        has_cache = true;
                        break;
                    }
                }
                if (!has_cache)
                {
                    bstate.addENode(cid, ENode(KernelId{0}, OpType::CACHE, "", {}, cls.shape, cls.strides,
                                                      cls.dtype, cls.mem_space, {cpu},
                                                      std::to_string(cand.base_eclass_id.value)));
                }
            }
        };
        for (uint32_t b = 0; b < bucket_states.size(); ++b)
        {
            if (b == full_idx || bucket_weights[b] <= 0.0f)
                continue;
            addCacheENodes(bucket_states[b]);
        }

        // Compute enode infos and prune egraphs
        LOG(INFO) << "[Planner.planAll timeline] cache enodes done at " << planAllSeconds() << "s";
        std::vector<std::vector<ENodeInfo>> all_enode_infos(buckets.size());
        // Full cycle listings (egraph_cycles{0..N}.txt: before any pass, then
        // after each pass) are debug output and cost seconds per bucket to
        // format; write them only on request.
        const bool cycle_diagnostics = std::getenv("TG_CYCLE_DIAGNOSTICS") != nullptr;
        const auto &cycle_passes = cycleStripPasses();
        std::vector<std::ofstream> cycle_logs(cycle_diagnostics ? cycle_passes.size() + 1 : 0);
        for (size_t i = 0; i < cycle_logs.size(); ++i)
        {
            const std::string path = "egraph_cycles" + std::to_string(i) + ".txt";
            cycle_logs[i].open(path, std::ios::out | std::ios::trunc);
            if (!cycle_logs[i])
                LOG(WARNING) << "[Planner.planAll] Could not open " << path << " for cycle diagnostics.";
        }
        // Buckets are independent here: run them in parallel and emit their
        // diagnostics afterwards in bucket order.
        struct BucketCycleReport
        {
            std::vector<std::ostringstream> logs;    // before, then after each pass
            std::vector<uint32_t> components;        // before, then after each pass
            std::vector<uint32_t> removed;           // per pass
        };
        std::vector<BucketCycleReport> cycle_reports(buckets.size());
        auto reduceBucket = [&](uint32_t b, Planner &worker_planner) {
            BucketCycleReport &r = cycle_reports[b];
            const auto reduce_start = std::chrono::steady_clock::now();
            auto stepSeconds = [&]() {
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - reduce_start).count();
            };
            bucket_states[b].rebuild(true);
            LOG(INFO) << "[Planner.reduce] bucket=" << b << " rebuild " << stepSeconds() << "s";

            // Pruning consumes kernel costs, not branch-order estimates.
            // Build those estimates once, after reduction has finished.
            all_enode_infos[b] = worker_planner.computeENodeInfos(bucket_states[b], false);
            if (std::getenv("TG_DEBUG_ROOTS"))
            {
                const EGraph &eg = bucket_states[b];
                const EClassId root = eg.findConst(eg.findEClassByLogicalId(rootId));
                std::stringstream ss;
                for (ENodeId enodeId : eg.getEClass(root).enodes)
                {
                    const ENode &en = eg.getENode(enodeId);
                    if (en.getOpType() != OpType::SCATTER)
                        continue;
                    ss << "\n  scatter cost=" << all_enode_infos[b][enodeId.value].cost << " kernel=0x" << std::hex
                       << en.getKernelId().value << std::dec << " name=" << en.getOpName();
                    const EClassId child = eg.findConst(en.getChildren()[0]);
                    for (ENodeId ce : eg.getEClass(child).enodes)
                    {
                        const ENode &cn = eg.getENode(ce);
                        ss << "\n    child " << toString(cn.getOpType()) << ":" << cn.getOpName() << " shape="
                           << toString(cn.getShape()) << " cost=" << all_enode_infos[b][ce.value].cost;
                        for (EClassId gc : cn.getChildren())
                        {
                            const EClass &gcls = eg.getEClass(eg.findConst(gc));
                            float best = TGConstants::INF;
                            for (ENodeId ge : gcls.enodes)
                                best = std::min(best, all_enode_infos[b][ge.value].cost);
                            ss << " [grandchild " << toString(gcls.shape) << " n=" << gcls.enodes.size() << " best=" << best << "]";
                        }
                    }
                }
                LOG(INFO) << "[debug prune] bucket=" << b << ss.str();
            }
            uint32_t scatterBefore = 0, scatterInf = 0;
            if (std::getenv("TG_DEBUG_ROOTS"))
                for (uint32_t e = 0; e < bucket_states[b].getENodes().size(); ++e)
                    if (bucket_states[b].getENode(ENodeId{e}).getOpType() == OpType::SCATTER)
                    {
                        ++scatterBefore;
                        scatterInf += all_enode_infos[b][e].cost == TGConstants::INF;
                    }
            worker_planner.pruneEGraph(bucket_states[b], all_enode_infos[b]);
            LOG(INFO) << "[Planner.reduce] bucket=" << b << " infos+prune " << stepSeconds() << "s";
            if (std::getenv("TG_DEBUG_ROOTS"))
            {
                uint32_t scatterAfter = 0;
                for (size_t i = 0; i < bucket_states[b].getNumClasses(); ++i)
                {
                    EClassId cid{static_cast<uint32_t>(i)};
                    if (bucket_states[b].findConst(cid) != cid)
                        continue;
                    for (ENodeId e : bucket_states[b].getEClass(cid).enodes)
                        scatterAfter += bucket_states[b].getENode(e).getOpType() == OpType::SCATTER;
                }
                LOG(INFO) << "[debug scatter] bucket=" << b << " before=" << scatterBefore << " inf=" << scatterInf
                          << " after_prune=" << scatterAfter;
            }

            bool any_unreachable_cleared = false;
            const EClassId root = bucket_states[b].findConst(bucket_states[b].findEClassByLogicalId(rootId));
            if (root != EClassId{})
            {
                bucket_states[b].root_class_id = root;
                std::vector<EClassId> reachable_vec = plan::collectReachableCids(bucket_states[b], root);
                std::unordered_set<EClassId> reachable_set(reachable_vec.begin(), reachable_vec.end());
                for (size_t i = 0; i < bucket_states[b].getNumClasses(); ++i)
                {
                    EClassId cid{static_cast<uint32_t>(i)};
                    if (bucket_states[b].findConst(cid) != cid)
                        continue;
                    if (reachable_set.find(cid) == reachable_set.end())
                    {
                        EClass &cls = bucket_states[b].getEClass(cid);
                        if (!cls.enodes.empty())
                        {
                            cls.enodes.clear();
                            any_unreachable_cleared = true;
                        }
                    }
                }
            }

            r.logs.resize(cycle_logs.size());
            LOG(INFO) << "[Planner.reduce] bucket=" << b << " unreachable " << stepSeconds() << "s";
            r.components.push_back(countCyclicComponents(bucket_states[b]));
            if (cycle_diagnostics)
            {
                r.logs[0] << "[Planner.planAll] Cycle diagnostic for bucket " << b << ":\n";
                printEGraphCycles(bucket_states[b], r.logs[0]);
            }
            for (size_t p = 0; p < cycle_passes.size(); ++p)
            {
                r.removed.push_back(cycle_passes[p].run(bucket_states[b]));
                r.components.push_back(countCyclicComponents(bucket_states[b]));
                if (cycle_diagnostics)
                {
                    r.logs[p + 1] << "[Planner.planAll] Cycle diagnostic for bucket " << b << ":\n"
                                  << "[Planner.planAll] Removed " << r.removed.back() << " "
                                  << cycle_passes[p].description << ".\n";
                    printEGraphCycles(bucket_states[b], r.logs[p + 1]);
                }
            }

            const uint32_t total_removed = std::accumulate(r.removed.begin(), r.removed.end(), 0u);
            LOG(INFO) << "[Planner.reduce] bucket=" << b << " cycle passes " << stepSeconds() << "s";
            if (total_removed > 0 || any_unreachable_cleared || bucket_states[b].getENodes().size() != all_enode_infos[b].size())
            {
                bucket_states[b].rebuild(true, false);
                all_enode_infos[b] = worker_planner.computeENodeInfos(bucket_states[b]);
                bucket_states[b].clearHashcons();
                bucket_states[b].constantHashIndex.clear();
                bucket_states[b].classes.shrink_to_fit();
                bucket_states[b].enodes.shrink_to_fit();
            }
            else
                worker_planner.computeENodeHeuristicCosts(bucket_states[b], all_enode_infos[b]);
            LOG(INFO) << "[Planner.reduce] bucket=" << b << " final infos " << stepSeconds() << "s";
            egraph_diagnostics::logClassSizeHistogram(bucket_states[b],
                                                      "bucket=" + std::to_string(b) + " stage=pruned");
            egraph_diagnostics::dumpRequestedClasses(bucket_states[b], b);
        };
        {
            const uint32_t nb = static_cast<uint32_t>(buckets.size());
            std::vector<uint32_t> state_representatives;
            for (uint32_t b = 0; b < nb; ++b)
                if (shared_state_to[b] == b)
                    state_representatives.push_back(b);
            auto copyReduced = [&](uint32_t b) {
                const uint32_t from = shared_state_to[b];
                all_enode_infos[b] = all_enode_infos[from];
                cycle_reports[b].components = cycle_reports[from].components;
                cycle_reports[b].removed = cycle_reports[from].removed;
                cycle_reports[b].logs.resize(cycle_reports[from].logs.size());
            };
            // With one partial group (plus the full bucket), the union graph
            // is reduced on this thread first: measured faster than reducing
            // it on a worker. Several groups are all reduced concurrently.
            uint32_t first_pooled = 0;
            if (!state_representatives.empty() && state_representatives.size() <= 2)
            {
                reduceBucket(state_representatives.front(), *this);
                first_pooled = 1;
            }
            const unsigned hw = std::max(1u, std::min(std::thread::hardware_concurrency(), std::max(nb, 1u)));
            std::atomic<uint32_t> next_representative{first_pooled};
            std::vector<std::thread> threads;
            std::vector<std::exception_ptr> errors(hw);
            for (unsigned t = 0; t < hw; ++t)
                threads.emplace_back([&, t]() {
#ifdef _WIN32
                    if (SamplingProfiler::enabled())
                        SamplingProfiler::get().registerCurrentThread();
#endif
                    try
                    {
                        Planner worker_planner(costModel, settings);
                        worker_planner.cost_memo_ = cost_memo_;
                        for (uint32_t i = next_representative++; i < state_representatives.size();
                             i = next_representative++)
                            reduceBucket(state_representatives[i], worker_planner);
                    }
                    catch (...)
                    {
                        errors[t] = std::current_exception();
                    }
#ifdef _WIN32
                    if (SamplingProfiler::enabled())
                        SamplingProfiler::get().unregisterCurrentThread();
#endif
                });
            for (auto &thread : threads)
                thread.join();
            for (auto &error : errors)
                if (error)
                    std::rethrow_exception(error);
            for (uint32_t b = 0; b < nb; ++b)
                if (shared_state_to[b] != b)
                    copyReduced(b);
        }

        if (std::getenv("TG_DEBUG_ROOTS"))
        {
            for (uint32_t b = 0; b < buckets.size(); ++b)
            {
                const EGraph &eg = bucket_states[shared_state_to[b]];
                const EClassId root = eg.findConst(eg.findEClassByLogicalId(rootId));
                for (ENodeId enodeId : eg.getEClass(root).enodes)
                {
                    const ENode &en = eg.getENode(enodeId);
                    const ENodeInfo &info = all_enode_infos[b][enodeId.value];
                    LOG(INFO) << "[debug root] bucket=" << b << " op=" << toString(en.getOpType()) << ":"
                              << en.getOpName() << " cost=" << info.cost << " dag=" << info.optimistic_dag_cost
                              << " dp=" << info.dp_cost;
                }
            }
        }
        bool all_buckets_dag = true;
        for (uint32_t b = 0; b < buckets.size(); ++b)
        {
            const BucketCycleReport &r = cycle_reports[b];
            for (size_t i = 0; i < cycle_logs.size(); ++i)
                if (cycle_logs[i].is_open())
                    cycle_logs[i] << r.logs[i].str();
            if (r.components.back() > 0)
                all_buckets_dag = false;

            LOG(INFO) << "[Planner.planAll] Bucket " << b << " cycle reduction:";
            LOG(INFO) << "  Initial cyclic components: " << r.components.front();
            for (size_t p = 0; p < cycle_passes.size(); ++p)
                LOG(INFO) << "  Pass " << (p + 1) << " (" << cycle_passes[p].name << "): " << r.components[p]
                          << " -> " << r.components[p + 1] << ", removed " << r.removed[p] << " "
                          << cycle_passes[p].description;
        }
        for (std::ofstream &log : cycle_logs)
            log.flush();

        if (all_buckets_dag)
        {
            LOG(INFO) << "[Planner.planAll] All buckets are acyclic (DAG). Enabling DAG-optimized reachability.";
        }

        if (settings.saturate_only)
        {
            LOG(INFO) << "[Planner.planAll] Saturate-only mode: cost estimation complete across all kernels & shapes. Exiting before SearchEngine.";
            return {{}, {}};
        }

        // Preallocate constants and inputs
        LOG(INFO) << "[Planner.planAll timeline] reduction done at " << planAllSeconds() << "s";
        std::unordered_map<BaseEClassId, ParallelBuffer> preallocated;
        preallocate(graph, bucket_states[full_idx], preallocated);

        std::unordered_map<MemSpace, uint32_t> preallocated_pages;
        for (const auto &pair : preallocated)
        {
            uint64_t extent = static_cast<uint64_t>(pair.second.offset) + pair.second.size;
            uint32_t align = plan::SearchState::getDefaultPageAlignment(pair.second.mem_space);
            uint32_t pages = static_cast<uint32_t>((extent + align - 1) / align);
            preallocated_pages[pair.second.mem_space] = std::max(preallocated_pages[pair.second.mem_space], pages);
        }

        // 2. SEARCH
        // TODO: move to SearchState constructor
        plan::SearchState search_state;
        search_state.buckets = buckets;
        search_state.bucket_weights = bucket_weights;
        search_state.candidates = candidates;
        search_state.mem_caps = settings.mem_caps;
        search_state.preallocated_buffers = preallocated;
        search_state.preallocated_pages = preallocated_pages;

        std::vector<std::shared_ptr<EGraph>> shared_search_graphs(buckets.size());
        for (uint32_t b = 0; b < buckets.size(); ++b)
            if (shared_state_to[b] == b)
                shared_search_graphs[b] = std::make_shared<EGraph>(std::move(bucket_states[b]));
        for (uint32_t b = 0; b < buckets.size(); ++b)
        {
            const uint32_t representative = shared_state_to[b];
            if (!shared_search_graphs[representative])
                Error::throw_err("[Planner.planAll] missing reduced e-graph representative");
            search_state.bucket_root_ids.push_back(shared_search_graphs[representative]->findEClassByLogicalId(rootId));
            search_state.bucket_egraphs.pushShared(shared_search_graphs[representative]);
            search_state.bucket_enode_infos.push_back(std::move(all_enode_infos[b]));
        }

        // Variable 1: cached_<base_eclass_id> in {0, 1}
        for (const auto &cand : candidates)
        {
            plan::VarInfo vinfo;
            vinfo.type = plan::VarType::CACHED;
            vinfo.name = "cached_" + std::to_string(cand.base_eclass_id.value);
            vinfo.base_eclass_id = cand.base_eclass_id;
            vinfo.mem_space = cand.mem_space;
            vinfo.size_bytes = cand.size_bytes;
            vinfo.size_pages = search_state.bytesToPages(cand.size_bytes, cand.mem_space);

            plan::Domain dom = settings.disable_caching ? plan::Domain::makeFixed(0, true) : plan::Domain::makeMask(0b11);
            plan::VarId vid = search_state.addVar(vinfo, dom);
            search_state.cached_vars[cand.base_eclass_id] = vid;
        }

        // Variables per bucket: omit structurally unreachable classes entirely.
        LOG(INFO) << "[Planner.planAll timeline] cache variables done at " << planAllSeconds() << "s";
        search_state.addAllBucketVariables(shared_state_to);

        for (uint32_t b = 0; b < buckets.size(); ++b)
        {
            LOG(INFO) << "[Planner.planAll] Search bucket " << b
                      << ": reachable classes=" << search_state.reachable_cids[b].size()
                      << ", egraph classes=" << search_state.bucket_egraphs[b].getNumClasses()
                      << ", egraph enodes=" << search_state.bucket_egraphs[b].getNumENodes();
        }

        // Set up SearchEngine with Propagators and Selector
        LOG(DEBUG) << "[Planner.planAll] Initializing SearchEngine with " << search_state.numVars()
                   << " variables across " << buckets.size() << " bucket(s)...";

        auto selector = std::make_shared<plan::PriorityQueueSelector>();
        auto brancher_impl = brancher ? brancher : std::make_shared<plan::HeuristicBrancher>();

        auto engine_holder = std::make_unique<plan::SearchEngine>(std::move(search_state), selector, brancher_impl);
        plan::SearchEngine &engine = *engine_holder;
        plan::addAllPropagators(engine, true, all_buckets_dag);

        LOG(DEBUG) << "[Planner.planAll] Launching SearchEngine solve (minCompileSeconds=" << minCompileSeconds << "s)...";
        LOG(INFO) << "[Planner.planAll timeline] search start at " << planAllSeconds() << "s";
#ifdef _WIN32
        if (SamplingProfiler::enabled())
        {
            SamplingProfiler::get().unregisterCurrentThread();
            SamplingProfiler::get().finish("prepare");
        }
#endif
#ifdef _WIN32
        if (SamplingProfiler::enabled())
        {
            SamplingProfiler::get().start();
            SamplingProfiler::get().registerCurrentThread();
        }
#endif
        bool solved = engine.solve(minCompileSeconds);
#ifdef _WIN32
        if (SamplingProfiler::enabled())
        {
            SamplingProfiler::get().unregisterCurrentThread();
            SamplingProfiler::get().finish("search");
        }
#endif
        LOG(INFO) << "[Planner.planAll timeline] search done at " << planAllSeconds() << "s";
        LOG(DEBUG) << "[Planner.planAll] SearchEngine solve returned: solved=" << solved
                   << ", best_cost=" << engine.incumbent_best_cost;
        if (!solved && engine.incumbent_extractions.empty())
        {
            Error::throw_err("[Planner.planAll] Search failed to find a valid execution graph across buckets.");
        }

        // Diagnostics: TG_DEBUG_CLASS=logical,ids prints each planned bucket's
        // alternatives for those nodes (cost, selected, cached), with the
        // alternatives of scatter forms' restricted children.
        if (const char *debugClasses = std::getenv("TG_DEBUG_CLASS"))
        {
            std::vector<LogicalId> ids;
            std::stringstream list(debugClasses);
            std::string item;
            while (std::getline(list, item, ','))
                ids.push_back(LogicalId{static_cast<uint32_t>(std::stoul(item))});
            for (uint32_t b = 0; b < buckets.size(); ++b)
            {
                const EGraph &eg = engine.state.bucket_egraphs[b];
                const auto &infos = engine.state.bucket_enode_infos[b];
                const ExtractionResult &ex = engine.incumbent_extractions[b];
                auto describe = [&](EClassId cid, const std::string &indent) {
                    cid = eg.findConst(cid);
                    const EClass &cls = eg.getEClass(cid);
                    auto sel = ex.selection_map.find(cid);
                    std::stringstream ss;
                    ss << indent << "class " << cid.value << " " << toString(cls.shape) << " clean=" << cls.is_clean
                       << " cached=" << (cls.base_eclass_id != BaseEClassId{} &&
                                         engine.incumbent_cached_nodes.count(cls.base_eclass_id))
                       << " selected=" << (sel == ex.selection_map.end() ? -1 : static_cast<int>(sel->second));
                    for (size_t i = 0; i < cls.enodes.size(); ++i)
                    {
                        const ENode &en = eg.getENode(cls.enodes[i]);
                        ss << "\n" << indent << "  [" << i << "] " << toString(en.getOpType()) << ":" << en.getOpName()
                           << " cost=" << infos[cls.enodes[i].value].cost;
                    }
                    return ss.str();
                };
                for (LogicalId id : ids)
                {
                    const EClassId cid = eg.findEClassByLogicalId(id);
                    if (cid == EClassId{})
                        continue;
                    std::stringstream ss;
                    ss << "[debug class] bucket " << b << " logical " << id.value << "\n" << describe(cid, "  ");
                    const EClass &cls = eg.getEClass(eg.findConst(cid));
                    for (ENodeId e : cls.enodes)
                        if (eg.getENode(e).getOpType() == OpType::SCATTER)
                        {
                            const EClassId r = eg.getENode(e).getChildren()[0];
                            ss << "\n    scatter child:\n" << describe(r, "      ");
                            for (ENodeId re : eg.getEClass(eg.findConst(r)).enodes)
                                for (EClassId gc : eg.getENode(re).getChildren())
                                    ss << "\n      grandchild:\n" << describe(gc, "        ");
                            break;
                        }
                    LOG(INFO) << ss.str();
                }
            }
        }

        // 3. COMPILE GRAPHS
        std::vector<CompiledGraph> compiled_graphs;
        for (uint32_t b = 0; b < buckets.size(); ++b)
        {
            CompiledGraph cg = buildCompiledGraph(
                rootId, graph, engine.state.bucket_egraphs[b],
                engine.incumbent_extractions[b],
                engine.state.bucket_enode_infos[b]);
            cg.bucket = buckets[b];
            compiled_graphs.push_back(std::move(cg));
        }
        LOG(INFO) << "[Planner.planAll timeline] graphs built at " << planAllSeconds() << "s";

        // The search state and the saturated e-graphs take most of a second
        // to free at large bucket counts; nothing reads them after this point,
        // so free them off the compile's critical path.
        std::unordered_set<BaseEClassId> cached_nodes = std::move(engine.incumbent_cached_nodes);
        std::thread([engine_holder = std::move(engine_holder), temp_graph = std::move(temp_graph),
                     full_state = std::move(full_state), bucket_states = std::move(bucket_states),
                     all_enode_infos = std::move(all_enode_infos),
                     shared_search_graphs = std::move(shared_search_graphs)]() mutable {}).detach();
        return {std::move(compiled_graphs), std::move(cached_nodes)};
    }

    ExtractionResult extractBest(const LogicalId rootId, const Graph &graph, const EGraph &egraph,
                                 bool stopOnFirstValid = true, bool strictCache = false, float minCompileSeconds = 0.0f,
                                 std::shared_ptr<plan::Brancher> brancher = nullptr,
                                 const std::vector<ENodeInfo> &enodeInfos = {},
                                 const std::unordered_set<EClassId> *cachedEClasses = nullptr,
                                 const std::unordered_set<EClassId> *cleanEClasses = nullptr)
    {
        Bucket b;
        auto [graphs, cached] = planAll(rootId, graph, {b}, {1.0f}, false, nullptr, minCompileSeconds, brancher);
        if (graphs.empty())
            Error::throw_err("[Planner.extractBest] Failed to extract valid plan.");

        ExtractionResult res;
        for (const auto &inst : graphs[0].instructions)
        {
            res.order.push_back(inst.eclass_id);
            res.buffers.push_back(inst.outBuffer);
            res.eclass_to_buf[inst.eclass_id] = inst.outBuffer.id;
        }
        res.cost = graphs[0].cost();
        return res;
    }

    CompiledGraph plan(LogicalId rootId, const Graph &graph, const Bucket &bucket,
                       bool doSaturate = true,
                       bool strictCache = false, TGStore *repo = nullptr, float minCompileSeconds = 0.0f,
                       std::shared_ptr<plan::Brancher> brancher = nullptr)
    {
        std::vector<Bucket> buckets = {bucket};
        std::vector<float> weights = {1.0f};
        auto [graphs, cached] = planAll(rootId, graph, buckets, weights, doSaturate, repo, minCompileSeconds, brancher);
        if (graphs.empty())
            Error::throw_err("[Planner.plan] Failed to generate compiled graph.");
        return graphs[0];
    }
};
