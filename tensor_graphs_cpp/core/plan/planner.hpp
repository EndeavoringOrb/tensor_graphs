// File: tensor_graphs_cpp/core/plan/planner.hpp
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/constants.hpp"
#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/logging.hpp"
#include "core/misc.hpp"
#include "core/ops/ops.hpp"
#include "core/plan/brancher.hpp"
#include "core/plan/domain.hpp"
#include "core/plan/mem.hpp"
#include "core/plan/propagator.hpp"
#include "core/plan/search_delegate.hpp"
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

struct SaturationResult
{
    EGraph egraph;
    std::unordered_map<LogicalId, EClassId> nodeToEClass;
    std::unordered_map<EClassId, LogicalId> eclassToLogical;
    std::unordered_set<EClassId> cleanEClasses;
};

struct ENodeDominationContext
{
    const EGraph &egraph;
    const std::vector<ENodeInfo> &enodeInfos;
    const std::unordered_map<EClassId, LogicalId> &eclassToLogical;
    const std::unordered_map<MemSpace, uint64_t> &mem_caps;
};

class MemCapENodeDominationRule
{
  public:
    TG_PRUNING_RULE(MemCapENodeDominationRule)
    MemCapENodeDominationRule(bool en = true) : enabled(en)
    {
    }

    bool check(ENodeId enodeId, size_t /*idx*/, const ENodeDominationContext &ctx) const
    {
        if (!enabled)
            return false;
        const ENode &enode = ctx.egraph.getENode(enodeId);
        MemSpace ms = enode.getMemSpace();

        if (ms.type == HandleType::STORAGE || ctx.mem_caps.find(ms) == ctx.mem_caps.end())
            return false;

        uint64_t cap = ctx.mem_caps.at(ms);
        uint64_t out_size = (getSizeBytes(enode.getShape(), enode.getDType()) + 4095) & ~4095ULL;

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
                    const EClass cls = ctx.egraph.getEClass(child);
                    if (cls.mem_space == ms)
                    {
                        uint64_t in_size = (getSizeBytes(cls.shape, cls.dtype) + 4095) & ~4095ULL;
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
        std::unordered_set<EClassId> seen_children;
        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = ctx.egraph.findConst(child);
            if (seen_children.insert(canon_child).second)
            {
                const EClass cls = ctx.egraph.getEClass(canon_child);
                if (cls.mem_space == ms)
                {
                    sum_inputs_in_ms += (getSizeBytes(cls.shape, cls.dtype) + 4095) & ~4095ULL;
                }
            }
        }

        uint64_t required_mem = (can_be_inplace ? 0 : out_size) + sum_inputs_in_ms;
        return required_mem > cap;
    }
};

class FasterEquivalentENodeDominationRule
{
  public:
    TG_PRUNING_RULE(FasterEquivalentENodeDominationRule)
    FasterEquivalentENodeDominationRule(bool en = true) : enabled(en)
    {
    }

    bool check(ENodeId enodeId, size_t /*idx*/, const ENodeDominationContext &ctx) const
    {
        if (!enabled)
            return false;
        float costA = ctx.enodeInfos[enodeId.value].cost;
        if (costA == TGConstants::INF || std::isnan(costA))
            return false;

        const ENode &a = ctx.egraph.getENode(enodeId);
        EClassId e_class_id = ctx.egraph.getENodeEClass(enodeId);
        const EClass cls = ctx.egraph.getEClass(ctx.egraph.findConst(e_class_id));
        const ENodeInfo &infoA = ctx.enodeInfos[enodeId.value];

        std::vector<uint32_t> a_inplace;
        if (a.getKernelId().value != 0 && KernelRegistry::get().hasKernel(a.getKernelId()))
        {
            a_inplace = KernelRegistry::get().getKernel(a.getKernelId()).safe_inplace_idxs;
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

            std::vector<uint32_t> b_inplace;
            if (b.getKernelId().value != 0 && KernelRegistry::get().hasKernel(b.getKernelId()))
            {
                b_inplace = KernelRegistry::get().getKernel(b.getKernelId()).safe_inplace_idxs;
            }

            bool inplace_compatible = true;
            for (uint32_t in_idx : a_inplace)
            {
                if (std::find(b_inplace.begin(), b_inplace.end(), in_idx) == b_inplace.end())
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
                if (b_inplace.size() > a_inplace.size())
                {
                    return true;
                }
                if (b_inplace.size() == a_inplace.size() && otherId < enodeId)
                {
                    return true;
                }
            }
        }

        return false;
    }
};

struct CacheContext
{
    const std::vector<CacheCandidate> &candidates;
    const std::vector<uint32_t> &num_users;
    const std::vector<std::vector<int>> &valid_choices;
    const std::unordered_set<BaseEClassId> &current_cache_selection;
    uint32_t k;
    int choice;
};

template <typename... Rules> struct CacheIterator
{
    prune::PruningRuleSet<Rules...> rules;

    std::vector<CacheCandidate> candidates;
    const std::unordered_map<MemSpace, uint64_t> &mem_caps;
    std::shared_ptr<SearchDelegate> delegate;
    const float *best_cost = nullptr;
    TimeoutChecker *timeout = nullptr;

    std::vector<uint32_t> num_users;
    std::vector<std::vector<int>> valid_choices;

    int k = 0;
    bool is_done = false;
    bool first_yield = true;
    std::vector<std::vector<int>> tried_choices;
    std::unordered_set<BaseEClassId> current_cache_selection;

    template <typename... Rs>
    CacheIterator(const std::vector<CacheCandidate> &_candidates,
                  const std::unordered_map<MemSpace, uint64_t> &_mem_caps, std::shared_ptr<SearchDelegate> _delegate,
                  const float *_best_cost = nullptr, TimeoutChecker *_timeout = nullptr, Rs &&..._rules)
        : rules(std::forward<Rs>(_rules)...), candidates(_candidates), mem_caps(_mem_caps), delegate(std::move(_delegate)),
          best_cost(_best_cost), timeout(_timeout)
    {
        if (delegate && best_cost)
        {
            delegate->set_best_cost_ptr(best_cost);
        }
        init();
        CacheContext ctx{candidates, num_users, valid_choices, current_cache_selection, 0, 0};
        rules.init(ctx);
    }

    bool can_abort()
    {
        return timeout && timeout->is_expired() && (best_cost != nullptr && *best_cost < TGConstants::INF);
    }

    void init()
    {
        uint32_t N = static_cast<uint32_t>(candidates.size());
        tried_choices.resize(N);
        valid_choices.resize(N);
        num_users.assign(N, 0);

        for (uint32_t i = 0; i < N; ++i)
        {
            num_users[i] = candidates[i].num_users;
            valid_choices[i].push_back(0);
            valid_choices[i].push_back(1);
        }

        if (delegate && N > 0)
        {
            std::vector<float> node_features;
            std::vector<uint32_t> edge_src;
            std::vector<uint32_t> edge_dst;
            for (uint32_t i = 0; i < N; ++i)
            {
                const CacheCandidate &candidate = candidates[i];
                node_features.push_back(static_cast<float>(candidate.size_bytes));
                node_features.push_back(static_cast<float>(OpType::INPUT));
                node_features.push_back(static_cast<float>(candidate.dtype));
                node_features.push_back(candidate.mem_space.type == HandleType::STORAGE ? 1.0f : 0.0f);
                node_features.push_back(static_cast<float>(num_users[i]));
            }

            delegate->init_cache_graph(node_features, edge_src, edge_dst);
        }
    }

    bool ascend()
    {
        k--;
        while (k >= 0)
        {
            if (valid_choices[k].empty())
            {
                k--;
                continue;
            }

            BaseEClassId id = candidates[k].base_eclass_id;
            current_cache_selection.erase(id);

            if (tried_choices[k].size() < valid_choices[k].size())
            {
                return true;
            }

            tried_choices[k].clear();
            if (delegate && valid_choices[k].size() > 1)
            {
                delegate->pop_state();
            }
            k--;
        }
        return false;
    }

    bool getNextCacheSelection(std::unordered_set<BaseEClassId> &out_cached_nodes)
    {
        if (is_done)
            return false;

        uint32_t N = static_cast<uint32_t>(candidates.size());
        if (N == 0)
        {
            if (first_yield)
            {
                first_yield = false;
                out_cached_nodes.clear();
                return true;
            }
            is_done = true;
            return false;
        }

        if (!first_yield)
        {
            if (!ascend())
            {
                is_done = true;
                return false;
            }
        }
        first_yield = false;

        while (k >= 0)
        {
            if (can_abort())
            {
                is_done = true;
                return false;
            }

            if (k == static_cast<int>(N))
            {
                out_cached_nodes = current_cache_selection;
                return true;
            }

            if (valid_choices[k].empty())
            {
                k++;
                continue;
            }

            BaseEClassId id = candidates[k].base_eclass_id;

            std::vector<int> unexplored;
            unexplored.reserve(valid_choices[k].size());
            for (int choice : valid_choices[k])
            {
                if (std::find(tried_choices[k].begin(), tried_choices[k].end(), choice) == tried_choices[k].end())
                {
                    unexplored.push_back(choice);
                }
            }

            if (unexplored.empty())
            {
                tried_choices[k].clear();
                if (delegate && valid_choices[k].size() > 1)
                {
                    delegate->pop_state();
                }
                if (!ascend())
                {
                    is_done = true;
                    return false;
                }
                continue;
            }

            std::vector<uint32_t> relative_order;
            if (delegate && valid_choices[k].size() > 1)
            {
                delegate->push_state();

                std::vector<ActionFeatureCache> features;
                features.reserve(unexplored.size());

                uint64_t node_size = candidates[k].size_bytes;

                for (int choice : unexplored)
                {
                    ActionFeatureCache f;
                    f.size = node_size;
                    f.num_users = static_cast<float>(num_users[k]);
                    f.logical_id = id.value;

                    if (choice == 0)
                    {
                        f.is_cached = 0.0f;
                        f.mem_space = MemSpace{0, HandleType::STORAGE};
                        f.mem_cap = 0;
                    }
                    else
                    {
                        f.is_cached = 1.0f;
                        f.mem_space = candidates[k].mem_space;
                        auto cap_it = mem_caps.find(f.mem_space);
                        f.mem_cap = (cap_it != mem_caps.end()) ? cap_it->second : 0;
                    }
                    features.push_back(f);
                }

                relative_order = delegate->order_cache(features);
            }
            else
            {
                relative_order.resize(unexplored.size());
                std::iota(relative_order.begin(), relative_order.end(), 0u);
            }

            bool chosen = false;
            for (uint32_t rel_idx : relative_order)
            {
                int choice = unexplored[rel_idx];
                tried_choices[k].push_back(choice);

                CacheContext ctx{candidates, num_users, valid_choices, current_cache_selection,
                                 static_cast<uint32_t>(k), choice};
                if (rules.is_pruned(choice, static_cast<size_t>(rel_idx), ctx))
                {
                    continue;
                }

                if (choice > 0)
                {
                    current_cache_selection.insert(id);
                }
                else
                {
                    current_cache_selection.erase(id);
                }

                chosen = true;
                k++;
                break;
            }

            if (!chosen)
            {
                tried_choices[k].clear();
                if (delegate && valid_choices[k].size() > 1)
                {
                    delegate->pop_state();
                }
                if (delegate && delegate->fast_fail())
                {
                    is_done = true;
                    return false;
                }
                if (!ascend())
                {
                    is_done = true;
                    return false;
                }
            }
        }

        is_done = true;
        return false;
    }
};

template <typename... Rules>
CacheIterator<std::decay_t<Rules>...> makeCacheIterator(const std::vector<CacheCandidate> &candidates,
                                                        const std::unordered_map<MemSpace, uint64_t> &mem_caps,
                                                        const float *best_cost = nullptr,
                                                        TimeoutChecker *timeout = nullptr, Rules &&...rules)
{
    return CacheIterator<std::decay_t<Rules>...>(candidates, mem_caps, nullptr, best_cost,
                                                 timeout, std::forward<Rules>(rules)...);
}

template <typename... Rules>
CacheIterator<std::decay_t<Rules>...> makeCacheIterator(const std::vector<CacheCandidate> &candidates,
                                                        const float *best_cost = nullptr,
                                                        TimeoutChecker *timeout = nullptr, Rules &&...rules)
{
    static const std::unordered_map<MemSpace, uint64_t> empty_caps;
    return CacheIterator<std::decay_t<Rules>...>(candidates, empty_caps, nullptr, best_cost,
                                                 timeout, std::forward<Rules>(rules)...);
}

template <typename... Rules>
CacheIterator<std::decay_t<Rules>...> makeCacheIteratorWithDelegate(
    const std::vector<CacheCandidate> &candidates,
    const std::unordered_map<MemSpace, uint64_t> &mem_caps, std::shared_ptr<SearchDelegate> delegate,
    const float *best_cost = nullptr, TimeoutChecker *timeout = nullptr, Rules &&...rules)
{
    return CacheIterator<std::decay_t<Rules>...>(candidates, mem_caps, std::move(delegate),
                                                 best_cost, timeout, std::forward<Rules>(rules)...);
}

using AllCacheRuleTypes = std::tuple<>;

template <typename BoolTuple>
inline auto makeConfiguredCacheIteratorFromBools(const std::vector<CacheCandidate> &candidates,
                                                 const std::unordered_map<MemSpace, uint64_t> &mem_caps,
                                                 std::shared_ptr<SearchDelegate> delegate, const BoolTuple &bool_flags,
                                                 const float *best_cost = nullptr, TimeoutChecker *timeout = nullptr)
{
    return std::apply(
        [&](auto &&...rs) {
            return makeCacheIteratorWithDelegate(candidates, mem_caps, std::move(delegate),
                                                 best_cost, timeout, rs...);
        },
        prune::instantiate_from_bools<AllCacheRuleTypes>(bool_flags));
}

inline auto makeConfiguredCacheIterator(const std::vector<CacheCandidate> &candidates,
                                        std::shared_ptr<SearchDelegate> delegate, const Settings &settings,
                                        const float *best_cost = nullptr, TimeoutChecker *timeout = nullptr)
{
    settings.validate_rules("cache");
    auto bool_flags = prune::extract_enabled_states<AllCacheRuleTypes>("cache", settings);
    return makeConfiguredCacheIteratorFromBools(candidates, settings.mem_caps,
                                                std::move(delegate), bool_flags, best_cost, timeout);
}

inline auto makeConfiguredCacheIterator(const std::vector<CacheCandidate> &candidates, const Settings &settings,
                                        const float *best_cost = nullptr, TimeoutChecker *timeout = nullptr)
{
    return makeConfiguredCacheIterator(candidates, nullptr, settings, best_cost, timeout);
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

using AllENodeDominationRuleTypes = std::tuple<MemCapENodeDominationRule, FasterEquivalentENodeDominationRule>;

struct Planner
{
    CostModel &costModel;
    prune::PruningRuleSet<MemCapENodeDominationRule, FasterEquivalentENodeDominationRule> domination_rules;
    const Settings &settings;

    struct BaseEGraphState
    {
        EGraph egraph;
        std::unordered_map<LogicalId, EClassId> nodeToEClass;
        std::unordered_map<EClassId, LogicalId> eclassToLogical;
    };

    BaseEGraphState baseState;
    bool baseStateInitialized = false;

    Planner(CostModel &costModel, const Settings &settings = Settings::get_default())
        : costModel(costModel),
          domination_rules(prune::instantiate_rules<AllENodeDominationRuleTypes>("enode", settings)),
          settings(settings)
    {
    }

    void applyDominationRules(const EGraph &egraph, std::vector<ENodeInfo> &enodeInfos,
                              const std::unordered_map<EClassId, LogicalId> &eclassToLogical)
    {
        ENodeDominationContext ctx{egraph, enodeInfos, eclassToLogical, settings.mem_caps};
        for (uint32_t i = 0; i < egraph.getENodes().size(); ++i)
        {
            ENodeId enodeId{i};
            if (enodeInfos[i].cost == TGConstants::INF)
                continue;
            if (domination_rules.is_pruned(enodeId, /*cand_idx=*/size_t{0}, ctx))
            {
                enodeInfos[i].cost = TGConstants::INF;
            }
        }
    }

    void preallocate(const Graph &graph, const EGraph &egraph,
                     const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                     const std::unordered_set<BaseEClassId> &cachedNodes,
                     std::unordered_map<BaseEClassId, ParallelBuffer> &out) const
    {
        out.clear();

        struct PreAllocEntry
        {
            BaseEClassId baseEClassId;
            MemSpace memSpace;
            std::vector<uint32_t> shape;
            DType dtype;
        };
        std::vector<PreAllocEntry> entries;

        MemSpace storage = MemSpace{0, HandleType::STORAGE};
        MemSpace ram = MemSpace{1, HandleType::CPP};

        auto add_input = [&](const TensorNode &node, LogicalId logicalId) {
            auto nodeIt = nodeToEClass.find(logicalId);
            if (nodeIt == nodeToEClass.end())
                return;
            const EClass &cls = egraph.getEClass(nodeIt->second);
            if (cls.base_eclass_id == BaseEClassId{} || cls.mem_space == storage)
                return;
            entries.push_back({cls.base_eclass_id, ram, node.getShape(), node.dtype});
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

        for (BaseEClassId baseEClassId : cachedNodes)
        {
            EClassId eclassId = egraph.findEClassByBaseId(baseEClassId);
            if (eclassId == EClassId{})
                continue;
            const EClass &cls = egraph.getEClass(eclassId);
            if (cls.mem_space == storage)
                continue;
            entries.push_back({baseEClassId, cls.mem_space, cls.shape, cls.dtype});
        }

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

            uint64_t size_bytes = getSizeBytes(e.shape, e.dtype);
            if (size_bytes == 0)
                continue;
            size_bytes = (size_bytes + 4095) & ~4095ULL;

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

    void saturate(EGraph &egraph, const std::unordered_set<EClassId> &protectedEClasses,
                  std::unordered_map<EClassId, LogicalId> &eclassToLogical, bool injected,
                  bool allowPushDownOnProtected = false, TGStore *repo = nullptr)
    {
        RuleCtx ctx{egraph, protectedEClasses, eclassToLogical, repo, &costModel};
        std::vector<std::unique_ptr<Rule>> rules;
        rules.emplace_back(makeProfiledRewriteRule<FusionRule>());
        rules.emplace_back(makeProfiledRewriteRule<DotSplitRule>());
        rules.emplace_back(makeProfiledRewriteRule<RemoveContiguous>());
        rules.emplace_back(makeProfiledRewriteRule<RemoveCopyChains>());
        rules.emplace_back(makeProfiledRewriteRule<ConsumerWeightReuseRule>());
        rules.emplace_back(makeProfiledRewriteRule<RemoveRedundantReshape>());
        if (injected)
        {
            rules.emplace_back(makeProfiledRewriteRule<InfinityDomination>());
            rules.emplace_back(makeProfiledRewriteRule<SlicePushDownElementwise>(allowPushDownOnProtected));
            rules.emplace_back(makeProfiledRewriteRule<SlicePushDownDot>(allowPushDownOnProtected));
        }

        std::map<std::string, uint32_t> ruleMatchCounts;
        uint64_t iterations = 0;
        bool changed = true;
        uint32_t nMatches = 0;
        ProgressTimer timer(0, "saturating");
        while (changed)
        {
            iterations++;
            uint32_t preUniqueNodes = egraph.getNumUniqueENodes();
            for (uint32_t eNodeIdx = 0; eNodeIdx < egraph.getENodes().size(); eNodeIdx++)
            {
                for (const auto &rule : rules)
                {
                    bool matched = rule->match(eNodeIdx, ctx);
                    if (!matched)
                        continue;
                    rule->apply(eNodeIdx, ctx);
                    changed = true;
                    ruleMatchCounts[rule->name()]++;
                    nMatches++;
                }
            }
            egraph.rebuild();
            uint32_t postUniqueNodes = egraph.getNumUniqueENodes();
            changed = preUniqueNodes != postUniqueNodes;
            std::stringstream ss;
            ss << "\n--- Saturation Summary (" << iterations << " iterations) ---" << std::endl;
            for (auto const &[name, count] : ruleMatchCounts)
            {
                ss << "  " << name << ": " << count << " matches\n";
            }
            ss << "Total Matches: " << nMatches;
            LOG(DEBUG) << ss.str();
            if (!changed)
            {
                LOG(INFO) << ss.str();
            }
            timer.tick();
        }
    }

    uint32_t deathCascade(EGraph &egraph)
    {
        uint32_t numClasses = egraph.getClasses().size();
        std::vector<bool> enode_valid(egraph.getENodes().size(), false);
        std::vector<uint32_t> valid_enode_count(numClasses, 0);
        std::vector<std::vector<ENodeId>> parents_map(numClasses);

        for (uint32_t i = 0; i < numClasses; ++i)
        {
            EClassId e_class_id = egraph.find(EClassId{i});
            if (e_class_id != EClassId{i})
                continue;

            const EClass &cls = egraph.getEClass(e_class_id);
            valid_enode_count[e_class_id.value] = static_cast<uint32_t>(cls.enodes.size());

            for (ENodeId enodeId : cls.enodes)
            {
                enode_valid[enodeId.value] = true;
                const ENode &enode = egraph.getENode(enodeId);
                for (EClassId child : enode.getChildren())
                {
                    EClassId canon_child = egraph.findConst(child);
                    parents_map[canon_child.value].push_back(enodeId);
                }
            }
        }

        std::vector<EClassId> dead_worklist;
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

            for (ENodeId parent_enode_id : parents_map[dead_cls.value])
            {
                if (enode_valid[parent_enode_id.value])
                {
                    enode_valid[parent_enode_id.value] = false;
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

        return cascadePruned;
    }

    void pruneEGraph(EGraph &egraph, const std::vector<ENodeInfo> &enodeInfos)
    {
        uint32_t totalPruned = 0;
        for (uint32_t i = 0; i < egraph.getClasses().size(); ++i)
        {
            EClassId e_class_id = egraph.find(EClassId{i});
            if (e_class_id != EClassId{i})
                continue;

            EClass &cls = egraph.getEClass(e_class_id);
            std::vector<ENodeId> validEnodes;
            validEnodes.reserve(cls.enodes.size());

            for (ENodeId enodeId : cls.enodes)
            {
                if (enodeId.value < enodeInfos.size() && enodeInfos[enodeId.value].cost != TGConstants::INF)
                {
                    validEnodes.push_back(enodeId);
                }
            }

            totalPruned += (cls.enodes.size() - validEnodes.size());
            cls.enodes = std::move(validEnodes);
        }

        totalPruned += deathCascade(egraph);

        if (totalPruned > 0)
        {
            LOG(DEBUG) << "[Planner.pruneEGraph] Pruned " << totalPruned << " dominated enodes from the search space.";
        }
    }

    std::vector<ENodeInfo> computeENodeInfos(const EGraph &egraph,
                                             const std::unordered_map<EClassId, LogicalId> &eclassToLogical,
                                             const std::unordered_set<BaseEClassId> &cachedNodes, bool strictCache)
    {
        std::vector<ENodeInfo> enodeInfos(egraph.getENodes().size());

        for (uint32_t i = 0; i < egraph.getENodes().size(); ++i)
        {
            const ENode &enode = egraph.getENodes()[i];
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
                if (strictCache && enode.getOpType() == OpType::CACHE)
                {
                    EClassId e_class_id = egraph.getENodeEClass(ENodeId{i});
                    EClassId canonId = egraph.findConst(e_class_id);
                    const EClass &cls = egraph.getEClass(canonId);
                    if (cls.base_eclass_id == BaseEClassId{} || cachedNodes.count(cls.base_eclass_id) == 0)
                        info.cost = TGConstants::INF;
                    else if (enode.getMemSpace() != cls.mem_space)
                        info.cost = TGConstants::INF;
                }
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

                info.cost = costModel.estimateCost(enode.getKernelId(), enode.getShape(), enode.getStrides(),
                                                   enode.getDType(), inShapes, inStrides, inDTypes, inConstants);
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

        applyDominationRules(egraph, enodeInfos, eclassToLogical);

        return enodeInfos;
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
        baseState.nodeToEClass.reserve(graph.nodes.size());

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
            EClassId e_class_id = baseState.egraph.addEClass(node.getShape(), node.strides, node.dtype, mem_space);
            baseState.nodeToEClass[nodeId] = e_class_id;
            if (graph.constantStaging.count(nodeId))
            {
                baseState.egraph.constantStaging[e_class_id] = graph.constantStaging.at(nodeId);
                uint64_t dataHash = tg_hash::computeConstantHash(node.getShape(), node.strides, node.dtype,
                                                                 *graph.constantStaging.at(nodeId));
                baseState.egraph.constantHashIndex[dataHash].push_back(e_class_id);
            }
        }

        for (LogicalId nodeId : topo)
        {
            const TensorNode &node = graph.getNode(nodeId);
            EClassId e_class_id = baseState.nodeToEClass[nodeId];

            if (node.opType == OpType::INPUT)
            {
                std::vector<EClassId> children;
                for (LogicalId pid : node.child_ids)
                    children.push_back(baseState.egraph.findConst(baseState.nodeToEClass[pid]));

                std::string contentHash = node.contentHash;
                if (graph.getInputDataType(nodeId) == InputDataType::RUNTIME)
                    contentHash = toString(nodeId);

                ENode enode =
                    ENode(KernelId{0}, node.opType, node.opName, children, node.getShape(), node.strides, node.dtype,
                          graph.getInputDataType(nodeId) == InputDataType::STORAGE ? storage : ram, {cpu}, contentHash,
                          0, node.debugOrigin);
                baseState.egraph.addENode(e_class_id, enode);
                continue;
            }

            std::vector<TensorNode> inputs;
            std::vector<MemSpace> input_mem_spaces;
            for (LogicalId pid : node.child_ids)
            {
                inputs.push_back(graph.getNode(pid));
                EClassId pid_eclass = baseState.egraph.findConst(baseState.nodeToEClass[pid]);
                input_mem_spaces.push_back(baseState.egraph.getEClass(pid_eclass).mem_space);
            }

            bool ignore_in_ms = (node.opType != OpType::COPY_TO);
            std::vector<KernelId> refs =
                KernelRegistry::get().findMatchingKernels(node.opType, node.opName, inputs, node, true, ram,
                                                          input_mem_spaces, {cpu}, false, ignore_in_ms, false, true);

            if (refs.empty())
            {
                Error::throw_err("[Planner.initBaseEGraph] couldn't find any kernels to init EClass " +
                                 toString(e_class_id));
            }

            for (KernelId uid : refs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                std::vector<EClassId> children;
                for (LogicalId pid : node.child_ids)
                {
                    children.push_back(baseState.egraph.findConst(baseState.nodeToEClass[pid]));
                }
                ENode enode = ENode(uid, node.opType, node.opName, children, node.getShape(), node.strides, node.dtype,
                                    ram, {cpu}, "", 0, node.debugOrigin);
                baseState.egraph.addENode(e_class_id, enode);
            }
        }

        for (const auto &pair : baseState.nodeToEClass)
        {
            baseState.eclassToLogical[baseState.egraph.findConst(pair.second)] = pair.first;
        }

        baseState.egraph.rebuild();
        baseState.egraph.populateBaseEClassIds();
        baseStateInitialized = true;
    }

    bool injectPartialPath(EGraph &egraph, const Graph &graph, LogicalId logicalId,
                           const std::vector<Region> &dirtyRegions, const std::unordered_set<BaseEClassId> &cachedNodes,
                           const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                           std::unordered_map<EClassId, LogicalId> &eclassToLogical)
    {
        if (dirtyRegions.empty() || nodeToEClass.find(logicalId) == nodeToEClass.end())
            return false;

        EClassId E_L = egraph.findConst(nodeToEClass.at(logicalId));
        const EClass lClass = egraph.getEClass(E_L);
        const TensorNode &sourceNode = graph.getNode(logicalId);

        std::vector<Region> canonRegions = normalizeRegions(dirtyRegions);
        MemSpace target_mem_space = lClass.mem_space;
        Engine cpu = Engine{0, EngineType::CPU};
        MemSpace ram = MemSpace{1, HandleType::CPP};

        auto addConst = [&](const std::vector<int32_t> &data) -> EClassId {
            std::vector<uint32_t> shape = {static_cast<uint32_t>(data.size())};
            std::vector<uint64_t> strides = {1};
            uint64_t hash = tg_hash::computeConstantHash(shape, strides, DType::INT32,
                                                         reinterpret_cast<const uint8_t *>(data.data()),
                                                         data.size() * sizeof(int32_t));
            auto it = egraph.constantHashIndex.find(hash);
            if (it != egraph.constantHashIndex.end() && !it->second.empty())
                return egraph.findConst(it->second[0]);

            EClassId cid = egraph.addEClass(shape, strides, DType::INT32, ram);
            ENode node(KernelId{0}, OpType::INPUT, "", {}, shape, strides, DType::INT32, ram, {cpu});
            egraph.addENode(cid, node);
            auto buf = std::make_shared<std::vector<uint8_t>>(data.size() * sizeof(int32_t));
            std::memcpy(buf->data(), data.data(), buf->size());
            egraph.constantStaging[cid] = buf;
            egraph.constantHashIndex[hash].push_back(cid);
            return cid;
        };

        EClassId current_E = E_L;
        for (const Region &reg : canonRegions)
        {
            std::vector<uint32_t> partialShape;
            std::vector<int32_t> starts, ends, steps;
            for (const Dim &d : reg.region)
            {
                starts.push_back(static_cast<int32_t>(d.start));
                ends.push_back(static_cast<int32_t>(d.stop));
                steps.push_back(1);
                partialShape.push_back(d.stop - d.start);
            }

            EClassId startsId = addConst(starts);
            EClassId endsId = addConst(ends);
            EClassId stepsId = addConst(steps);

            EClassId slicedEClass = egraph.addEClass(partialShape, calcContiguousStrides(partialShape),
                                                    sourceNode.dtype, target_mem_space);

            if (sourceNode.opType == OpType::INPUT)
            {
                std::string contentHash = sourceNode.contentHash + "_slice";
                ENode inputNode(KernelId{0}, OpType::INPUT, "", {}, partialShape, calcContiguousStrides(partialShape),
                                sourceNode.dtype, target_mem_space, {cpu}, contentHash);
                egraph.addENode(slicedEClass, inputNode);
            }

            EClassId scatterEClass = egraph.addEClass(lClass.shape, lClass.strides, lClass.dtype, target_mem_space);
            EClassId shapeId = addConst(std::vector<int32_t>(lClass.shape.begin(), lClass.shape.end()));

            std::vector<TensorNode> sIns(5);
            sIns[0].setShape(partialShape);
            sIns[0].dtype = lClass.dtype;
            sIns[1].setShape({(uint32_t)starts.size()});
            sIns[1].dtype = DType::INT32;
            sIns[2].setShape({(uint32_t)ends.size()});
            sIns[2].dtype = DType::INT32;
            sIns[3].setShape({(uint32_t)steps.size()});
            sIns[3].dtype = DType::INT32;
            sIns[4].setShape({(uint32_t)lClass.shape.size()});
            sIns[4].dtype = DType::INT32;

            TensorNode sOut;
            sOut.setShape(lClass.shape);
            sOut.dtype = lClass.dtype;

            std::vector<MemSpace> scatterInputSpaces = {target_mem_space, ram, ram, ram, ram};
            auto scatterRefs = KernelRegistry::get().findMatchingKernels(OpType::SCATTER, "", sIns, sOut, true,
                                                                         target_mem_space, scatterInputSpaces, {cpu});
            for (KernelId uid : scatterRefs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                std::vector<uint64_t> strides = (kernel.is_view) ? lClass.strides : calcContiguousStrides(lClass.shape);
                ENode sn(uid, OpType::SCATTER, "", {slicedEClass, startsId, endsId, stepsId, shapeId}, lClass.shape,
                         strides, lClass.dtype, target_mem_space, {cpu});
                egraph.addENode(scatterEClass, sn);
            }

            current_E = scatterEClass;
        }

        egraph.merge(E_L, current_E);
        eclassToLogical[egraph.find(E_L)] = logicalId;
        return true;
    }

    bool injectInputPartialPaths(EGraph &egraph, const Graph &graph,
                                 const std::unordered_map<LogicalId, std::vector<Region>> &dirtyOutputRegions,
                                 const std::unordered_set<BaseEClassId> &cachedNodes,
                                 const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                                 std::unordered_map<EClassId, LogicalId> &eclassToLogical)
    {
        bool injected = false;
        for (const auto &kv : dirtyOutputRegions)
        {
            LogicalId nodeId = kv.first;
            if (!graph.hasNode(nodeId) || !nodeToEClass.count(nodeId))
                continue;
            const TensorNode &node = graph.getNode(nodeId);
            if (node.opType == OpType::INPUT && graph.constantStaging.count(nodeId) == 0 && !kv.second.empty())
            {
                injected = injectPartialPath(egraph, graph, nodeId, kv.second, cachedNodes, nodeToEClass,
                                             eclassToLogical) ||
                           injected;
            }
        }
        if (injected)
            egraph.rebuild();
        return injected;
    }

    bool injectOutputPartialPaths(EGraph &egraph, const Graph &graph, LogicalId rootId,
                                  const std::vector<Region> &outputNeeded,
                                  const std::unordered_set<BaseEClassId> &cachedNodes,
                                  const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                                  std::unordered_map<EClassId, LogicalId> &eclassToLogical)
    {
        bool injected = false;
        if (!outputNeeded.empty() && nodeToEClass.count(rootId))
        {
            injected = injectPartialPath(egraph, graph, rootId, outputNeeded, cachedNodes, nodeToEClass,
                                         eclassToLogical);
        }
        if (injected)
            egraph.rebuild();
        return injected;
    }

    SaturationResult saturateBucket(const LogicalId rootId, const Graph &graph, const Bucket &bucket,
                                    const std::unordered_set<BaseEClassId> &cachedNodes = {}, bool doSaturate = true,
                                    TGStore *repo = nullptr, const SaturationResult *startingState = nullptr)
    {
        SaturationResult result;
        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);
        bool base_state_has_base_ids = false;

        if (startingState)
        {
            result.egraph = startingState->egraph;
            result.nodeToEClass = startingState->nodeToEClass;
            result.eclassToLogical = startingState->eclassToLogical;
        }
        else
        {
            Graph tempGraph = graph;
            initBaseEGraph(rootId, tempGraph, topo, repo, false);
            result.egraph = baseState.egraph;
            result.nodeToEClass = baseState.nodeToEClass;
            result.eclassToLogical = baseState.eclassToLogical;
        }

        for (const EClass &cls : result.egraph.getClasses())
        {
            if (result.egraph.findConst(cls.id) == cls.id && cls.base_eclass_id != BaseEClassId{})
            {
                base_state_has_base_ids = true;
                break;
            }
        }

        std::unordered_map<LogicalId, bool> logicalDirty;
        for (LogicalId nodeId : topo)
        {
            bool dirty = bucket.inputDirtyRegions.count(nodeId) && !bucket.inputDirtyRegions.at(nodeId).empty();
            if (!dirty)
            {
                for (LogicalId child : graph.getNode(nodeId).child_ids)
                {
                    if (logicalDirty[child])
                    {
                        dirty = true;
                        break;
                    }
                }
            }
            logicalDirty[nodeId] = dirty;
        }

        Engine cpu = Engine{0, EngineType::CPU};
        for (BaseEClassId baseEClassId : cachedNodes)
        {
            EClassId eclassId = result.egraph.findEClassByBaseId(baseEClassId);
            if (eclassId == EClassId{})
                continue;

            const EClass cls = result.egraph.getEClass(eclassId);
            bool hasCache = false;
            for (ENodeId enodeId : cls.enodes)
            {
                if (result.egraph.getENode(enodeId).getOpType() == OpType::CACHE &&
                    result.egraph.getENode(enodeId).getMemSpace() == cls.mem_space)
                {
                    hasCache = true;
                    break;
                }
            }
            if (!hasCache)
            {
                ENode cacheNode(KernelId{0}, OpType::CACHE, "", {}, cls.shape, cls.strides, cls.dtype, cls.mem_space,
                                {cpu}, std::to_string(baseEClassId.value));
                result.egraph.addENode(eclassId, cacheNode);
            }
        }

        std::unordered_set<EClassId> protectedEClasses;
        for (BaseEClassId baseEClassId : cachedNodes)
        {
            EClassId eclassId = result.egraph.findEClassByBaseId(baseEClassId);
            if (eclassId != EClassId{})
                protectedEClasses.insert(eclassId);
        }

        const bool dirtyInjected =
            injectInputPartialPaths(result.egraph, graph, bucket.inputDirtyRegions, cachedNodes, result.nodeToEClass,
                                    result.eclassToLogical);
        const bool neededInjected = injectOutputPartialPaths(result.egraph, graph, rootId, bucket.outputNeededRegion,
                                                             cachedNodes, result.nodeToEClass, result.eclassToLogical);

        if (doSaturate && settings.do_saturate && (!base_state_has_base_ids || dirtyInjected || neededInjected))
            saturate(result.egraph, protectedEClasses, result.eclassToLogical, true, false, repo);

        std::unordered_map<EClassId, LogicalId> canonicalLogical;
        for (const auto &kv : result.eclassToLogical)
            canonicalLogical[result.egraph.findConst(kv.first)] = kv.second;
        result.eclassToLogical = std::move(canonicalLogical);

        const uint32_t maxClasses = static_cast<uint32_t>(result.egraph.getClasses().size());
        std::vector<uint8_t> clean(maxClasses, 0);
        for (uint32_t i = 0; i < maxClasses; ++i)
        {
            EClassId id{i};
            if (result.egraph.findConst(id) != id)
                continue;
            auto logicalIt = result.eclassToLogical.find(id);
            if (logicalIt != result.eclassToLogical.end() && !logicalDirty[logicalIt->second])
                clean[i] = 1;
            if (result.egraph.constantStaging.count(id))
                clean[i] = 1;
            for (ENodeId enodeId : result.egraph.getEClass(id).enodes)
            {
                if (result.egraph.getENode(enodeId).getOpType() == OpType::CACHE)
                    clean[i] = 1;
            }
        }
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (uint32_t i = 0; i < maxClasses; ++i)
            {
                EClassId id{i};
                if (result.egraph.findConst(id) != id || clean[i])
                    continue;
                for (ENodeId enodeId : result.egraph.getEClass(id).enodes)
                {
                    const ENode &enode = result.egraph.getENode(enodeId);
                    if (enode.getOpType() == OpType::INPUT)
                        continue;
                    bool allChildrenClean = true;
                    for (EClassId child : enode.getChildren())
                    {
                        EClassId canonChild = result.egraph.findConst(child);
                        allChildrenClean =
                            allChildrenClean && canonChild.value < clean.size() && clean[canonChild.value];
                    }
                    if (allChildrenClean)
                    {
                        clean[i] = 1;
                        changed = true;
                        break;
                    }
                }
            }
        }
        for (uint32_t i = 0; i < maxClasses; ++i)
        {
            if (clean[i])
                result.cleanEClasses.insert(result.egraph.findConst(EClassId{i}));
        }
        for (auto &kv : result.nodeToEClass)
            kv.second = result.egraph.findConst(kv.second);
        if (!startingState && !base_state_has_base_ids)
            result.egraph.populateBaseEClassIds();
        return result;
    }

    CompiledGraph buildCompiledGraph(LogicalId rootId, const Graph &graph, const EGraph &egraph,
                                     const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                                     const ExtractionResult &extraction,
                                     const std::unordered_map<EClassId, LogicalId> &eclassToLogical,
                                     const std::vector<ENodeInfo> &enodeInfos)
    {
        CompiledGraph compiled;

        for (EClassId eclass_id : extraction.order)
        {
            const ENode &enode =
                egraph.getENode(egraph.getEClass(eclass_id).enodes[extraction.selection_map.at(eclass_id)]);

            LogicalId logical_id;
            if (eclassToLogical.count(eclass_id))
            {
                logical_id = eclassToLogical.at(eclass_id);
            }
            else
            {
                EClassId base_eclass = resolve_view_alias(eclass_id, egraph, extraction.selection_map, enodeInfos);
                if (eclassToLogical.count(base_eclass))
                {
                    logical_id = eclassToLogical.at(base_eclass);
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
            inst.kernel_id = enode.getKernelId();

            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = egraph.findConst(child);
                inst.children.push_back(canon_child);

                if (!compiled.eclass_to_logical.count(canon_child))
                {
                    if (eclassToLogical.count(canon_child))
                    {
                        compiled.eclass_to_logical[canon_child] = eclassToLogical.at(canon_child);
                        compiled.logical_to_eclass[eclassToLogical.at(canon_child)] = canon_child;
                    }
                    else
                    {
                        EClassId base_child =
                            resolve_view_alias(canon_child, egraph, extraction.selection_map, enodeInfos);
                        if (eclassToLogical.count(base_child))
                        {
                            compiled.eclass_to_logical[canon_child] = eclassToLogical.at(base_child);
                            compiled.logical_to_eclass[eclassToLogical.at(base_child)] = canon_child;
                        }
                    }
                }
            }
            inst.inBuffers.resize(inst.children.size());

            auto out_buf_it = extraction.eclass_to_buf.find(eclass_id);
            BufferId out_buf_id =
                (out_buf_it != extraction.eclass_to_buf.end()) ? out_buf_it->second : BufferId{UINT32_MAX};

            for (uint32_t i = 0; i < extraction.buffers.size(); i++)
            {
                if (out_buf_id.value != UINT32_MAX && extraction.buffers[i].id == out_buf_id)
                {
                    inst.outBuffer = extraction.buffers[i];
                }
                for (uint32_t j = 0; j < inst.children.size(); j++)
                {
                    auto in_buf_it = extraction.eclass_to_buf.find(inst.children[j]);
                    if (in_buf_it != extraction.eclass_to_buf.end() && extraction.buffers[i].id == in_buf_it->second)
                    {
                        inst.inBuffers[j] = extraction.buffers[i];
                    }
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
                        else if (eclassToLogical.count(child_id) &&
                                 graph.constantStaging.count(eclassToLogical.at(child_id)))
                        {
                            tempGraph.constantStaging[fakeId] = graph.constantStaging.at(eclassToLogical.at(child_id));
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

        for (const auto &pair : nodeToEClass)
        {
            LogicalId lid = pair.first;
            EClassId cid = egraph.findConst(pair.second);
            compiled.logical_to_eclass[lid] = cid;
            if (!compiled.eclass_to_logical.count(cid))
            {
                compiled.eclass_to_logical[cid] = lid;
            }

            if (!compiled.nodeViews.count(cid))
            {
                auto out_buf_it = extraction.eclass_to_buf.find(cid);
                if (out_buf_it != extraction.eclass_to_buf.end())
                {
                    BufferId buf_id = out_buf_it->second;
                    for (const auto &buf : extraction.buffers)
                    {
                        if (buf.id == buf_id)
                        {
                            const EClass &cls = egraph.getEClass(cid);
                            compiled.nodeViews[cid] =
                                TensorView(cls.shape, buf.offset >= 0 ? buf.offset : 0, cls.strides, cls.dtype);
                            break;
                        }
                    }
                }
            }
        }

        for (const auto &kv : eclassToLogical)
        {
            if (!compiled.eclass_to_logical.count(kv.first))
            {
                compiled.eclass_to_logical[kv.first] = kv.second;
            }
            if (!compiled.logical_to_eclass.count(kv.second))
            {
                compiled.logical_to_eclass[kv.second] = kv.first;
            }
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
        // 1. SATURATE
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

        const SaturationResult full_state = saturateBucket(rootId, graph, buckets[full_idx], {}, doSaturate, repo);
        std::vector<SaturationResult> bucket_states(buckets.size());
        bucket_states[full_idx] = full_state;

        for (uint32_t b = 0; b < buckets.size(); ++b)
        {
            if (b != full_idx)
            {
                bucket_states[b] = saturateBucket(rootId, graph, buckets[b], {}, false, repo, &full_state);
            }
        }

        // Cache candidate discovery across saturated egraphs
        std::vector<CacheCandidate> candidates;
        if (!settings.disable_caching)
        {
            std::unordered_map<LogicalId, uint32_t> user_counts;
            for (const auto &pair : graph.nodes)
            {
                for (LogicalId child_id : pair.second.child_ids)
                    user_counts[child_id]++;
            }

            for (const EClass &cls : full_state.egraph.getClasses())
            {
                if (full_state.egraph.findConst(cls.id) != cls.id || cls.base_eclass_id == BaseEClassId{} ||
                    cls.mem_space.type == HandleType::STORAGE || getSizeBytes(cls.shape, cls.dtype) == 0)
                    continue;

                bool clean_in_any = false;
                for (const auto &bstate : bucket_states)
                {
                    EClassId bid = bstate.egraph.findEClassByBaseId(cls.base_eclass_id);
                    if (bid != EClassId{} && bstate.cleanEClasses.count(bid))
                    {
                        clean_in_any = true;
                        break;
                    }
                }
                bool runtime_input = false;
                auto log_it = full_state.eclassToLogical.find(cls.id);
                if (log_it != full_state.eclassToLogical.end() && graph.hasNode(log_it->second))
                {
                    runtime_input = graph.getNode(log_it->second).opType == OpType::INPUT &&
                                    graph.getInputDataType(log_it->second) == InputDataType::RUNTIME;
                }
                if (clean_in_any || runtime_input)
                {
                    uint32_t n_users = (log_it != full_state.eclassToLogical.end()) ? user_counts[log_it->second] : 0;
                    candidates.push_back(
                        {cls.base_eclass_id, getSizeBytes(cls.shape, cls.dtype), cls.dtype, cls.mem_space, n_users});
                }
            }
            std::stable_sort(candidates.begin(), candidates.end(),
                             [](const CacheCandidate &a, const CacheCandidate &b) { return a.num_users > b.num_users; });
        }

        // Add CACHE enodes to bucket egraphs for clean candidates
        Engine cpu = Engine{0, EngineType::CPU};
        for (auto &bstate : bucket_states)
        {
            for (const auto &cand : candidates)
            {
                EClassId cid = bstate.egraph.findEClassByBaseId(cand.base_eclass_id);
                if (cid == EClassId{} || bstate.cleanEClasses.count(cid) == 0)
                    continue;
                const EClass cls = bstate.egraph.getEClass(cid);
                bool has_cache = false;
                for (ENodeId en_id : cls.enodes)
                {
                    if (bstate.egraph.getENode(en_id).getOpType() == OpType::CACHE &&
                        bstate.egraph.getENode(en_id).getMemSpace() == cls.mem_space)
                    {
                        has_cache = true;
                        break;
                    }
                }
                if (!has_cache)
                {
                    bstate.egraph.addENode(cid, ENode(KernelId{0}, OpType::CACHE, "", {}, cls.shape, cls.strides,
                                                      cls.dtype, cls.mem_space, {cpu},
                                                      std::to_string(cand.base_eclass_id.value)));
                }
            }
        }

        // Compute enode infos and prune egraphs
        std::vector<std::vector<ENodeInfo>> all_enode_infos(buckets.size());
        for (uint32_t b = 0; b < buckets.size(); ++b)
        {
            all_enode_infos[b] =
                computeENodeInfos(bucket_states[b].egraph, bucket_states[b].eclassToLogical, {}, false);
            pruneEGraph(bucket_states[b].egraph, all_enode_infos[b]);
        }

        // Preallocate constants and inputs
        std::unordered_map<BaseEClassId, ParallelBuffer> preallocated;
        preallocate(graph, full_state.egraph, full_state.nodeToEClass, {}, preallocated);

        std::unordered_map<MemSpace, uint32_t> preallocated_pages;
        for (const auto &pair : preallocated)
        {
            uint64_t extent = static_cast<uint64_t>(pair.second.offset) + pair.second.size;
            uint32_t pages = static_cast<uint32_t>((extent + 4095) / 4096);
            preallocated_pages[pair.second.mem_space] = std::max(preallocated_pages[pair.second.mem_space], pages);
        }

        // 2. SEARCH
        plan::SearchState search_state;
        search_state.buckets = buckets;
        search_state.bucket_weights = bucket_weights;
        search_state.candidates = candidates;
        search_state.mem_caps = settings.mem_caps;
        search_state.preallocated_buffers = preallocated;
        search_state.preallocated_pages = preallocated_pages;

        for (uint32_t b = 0; b < buckets.size(); ++b)
        {
            search_state.bucket_egraphs.push_back(bucket_states[b].egraph);
            search_state.bucket_root_ids.push_back(bucket_states[b].egraph.findConst(bucket_states[b].nodeToEClass.at(rootId)));
            search_state.bucket_clean_eclasses.push_back(bucket_states[b].cleanEClasses);
            search_state.bucket_node_to_eclass.push_back(bucket_states[b].nodeToEClass);
            search_state.bucket_eclass_to_logical.push_back(bucket_states[b].eclassToLogical);
            search_state.bucket_enode_infos.push_back(all_enode_infos[b]);
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

            plan::Domain dom = settings.disable_caching ? plan::Domain::makeFixed(0, true) : plan::Domain::makeMask(0b11);
            plan::VarId vid = search_state.addVar(vinfo, dom);
            search_state.cached_vars[cand.base_eclass_id] = vid;
        }

        // Variables per bucket: selected, start, offset
        search_state.selected_vars.resize(buckets.size());
        search_state.start_vars.resize(buckets.size());
        search_state.offset_vars.resize(buckets.size());

        for (uint32_t b = 0; b < buckets.size(); ++b)
        {
            const auto &egraph = search_state.bucket_egraphs[b];
            EClassId root_cid = search_state.bucket_root_ids[b];
            uint32_t total_classes = static_cast<uint32_t>(egraph.getClasses().size());

            for (const auto &cls : egraph.getClasses())
            {
                EClassId cid = egraph.findConst(cls.id);
                if (cid != cls.id)
                    continue;

                uint32_t n_enodes = static_cast<uint32_t>(cls.enodes.size());
                if (n_enodes > 31)
                {
                    Error::throw_err("EClass " + std::to_string(cid.value) + " has " + std::to_string(n_enodes) +
                                     " enodes, exceeding domain bitmask capacity (max 31).");
                }

                // selected_<bucket_id>_<eclass_id> in {0, 1, ..., n_enodes}
                plan::VarInfo sel_info;
                sel_info.type = plan::VarType::SELECTED;
                sel_info.bucket_idx = b;
                sel_info.eclass_id = cid;
                sel_info.name = "selected_" + std::to_string(b) + "_" + std::to_string(cid.value);

                uint32_t mask = (n_enodes > 0) ? ((1u << (n_enodes + 1)) - 1) : 1u;
                if (cid == root_cid)
                {
                    mask &= ~1u; // Root must be selected
                }

                plan::VarId sel_vid = search_state.addVar(sel_info, plan::Domain::makeMask(mask));
                search_state.selected_vars[b][cid] = sel_vid;

                // start_<bucket_id>_<eclass_id>_<enode_id> in [0, len(eclasses)]
                for (uint32_t en_idx = 0; en_idx < n_enodes; ++en_idx)
                {
                    plan::VarInfo st_info;
                    st_info.type = plan::VarType::START;
                    st_info.bucket_idx = b;
                    st_info.eclass_id = cid;
                    st_info.enode_idx = en_idx;
                    st_info.name = "start_" + std::to_string(b) + "_" + std::to_string(cid.value) + "_" +
                                   std::to_string(en_idx);

                    plan::VarId st_vid = search_state.addVar(st_info, plan::Domain::makeRange(0, total_classes));
                    search_state.start_vars[b][cid].push_back(st_vid);
                }

                // offset_<bucket_id>_<eclass_id> in [preallocated_pages, max_pages]
                if (cls.mem_space.type != HandleType::STORAGE)
                {
                    plan::VarInfo off_info;
                    off_info.type = plan::VarType::OFFSET;
                    off_info.bucket_idx = b;
                    off_info.eclass_id = cid;
                    off_info.mem_space = cls.mem_space;
                    off_info.size_bytes = getSizeBytes(cls.shape, cls.dtype);
                    off_info.size_pages = search_state.bytesToPages(off_info.size_bytes, cls.mem_space);
                    off_info.name = "offset_" + std::to_string(b) + "_" + std::to_string(cid.value);

                    uint32_t align = search_state.getPageAlignment(cls.mem_space);
                    uint64_t cap = search_state.getMemoryCap(cls.mem_space);
                    uint32_t max_p = (cap > off_info.size_bytes) ? static_cast<uint32_t>((cap - off_info.size_bytes) / align) : 0;
                    uint32_t min_p = preallocated_pages[cls.mem_space];

                    plan::VarId off_vid = search_state.addVar(off_info, plan::Domain::makeRange(min_p, std::max(min_p, max_p)));
                    search_state.offset_vars[b][cid] = off_vid;
                }
            }
        }

        // Set up SearchEngine with Propagators and Selector
        LOG(DEBUG) << "[Planner.planAll] Initializing SearchEngine with " << search_state.numVars()
                   << " variables across " << buckets.size() << " bucket(s)...";

        auto selector = std::make_shared<plan::PriorityQueueSelector>();
        auto brancher_impl = brancher ? brancher : std::make_shared<plan::HeuristicBrancher>();

        plan::SearchEngine engine(std::move(search_state), selector, brancher_impl);
        engine.addPropagator(std::make_unique<plan::SelectionPropagator>());
        engine.addPropagator(std::make_unique<plan::CachePropagator>());
        engine.addPropagator(std::make_unique<plan::TopologicalOrderPropagator>());
        engine.addPropagator(std::make_unique<plan::EngineSchedulePropagator>());
        engine.addPropagator(std::make_unique<plan::MemoryNonOverlapPropagator>());
        engine.addPropagator(std::make_unique<plan::CostLowerBoundPropagator>());

        LOG(DEBUG) << "[Planner.planAll] Launching SearchEngine solve (minCompileSeconds=" << minCompileSeconds << "s)...";
        bool solved = engine.solve(minCompileSeconds);
        LOG(DEBUG) << "[Planner.planAll] SearchEngine solve returned: solved=" << solved
                   << ", best_cost=" << engine.incumbent_best_cost;
        if (!solved && engine.incumbent_extractions.empty())
        {
            Error::throw_err("[Planner.planAll] Search failed to find a valid execution graph across buckets.");
        }

        // 3. COMPILE GRAPHS
        std::vector<CompiledGraph> compiled_graphs;
        for (uint32_t b = 0; b < buckets.size(); ++b)
        {
            CompiledGraph cg = buildCompiledGraph(
                rootId, graph, engine.state.bucket_egraphs[b], engine.state.bucket_node_to_eclass[b],
                engine.incumbent_extractions[b], engine.state.bucket_eclass_to_logical[b],
                engine.state.bucket_enode_infos[b]);
            cg.bucket = buckets[b];
            compiled_graphs.push_back(std::move(cg));
        }

        return {std::move(compiled_graphs), std::move(engine.incumbent_cached_nodes)};
    }

    ExtractionResult extractBest(const LogicalId rootId, const Graph &graph, const EGraph &egraph,
                                 const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                                 const std::unordered_set<BaseEClassId> &cachedNodes,
                                 const std::unordered_map<EClassId, LogicalId> &eclassToLogical,
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
                       const std::unordered_set<BaseEClassId> &cachedNodes = {}, bool doSaturate = true,
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
