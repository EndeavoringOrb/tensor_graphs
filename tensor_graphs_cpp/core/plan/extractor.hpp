#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
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
#include "core/misc.hpp"
#include "core/plan/pruning.hpp"
#include "core/plan/search_delegate.hpp"
#include "core/plan/validators/validator.hpp"
#include "core/rewrite.hpp"
#include "core/settings.hpp"
#include "core/shape_propagator.hpp"
#include "core/types.hpp"

struct ENodeInfo
{
    float cost;
    bool is_view;
    float dp_cost = 0.0f;
};

struct DispatchContext
{
    const EGraph &egraph;
    const std::unordered_map<EClassId, uint32_t> &selection_map;
    const std::vector<ENodeInfo> &enodeInfos;
    const std::vector<EClassId> &ordered;
    const std::vector<EClassId> &current_ready;
    uint32_t pos;
};

// =============================================================================
// Dispatch pruning rules (unified pattern, see core/plan/pruning.hpp)
// =============================================================================

struct DispatchNodeMeta
{
    static constexpr uint64_t kNoKey = ~uint64_t{0};

    std::vector<uint64_t> eng_key;
    std::vector<uint64_t> ms_key;
    std::vector<uint8_t> has_eng;
    std::vector<uint8_t> in_selection;
    std::vector<uint8_t> input_like;
    std::vector<float> cost;

    void initFrom(const DispatchContext &ctx)
    {
        const size_t n = ctx.egraph.getClasses().size();
        eng_key.assign(n, kNoKey);
        ms_key.assign(n, kNoKey);
        has_eng.assign(n, 0);
        in_selection.assign(n, 0);
        input_like.assign(n, 0);
        cost.assign(n, 0.0f);

        for (const auto &kv : ctx.selection_map)
        {
            EClassId node = ctx.egraph.findConst(kv.first);
            if (node.value >= n)
                continue;
            in_selection[node.value] = 1;

            const ENodeId eid = ctx.egraph.getEClass(node).enodes[kv.second];
            const ENode &en = ctx.egraph.getENode(eid);
            if (!en.getEngines().empty())
            {
                has_eng[node.value] = 1;
                const Engine &e = en.getEngines()[0];
                eng_key[node.value] = (static_cast<uint64_t>(e.type) << 32) | e.idx;
            }
            const MemSpace &m = en.getMemSpace();
            ms_key[node.value] = (static_cast<uint64_t>(m.type) << 32) | m.idx;
            cost[node.value] = (eid.value < ctx.enodeInfos.size()) ? ctx.enodeInfos[eid.value].cost : 0.0f;
            input_like[node.value] = (en.getOpType() == OpType::INPUT || en.getOpType() == OpType::CACHE) ? 1 : 0;
        }
    }
};

class InputDispatchDominationRule
{
    bool enabled = true;

  public:
    InputDispatchDominationRule(bool en = true) : enabled(en)
    {
    }

    const char *name() const
    {
        return "InputDispatchDominationRule";
    }

    bool check(EClassId cand, size_t /*cand_idx*/, const DispatchContext &ctx) const
    {
        if (!enabled)
            return false;

        // O(1) early-exit: immediately ignore any non-input/non-cache compute node
        uint32_t sel_cand = ctx.selection_map.at(cand);
        ENodeId enode_cand_id = ctx.egraph.getEClass(cand).enodes[sel_cand];
        const ENode &enode_cand = ctx.egraph.getENode(enode_cand_id);
        OpType cand_op = enode_cand.getOpType();

        if (cand_op != OpType::INPUT && cand_op != OpType::CACHE)
            return false;

        // Enforce strictly monotonic ordering (by EClassId) among all ready INPUT and CACHE nodes
        for (EClassId other : ctx.current_ready)
        {
            if (other.value >= cand.value)
                continue;

            uint32_t sel_other = ctx.selection_map.at(other);
            ENodeId enode_other_id = ctx.egraph.getEClass(other).enodes[sel_other];
            const ENode &enode_other = ctx.egraph.getENode(enode_other_id);
            OpType other_op = enode_other.getOpType();

            if (other_op == OpType::INPUT || other_op == OpType::CACHE)
            {
                return true; // Prune: `other` has smaller EClassId and must be dispatched first
            }
        }

        return false;
    }
};

// =============================================================================
// DispatchIterator
// =============================================================================
template <typename... Rules> struct DispatchIterator
{
  public:
    prune::PruningRuleSet<Rules...> rules;

    template <typename... Rs>
    DispatchIterator(const EGraph &_egraph, const std::unordered_map<EClassId, uint32_t> &selection_map,
                     const std::vector<ENodeInfo> &_enode_infos, std::shared_ptr<SearchDelegate> _delegate,
                     Rs &&..._rules)
        : rules(std::forward<Rs>(_rules)...), egraph(_egraph), enodeInfos(_enode_infos), delegate(std::move(_delegate))
    {
        selection_map_ref = &selection_map;
        initOrderState(selection_map);

        DispatchContext ctx{egraph, selection_map, enodeInfos, ordered, current_ready, 0};
        rules.init(ctx);
    }

    bool getNextDispatchOrder(const std::unordered_map<EClassId, uint32_t> &selection_map,
                              std::vector<EClassId> &out_order)
    {
        LOG(DEBUG) << "getNextDispatchOrder";
        if (is_done)
            return false;

        if (!first_yield)
        {
            if (!ascend())
            {
                is_done = true;
                return false;
            }
        }
        first_yield = false;

        uint32_t total_nodes = static_cast<uint32_t>(num_nodes_in_selection);
        if (total_nodes == 0)
        {
            is_done = true;
            return false;
        }

        while (true)
        {
            uint32_t pos = static_cast<uint32_t>(ordered.size());

            if (pos == total_nodes)
            {
                DispatchContext leaf_ctx{egraph, selection_map, enodeInfos, ordered, current_ready, pos};
                if (rules.validate_leaf(leaf_ctx))
                {
                    out_order = ordered;
                    iter++;
                    return true;
                }
                if (!ascend())
                {
                    is_done = true;
                    return false;
                }
                continue;
            }

            if (selection_at_pos[pos] == 0)
            {
                if (delegate)
                {
                    delegate->push_state();

                    std::vector<ActionFeatureExtractDispatch> features;
                    features.reserve(current_ready.size());
                    for (auto id : current_ready)
                    {
                        ActionFeatureExtractDispatch f;
                        uint32_t sel = selection_map.at(id);
                        ENodeId enodeId = egraph.getEClass(id).enodes[sel];
                        const ENode &enode = egraph.getENode(enodeId);

                        f.cost = (enodeId.value < enodeInfos.size()) ? enodeInfos[enodeId.value].cost : 0.0f;
                        f.dp_cost = (enodeId.value < enodeInfos.size()) ? enodeInfos[enodeId.value].dp_cost : 0.0f;
                        f.size = countElements(enode.getShape()) * getDTypeSize(enode.getDType());
                        f.mem_space = enode.getMemSpace();
                        for (const auto &eng : enode.getEngines())
                        {
                            f.engine_idxs.push_back(eng.idx);
                        }

                        Graph g;
                        std::vector<LogicalId> inIds;
                        for (EClassId child : enode.getChildren())
                        {
                            const EClass &cCls = egraph.getEClass(egraph.findConst(child));
                            inIds.push_back(g.input(cCls.shape, cCls.dtype, cCls.strides));
                        }

                        if (enode.getOpType() == OpType::FUSED)
                        {
                            if (KernelRegistry::get().hasKernel(enode.getKernelId()))
                            {
                                auto refFact = KernelRegistry::get().getKernel(enode.getKernelId()).refFactory;
                                if (refFact)
                                    refFact(inIds, g);
                            }
                        }
                        else
                        {
                            g.allocateNode(enode.getOpType(), enode.getOpName(), enode.getDType(), inIds,
                                           enode.getShape(), enode.getStrides(), "");
                        }
                        f.graph = std::move(g);

                        features.push_back(f);
                    }
                    choice_orders[pos] = delegate->order_dispatch(features);
                }
                else
                {
                    choice_orders[pos].resize(current_ready.size());
                    std::iota(choice_orders[pos].begin(), choice_orders[pos].end(), 0u);
                }
            }

            bool chosen = false;
            while (selection_at_pos[pos] < current_ready.size())
            {
                uint32_t choice_idx = selection_at_pos[pos];
                selection_at_pos[pos] = choice_idx + 1;
                uint32_t choice = choice_orders[pos][choice_idx];

                EClassId node = current_ready[choice];

                {
                    DispatchContext ctx{egraph, selection_map, enodeInfos, ordered, current_ready, pos};
                    if (rules.is_pruned(node, choice, ctx))
                    {
                        continue;
                    }
                }

                ordered.push_back(node);
                chosen_at_pos[pos] = node;
                choice_at_pos[pos] = choice;

                current_ready.erase(current_ready.begin() + choice);

                added_nodes_at_pos[pos].clear();
                for (EClassId dep : dependents[node.value])
                {
                    current_in_degree[dep.value]--;
                    if (current_in_degree[dep.value] == 0)
                    {
                        current_ready.push_back(dep);
                        added_nodes_at_pos[pos].push_back(dep);
                    }
                }

                {
                    DispatchContext push_ctx{egraph, selection_map, enodeInfos, ordered, current_ready, pos};
                    rules.on_push(node, push_ctx);
                }
                chosen = true;
                break;
            }

            if (!chosen)
            {
                selection_at_pos[pos] = 0;
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
    }

    uint32_t getIter() const
    {
        return iter;
    }

  private:
    const EGraph &egraph;
    const std::vector<ENodeInfo> &enodeInfos;
    std::shared_ptr<SearchDelegate> delegate;
    const std::unordered_map<EClassId, uint32_t> *selection_map_ref = nullptr;
    size_t num_nodes_in_selection = 0;
    std::vector<EClassId> ordered;
    std::vector<int32_t> current_in_degree;
    std::vector<std::vector<EClassId>> dependents;

    std::vector<EClassId> current_ready;
    std::vector<std::vector<EClassId>> added_nodes_at_pos;
    std::vector<EClassId> chosen_at_pos;
    std::vector<uint32_t> choice_at_pos;
    std::vector<uint32_t> selection_at_pos;
    std::vector<std::vector<uint32_t>> choice_orders;

    bool is_done = false;
    bool first_yield = true;
    uint32_t iter = 0;

    void initOrderState(const std::unordered_map<EClassId, uint32_t> &selection_map)
    {
        num_nodes_in_selection = selection_map.size();
        ordered.clear();
        ordered.reserve(num_nodes_in_selection);
        is_done = false;
        first_yield = true;
        iter = 0;

        uint32_t max_class_id = static_cast<uint32_t>(egraph.getClasses().size());

        current_in_degree.assign(max_class_id, 0);
        dependents.clear();
        dependents.resize(max_class_id);

        current_ready.clear();
        current_ready.reserve(num_nodes_in_selection);

        added_nodes_at_pos.clear();
        added_nodes_at_pos.resize(num_nodes_in_selection + 1);

        chosen_at_pos.assign(num_nodes_in_selection + 1, EClassId{UINT32_MAX});
        choice_at_pos.assign(num_nodes_in_selection + 1, 0);
        selection_at_pos.assign(num_nodes_in_selection + 1, 0);
        choice_orders.clear();
        choice_orders.resize(num_nodes_in_selection + 1);

        std::vector<uint8_t> in_selection(max_class_id, 0);
        for (const auto &kv : selection_map)
        {
            EClassId canon_key = egraph.findConst(kv.first);
            if (canon_key.value < max_class_id)
            {
                in_selection[canon_key.value] = 1;
            }
        }

        std::vector<EClassId> unique_children;
        for (const auto &kv : selection_map)
        {
            EClassId node = egraph.findConst(kv.first);
            if (node.value >= max_class_id)
                continue;

            uint32_t sel = kv.second;
            ENodeId enode_id = egraph.getEClass(node).enodes[sel];
            const ENode &enode = egraph.getENode(enode_id);

            unique_children.clear();
            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = egraph.findConst(child);
                if (canon_child.value < max_class_id && in_selection[canon_child.value] && canon_child != node)
                {
                    if (std::find(unique_children.begin(), unique_children.end(), canon_child) == unique_children.end())
                    {
                        unique_children.push_back(canon_child);
                    }
                }
            }

            current_in_degree[node.value] = static_cast<int32_t>(unique_children.size());
            if (current_in_degree[node.value] == 0)
            {
                current_ready.push_back(node);
            }

            for (EClassId canon_child : unique_children)
            {
                dependents[canon_child.value].push_back(node);
            }
        }

        if (delegate)
        {
            std::vector<float> node_features;
            std::vector<uint32_t> edge_src;
            std::vector<uint32_t> edge_dst;

            std::unordered_map<EClassId, uint32_t> class_to_node_idx;
            uint32_t node_idx = 0;

            for (const auto &kv : selection_map)
            {
                EClassId u = egraph.findConst(kv.first);
                if (u.value < max_class_id)
                {
                    class_to_node_idx[u] = node_idx++;
                }
            }

            for (const auto &kv : selection_map)
            {
                EClassId u = egraph.findConst(kv.first);
                if (u.value >= max_class_id)
                    continue;

                uint32_t sel = kv.second;
                ENodeId enode_id = egraph.getEClass(u).enodes[sel];
                const ENode &enode = egraph.getENode(enode_id);

                node_features.push_back((float)countElements(enode.getShape()) * getDTypeSize(enode.getDType()));
                node_features.push_back((float)enode.getOpType());
                node_features.push_back((enode_id.value < enodeInfos.size()) ? enodeInfos[enode_id.value].cost : 0.0f);
                node_features.push_back((float)enode.getMemSpace().type);
                node_features.push_back((enode_id.value < enodeInfos.size()) ? enodeInfos[enode_id.value].dp_cost
                                                                             : 0.0f);

                uint32_t src_idx = class_to_node_idx[u];
                for (EClassId child : enode.getChildren())
                {
                    EClassId canon_child = egraph.findConst(child);
                    if (class_to_node_idx.count(canon_child))
                    {
                        edge_src.push_back(src_idx);
                        edge_dst.push_back(class_to_node_idx[canon_child]);
                    }
                }
            }
            delegate->init_dispatch_graph(node_features, edge_src, edge_dst);
        }
    }

    bool ascend()
    {
        if (delegate)
        {
            delegate->pop_state();
        }
        uint32_t pos = static_cast<uint32_t>(ordered.size());
        selection_at_pos[pos] = 0;
        if (ordered.empty())
            return false;

        EClassId undone = ordered.back();
        {
            DispatchContext pop_ctx{egraph, *selection_map_ref, enodeInfos, ordered, current_ready, pos - 1};
            rules.on_pop(undone, pop_ctx);
        }
        ordered.pop_back();

        uint32_t parent_pos = pos - 1;

        for (auto it_dep = added_nodes_at_pos[parent_pos].rbegin(); it_dep != added_nodes_at_pos[parent_pos].rend();
             ++it_dep)
        {
            EClassId dep = *it_dep;
            auto it = std::find(current_ready.begin(), current_ready.end(), dep);
            if (it != current_ready.end())
            {
                current_ready.erase(it);
            }
            current_in_degree[dep.value]++;
        }
        added_nodes_at_pos[parent_pos].clear();

        EClassId restored_node = chosen_at_pos[parent_pos];
        uint32_t original_choice = choice_at_pos[parent_pos];
        current_ready.insert(current_ready.begin() + original_choice, restored_node);

        return true;
    }
};

template <typename... Rules>
DispatchIterator<std::decay_t<Rules>...> makeDispatchIterator(
    const EGraph &egraph, const std::unordered_map<EClassId, uint32_t> &selection_map,
    const std::vector<ENodeInfo> &enodeInfos, Rules &&...rules)
{
    return DispatchIterator<std::decay_t<Rules>...>(egraph, selection_map, enodeInfos, nullptr,
                                                    std::forward<Rules>(rules)...);
}

template <typename... Rules>
DispatchIterator<std::decay_t<Rules>...> makeDispatchIteratorWithDelegate(
    const EGraph &egraph, const std::unordered_map<EClassId, uint32_t> &selection_map,
    const std::vector<ENodeInfo> &enodeInfos, std::shared_ptr<SearchDelegate> delegate, Rules &&...rules)
{
    return DispatchIterator<std::decay_t<Rules>...>(egraph, selection_map, enodeInfos, std::move(delegate),
                                                    std::forward<Rules>(rules)...);
}

inline auto makeConfiguredDispatchIterator(const EGraph &egraph,
                                           const std::unordered_map<EClassId, uint32_t> &selection_map,
                                           const std::vector<ENodeInfo> &enodeInfos,
                                           std::shared_ptr<SearchDelegate> delegate, const Settings &settings)
{
    settings.validate_dispatch_rules();
    return makeDispatchIteratorWithDelegate(
        egraph, selection_map, enodeInfos, std::move(delegate),
        InputDispatchDominationRule(settings.is_rule_enabled("dispatch", "InputDispatchDominationRule")));
}

inline auto makeConfiguredDispatchIterator(const EGraph &egraph,
                                           const std::unordered_map<EClassId, uint32_t> &selection_map,
                                           const std::vector<ENodeInfo> &enodeInfos, const Settings &settings)
{
    return makeConfiguredDispatchIterator(egraph, selection_map, enodeInfos, nullptr, settings);
}

// =============================================================================
// ExtractContext -- view into Extractor state at check() time
// =============================================================================
struct ExtractContext
{
    const EGraph &egraph;
    const std::vector<ENodeInfo> &enodeInfos;
    const std::unordered_map<EClassId, uint32_t> &selection_map;
    const std::vector<EClassId> &path;
    EClassId current; // EClass being decided
    uint32_t sel;     // index into current's enodes of the candidate ENode
};

// =============================================================================
// Extractor pruning rules (plain structs; conform to prune::PruningRuleSet)
// =============================================================================

// Rule: Skip ENodes whose cost has already been marked INF by the cost model /
// pre-extraction domination rules. Reduces dead-branch exploration.
class InfiniteCostSkipRule
{
  public:
    bool enabled = true;
    InfiniteCostSkipRule(bool en = true) : enabled(en)
    {
    }
    const char *name() const
    {
        return "InfiniteCostSkipRule";
    }
    bool check(ENodeId /*cand*/, size_t /*cand_idx*/, const ExtractContext &ctx) const
    {
        if (!enabled)
            return false;
        const auto &enodes = ctx.egraph.getEClass(ctx.current).enodes;
        if (ctx.sel >= enodes.size())
            return false;
        ENodeId enode_id = enodes[ctx.sel];
        if (enode_id.value >= ctx.enodeInfos.size())
            return false;
        return ctx.enodeInfos[enode_id.value].cost == TGConstants::INF;
    }
};

// Rule: Symmetry breaking -- if a sibling EClass in the path already selected
// an ENode with the same kernelId (and equivalent children), the candidate is
// dominated. Prunes redundant permutation of equivalent fused rewrites.
class SiblingEquivalentSkipRule
{
  public:
    bool enabled = true;
    SiblingEquivalentSkipRule(bool en = true) : enabled(en)
    {
    }
    const char *name() const
    {
        return "SiblingEquivalentSkipRule";
    }
    bool check(ENodeId /*cand*/, size_t /*cand_idx*/, const ExtractContext &ctx) const
    {
        if (!enabled)
            return false;
        const auto &enodes = ctx.egraph.getEClass(ctx.current).enodes;
        if (ctx.sel >= enodes.size())
            return false;
        ENodeId cand_id = enodes[ctx.sel];
        const ENode &cand_en = ctx.egraph.getENode(cand_id);
        KernelId cand_kid = cand_en.getKernelId();
        if (cand_kid.value == 0)
            return false; // no kernel, skip

        // Walk the path backwards (siblings are path entries before `current`).
        for (auto it = ctx.path.rbegin(); it != ctx.path.rend(); ++it)
        {
            EClassId sibling = *it;
            if (sibling == ctx.current)
                break;
            auto sel_it = ctx.selection_map.find(sibling);
            if (sel_it == ctx.selection_map.end())
                continue;
            ENodeId s_enode_id = ctx.egraph.getEClass(sibling).enodes[sel_it->second];
            const ENode &s_en = ctx.egraph.getENode(s_enode_id);
            if (s_en.getKernelId() != cand_kid)
                continue;
            if (s_en.getChildren().size() != cand_en.getChildren().size())
                continue;
            // children must be canonically equal
            bool same_children = true;
            for (size_t c = 0; c < cand_en.getChildren().size(); ++c)
            {
                if (ctx.egraph.findConst(cand_en.getChildren()[c]) != ctx.egraph.findConst(s_en.getChildren()[c]))
                {
                    same_children = false;
                    break;
                }
            }
            if (same_children)
                return true;
        }
        return false;
    }
};

// =============================================================================
// Extractor<Rules...> -- zero-overhead, rules inlined via std::tuple
// =============================================================================
template <typename... Rules> struct Extractor
{
  private:
    std::vector<std::unique_ptr<ISelectionValidator>> validators;

  public:
    prune::PruningRuleSet<Rules...> rules;

    std::unordered_map<EClassId, uint32_t> selection_map;
    const EGraph &egraph;
    const std::vector<ENodeInfo> &enodeInfos;
    std::shared_ptr<SearchDelegate> delegate;
    std::vector<EClassId> path;
    std::vector<bool> in_path;
    std::vector<int> path_pos;
    std::vector<EClassId> to_process;
    std::vector<bool> has_options;
    uint32_t active_options = 0;
    std::unordered_map<EClassId, uint32_t> next_sel;
    std::unordered_map<EClassId, std::vector<uint32_t>> current_orders;
    EClassId target_backtrack_eclass = EClassId{UINT32_MAX};
    uint64_t numClasses;

    template <typename... Rs>
    Extractor(const EGraph &_egraph, EClassId root_eclass_id, const std::vector<ENodeInfo> &_enodeInfos,
              std::shared_ptr<SearchDelegate> _delegate, Rs &&..._rules)
        : rules(std::forward<Rs>(_rules)...), egraph(_egraph), enodeInfos(_enodeInfos), delegate(std::move(_delegate)),
          numClasses(_egraph.classes.size()), to_process({root_eclass_id}), in_path(_egraph.classes.size(), false),
          path_pos(_egraph.classes.size(), -1), has_options(_egraph.classes.size(), false)
    {
        ExtractContext ctx{egraph, enodeInfos, selection_map, path, EClassId{UINT32_MAX}, 0};
        rules.init(ctx);
    }

    void registerValidator(std::unique_ptr<ISelectionValidator> validator)
    {
        validators.push_back(std::move(validator));
    }

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<EClassId> &order,
                  std::vector<ParallelBuffer> &buffers, std::unordered_map<EClassId, BufferId> &eclass_to_buf,
                  float &cost, std::vector<EClassId> &conflict_nodes)
    {
        for (const auto &validator : validators)
        {
            if (!validator->validate(selection_map, order, path, buffers, eclass_to_buf, cost, conflict_nodes))
            {
                return false;
            }
        }
        return true;
    }

    bool is_done() const
    {
        return to_process.empty() && path.empty();
    }

    bool getNextSelection()
    {
        LOG(DEBUG) << "getNextSelection";
        if (is_done())
            return false;

        while (!to_process.empty())
        {
            EClassId current = to_process.back();
            to_process.pop_back();

            if (selection_map.find(current) != selection_map.end())
            {
                continue;
            }

            path.push_back(current);
            in_path[current.value] = true;
            path_pos[current.value] = path.size() - 1;

            uint32_t sel = 0;
            auto nextIt = next_sel.find(current);
            if (nextIt != next_sel.end())
            {
                sel = nextIt->second;
                next_sel.erase(nextIt);
            }

            const auto &enodes = egraph.getEClass(current).enodes;

            if (sel == 0)
            {
                if (delegate)
                {
                    delegate->push_state();

                    std::vector<ActionFeatureExtractDispatch> features;
                    features.reserve(enodes.size());
                    for (ENodeId enodeId : enodes)
                    {
                        const ENode &enode = egraph.getENode(enodeId);
                        ActionFeatureExtractDispatch f;
                        f.cost = enodeInfos[enodeId.value].cost;
                        f.dp_cost = enodeInfos[enodeId.value].dp_cost;
                        f.size = (float)countElements(enode.getShape()) * getDTypeSize(enode.getDType());
                        f.mem_space = enode.getMemSpace();
                        for (const auto &eng : enode.getEngines())
                        {
                            f.engine_idxs.push_back(eng.idx);
                        }

                        Graph g;
                        std::vector<LogicalId> inIds;
                        for (EClassId child : enode.getChildren())
                        {
                            const EClass &cCls = egraph.getEClass(egraph.findConst(child));
                            inIds.push_back(g.input(cCls.shape, cCls.dtype, cCls.strides));
                        }

                        if (enode.getOpType() == OpType::FUSED)
                        {
                            if (KernelRegistry::get().hasKernel(enode.getKernelId()))
                            {
                                auto refFact = KernelRegistry::get().getKernel(enode.getKernelId()).refFactory;
                                if (refFact)
                                    refFact(inIds, g);
                            }
                        }
                        else
                        {
                            g.allocateNode(enode.getOpType(), enode.getOpName(), enode.getDType(), inIds,
                                           enode.getShape(), enode.getStrides(), "");
                        }
                        f.graph = std::move(g);

                        features.push_back(f);
                    }
                    current_orders[current] = delegate->order_enodes(features);
                }
                else
                {
                    current_orders[current].resize(enodes.size());
                    std::iota(current_orders[current].begin(), current_orders[current].end(), 0u);
                }
            }

            bool found_valid = false;
            for (; sel < enodes.size(); ++sel)
            {
                uint32_t chosen_sel = current_orders[current][sel];
                ENodeId enode_id = enodes[chosen_sel];

                ExtractContext pctx{egraph, enodeInfos, selection_map, path, current, chosen_sel};
                if (rules.is_pruned(enode_id, static_cast<size_t>(chosen_sel), pctx))
                {
                    continue;
                }

                selection_map[current] = chosen_sel;

                bool step_valid = true;
                std::vector<EClassId> dummy_conflict;
                for (const auto &validator : validators)
                {
                    if (!validator->validateStep(current, enode_id, selection_map, dummy_conflict))
                    {
                        step_valid = false;
                        break;
                    }
                }

                if (step_valid)
                {
                    found_valid = true;
                    break;
                }
                else
                {
                    selection_map.erase(current);
                }
            }

            if (!found_valid)
            {
                if (delegate)
                    delegate->pop_state();

                if (delegate && delegate->fast_fail())
                {
                    to_process.clear();
                    return false;
                }

                return false;
            }

            if (enodes.size() > sel + 1)
            {
                if (!has_options[current.value])
                {
                    has_options[current.value] = true;
                    active_options++;
                }
            }

            uint32_t chosen_sel = selection_map[current];
            ENodeId enode_id = enodes[chosen_sel];
            const ENode &node = egraph.getENode(enode_id);
            const auto &children = node.getChildren();
            for (auto it = children.rbegin(); it != children.rend(); ++it)
            {
                EClassId childEClass = egraph.findConst(*it);
                if (selection_map.find(childEClass) == selection_map.end())
                {
                    to_process.push_back(childEClass);
                }
            }
        }
        return true;
    }

    void ascend()
    {
        LOG(DEBUG) << "ascend";

        while (!path.empty())
        {
            EClassId current = path.back();
            path.pop_back();
            in_path[current.value] = false;

            if (selection_map.find(current) == selection_map.end())
                continue;

            uint32_t chosen_sel = selection_map[current];
            const auto &enodes = egraph.getEClass(current).enodes;

            uint32_t iteration_index = chosen_sel;
            auto it = std::find(current_orders[current].begin(), current_orders[current].end(), chosen_sel);
            if (it != current_orders[current].end())
            {
                iteration_index = static_cast<uint32_t>(std::distance(current_orders[current].begin(), it));
            }

            if (iteration_index + 1 < enodes.size())
            {
                next_sel[current] = iteration_index + 1;

                selection_map.erase(current);

                if (enodes.size() <= iteration_index + 2)
                {
                    if (has_options[current.value])
                    {
                        has_options[current.value] = false;
                        active_options--;
                    }
                }
                else
                {
                    if (!has_options[current.value])
                    {
                        has_options[current.value] = true;
                        active_options++;
                    }
                }

                to_process.clear();
                for (EClassId eclass : path)
                {
                    ENodeId n_id = egraph.getEClass(eclass).enodes[selection_map[eclass]];
                    const ENode &n = egraph.getENode(n_id);
                    const auto &children = n.getChildren();
                    for (auto c_it = children.rbegin(); c_it != children.rend(); ++c_it)
                    {
                        EClassId childEClass = egraph.findConst(*c_it);
                        if (selection_map.find(childEClass) == selection_map.end())
                        {
                            to_process.push_back(childEClass);
                        }
                    }
                }
                to_process.push_back(current);
                break;
            }
            else
            {
                selection_map.erase(current);
                current_orders.erase(current);
                if (delegate)
                    delegate->pop_state();
                if (has_options[current.value])
                {
                    has_options[current.value] = false;
                    active_options--;
                }
            }
        }
    }
};

// =============================================================================
// Factory helpers for Extractor
// =============================================================================
template <typename... Rules>
Extractor<std::decay_t<Rules>...> makeExtractor(const EGraph &egraph, EClassId root_eclass_id,
                                                const std::vector<ENodeInfo> &enodeInfos, Rules &&...rules)
{
    return Extractor<std::decay_t<Rules>...>(egraph, root_eclass_id, enodeInfos, nullptr,
                                             std::forward<Rules>(rules)...);
}

template <typename... Rules>
Extractor<std::decay_t<Rules>...> makeExtractorWithDelegate(const EGraph &egraph, EClassId root_eclass_id,
                                                            const std::vector<ENodeInfo> &enodeInfos,
                                                            std::shared_ptr<SearchDelegate> delegate, Rules &&...rules)
{
    return Extractor<std::decay_t<Rules>...>(egraph, root_eclass_id, enodeInfos, std::move(delegate),
                                             std::forward<Rules>(rules)...);
}

inline auto makeConfiguredExtractor(const EGraph &egraph, EClassId root_eclass_id,
                                    const std::vector<ENodeInfo> &enodeInfos, std::shared_ptr<SearchDelegate> delegate,
                                    const Settings &settings)
{
    settings.validate_rules("extract");
    return makeExtractorWithDelegate(
        egraph, root_eclass_id, enodeInfos, std::move(delegate),
        InfiniteCostSkipRule(settings.is_rule_enabled("extract", "InfiniteCostSkipRule")),
        SiblingEquivalentSkipRule(settings.is_rule_enabled("extract", "SiblingEquivalentSkipRule")));
}

inline auto makeConfiguredExtractor(const EGraph &egraph, EClassId root_eclass_id,
                                    const std::vector<ENodeInfo> &enodeInfos, const Settings &settings)
{
    return makeConfiguredExtractor(egraph, root_eclass_id, enodeInfos, nullptr, settings);
}