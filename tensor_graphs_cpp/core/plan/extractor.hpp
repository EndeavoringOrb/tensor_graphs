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
#include "core/plan/search_delegate.hpp"
#include "core/plan/validators/validator.hpp"
#include "core/rewrite.hpp"
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

class IDispatchDominationRule
{
  public:
    virtual ~IDispatchDominationRule() = default;
    virtual std::string name() const = 0;
    virtual bool is_dominated(EClassId candidate, size_t candidate_idx, const DispatchContext &ctx) = 0;
};

// =============================================================================
// Rule A: Multi-Engine Commutativity (Partial Order Reduction)
// =============================================================================
// If candidate node `u` and another ready node `v` execute on different engines
// with disjoint memory spaces and are independent, their discrete relative order
// does not affect asynchronous concurrency or makespan. Enforce a canonical
// order (e.g. Engine index/type priority) to break symmetric interleavings.
class MultiEngineCommutativityRule : public IDispatchDominationRule
{
  public:
    std::string name() const override
    {
        return "MultiEngineCommutativityRule";
    }

    bool is_dominated(EClassId candidate, size_t candidate_idx, const DispatchContext &ctx) override
    {
        if (candidate_idx == 0 || ctx.current_ready.size() <= 1)
            return false;

        const auto &cand_enode =
            ctx.egraph.getENode(ctx.egraph.getEClass(candidate).enodes[ctx.selection_map.at(candidate)]);
        if (cand_enode.getEngines().empty())
            return false;
        const Engine &cand_engine = cand_enode.getEngines()[0];
        MemSpace cand_ms = cand_enode.getMemSpace();

        // Check if there is an earlier ready node `v` with a lower canonical engine ID
        // that runs on a disjoint engine and disjoint memory space
        for (size_t i = 0; i < candidate_idx; ++i)
        {
            EClassId other = ctx.current_ready[i];
            const auto &other_enode =
                ctx.egraph.getENode(ctx.egraph.getEClass(other).enodes[ctx.selection_map.at(other)]);
            if (other_enode.getEngines().empty())
                continue;

            const Engine &other_engine = other_enode.getEngines()[0];
            MemSpace other_ms = other_enode.getMemSpace();

            if (other_engine != cand_engine && other_ms != cand_ms)
            {
                // Canonical ordering: prefer lower engine type or index first
                if (std::make_pair(static_cast<uint32_t>(other_engine.type), other_engine.idx) <
                    std::make_pair(static_cast<uint32_t>(cand_engine.type), cand_engine.idx))
                {
                    return true; // candidate `u` is dominated; `other` must be dispatched first
                }
            }
        }
        return false;
    }
};

// =============================================================================
// Rule B: Disjoint Subgraph Symmetry Breaking
// =============================================================================
// If independent subgraphs run on disjoint memory spaces, once execution of a
// subgraph's component has started, avoid switching to a disjoint subgraph if
// the current component still has ready work. This collapses O((|A|+|B|)!/(|A|!|B|!))
// interleavings without altering makespan or peak memory.
class DisjointSubgraphSymmetryRule : public IDispatchDominationRule
{
  public:
    std::string name() const override
    {
        return "DisjointSubgraphSymmetryRule";
    }

    bool is_dominated(EClassId candidate, size_t candidate_idx, const DispatchContext &ctx) override
    {
        if (ctx.ordered.empty() || ctx.current_ready.size() <= 1)
            return false;

        // Determine the memory space of the most recently dispatched operation
        EClassId last_dispatched = ctx.ordered.back();
        auto last_sel_it = ctx.selection_map.find(last_dispatched);
        if (last_sel_it == ctx.selection_map.end())
            return false;

        const auto &last_enode = ctx.egraph.getENode(ctx.egraph.getEClass(last_dispatched).enodes[last_sel_it->second]);
        MemSpace active_ms = last_enode.getMemSpace();

        // Check if there is still a ready node belonging to the same active memory space
        bool has_active_ready = false;
        for (EClassId ready_node : ctx.current_ready)
        {
            const auto &enode =
                ctx.egraph.getENode(ctx.egraph.getEClass(ready_node).enodes[ctx.selection_map.at(ready_node)]);
            if (enode.getMemSpace() == active_ms)
            {
                has_active_ready = true;
                break;
            }
        }

        if (!has_active_ready)
            return false;

        // If the active memory space still has ready nodes, candidate is dominated if it belongs to a disjoint memory
        // space
        const auto &cand_enode =
            ctx.egraph.getENode(ctx.egraph.getEClass(candidate).enodes[ctx.selection_map.at(candidate)]);
        if (cand_enode.getMemSpace() != active_ms)
        {
            return true;
        }

        return false;
    }
};

// =============================================================================
// Rule C: Last-Reader Buffer-Free Dominance
// =============================================================================
// If ready node `u` is the LAST remaining reader of its inputs (dispatching it
// immediately frees those input buffers), while candidate `v` on the same engine
// with equal cost does NOT free any buffer, candidate `v` is dominated by `u`.
class LastReaderBufferFreeDominationRule : public IDispatchDominationRule
{
  public:
    std::string name() const override
    {
        return "LastReaderBufferFreeDominationRule";
    }

    bool is_dominated(EClassId candidate, size_t candidate_idx, const DispatchContext &ctx) override
    {
        if (ctx.current_ready.size() <= 1)
            return false;

        const auto &cand_enode =
            ctx.egraph.getENode(ctx.egraph.getEClass(candidate).enodes[ctx.selection_map.at(candidate)]);
        if (cand_enode.getEngines().empty())
            return false;
        const Engine &cand_engine = cand_enode.getEngines()[0];
        float cand_cost =
            ctx.enodeInfos[ctx.egraph.getEClass(candidate).enodes[ctx.selection_map.at(candidate)].value].cost;

        // Set of undispatched nodes in the selection map
        std::unordered_set<EClassId> dispatched(ctx.ordered.begin(), ctx.ordered.end());

        auto frees_buffers = [&](EClassId node) -> bool {
            const auto &enode = ctx.egraph.getENode(ctx.egraph.getEClass(node).enodes[ctx.selection_map.at(node)]);
            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = ctx.egraph.findConst(child);
                bool has_other_undispatched_reader = false;

                for (const auto &kv : ctx.selection_map)
                {
                    if (kv.first == node || dispatched.count(kv.first))
                        continue;
                    const auto &other_node = ctx.egraph.getENode(ctx.egraph.getEClass(kv.first).enodes[kv.second]);
                    for (EClassId other_child : other_node.getChildren())
                    {
                        if (ctx.egraph.findConst(other_child) == canon_child)
                        {
                            has_other_undispatched_reader = true;
                            break;
                        }
                    }
                    if (has_other_undispatched_reader)
                        break;
                }

                if (!has_other_undispatched_reader)
                {
                    return true; // `node` is the last reader of `canon_child`
                }
            }
            return false;
        };

        bool cand_frees = frees_buffers(candidate);
        if (cand_frees)
            return false; // Candidate frees memory; not dominated

        // Check if there is another ready node `other` on the same engine that frees memory with <= cost
        for (size_t i = 0; i < ctx.current_ready.size(); ++i)
        {
            if (i == candidate_idx)
                continue;
            EClassId other = ctx.current_ready[i];
            const auto &other_enode =
                ctx.egraph.getENode(ctx.egraph.getEClass(other).enodes[ctx.selection_map.at(other)]);
            if (other_enode.getEngines().empty() || other_enode.getEngines()[0] != cand_engine)
                continue;

            float other_cost =
                ctx.enodeInfos[ctx.egraph.getEClass(other).enodes[ctx.selection_map.at(other)].value].cost;
            if (other_cost <= cand_cost + 1e-6f && frees_buffers(other))
            {
                return true; // `candidate` is dominated by `other`
            }
        }

        return false;
    }
};

// =============================================================================
// Single-Engine Generalization Rule
// =============================================================================
class SingleEngineDispatchDominationRule : public IDispatchDominationRule
{
  public:
    std::string name() const override
    {
        return "SingleEngineDispatchDomination";
    }

    bool is_dominated(EClassId candidate, size_t candidate_idx, const DispatchContext &ctx) override
    {
        if (candidate_idx == 0 || ctx.current_ready.size() <= 1)
            return false;

        const auto &enode0 = ctx.egraph.getENode(
            ctx.egraph.getEClass(ctx.current_ready[0]).enodes[ctx.selection_map.at(ctx.current_ready[0])]);
        if (enode0.getEngines().empty())
            return false;
        const Engine &target_engine = enode0.getEngines()[0];

        for (size_t i = 1; i < ctx.current_ready.size(); ++i)
        {
            const auto &enode = ctx.egraph.getENode(
                ctx.egraph.getEClass(ctx.current_ready[i]).enodes[ctx.selection_map.at(ctx.current_ready[i])]);
            if (enode.getEngines().empty() || !(enode.getEngines()[0] == target_engine))
            {
                return false;
            }
        }

        for (const auto &kv : ctx.selection_map)
        {
            const auto &enode = ctx.egraph.getENode(ctx.egraph.getEClass(kv.first).enodes[kv.second]);
            if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
                continue;
            if (enode.getEngines().empty() || !(enode.getEngines()[0] == target_engine))
            {
                return false;
            }
        }

        return true;
    }
};

struct DispatchIterator
{
  public:
    std::vector<std::shared_ptr<IDispatchDominationRule>> domination_rules;

    DispatchIterator(const EGraph &_egraph, const std::unordered_map<EClassId, uint32_t> &selection_map,
                     const std::vector<ENodeInfo> &_enode_infos, std::shared_ptr<SearchDelegate> _delegate = nullptr)
        : egraph(_egraph), enodeInfos(_enode_infos), delegate(_delegate)
    {
        initOrderState(selection_map);
    }

    void addDominationRule(std::shared_ptr<IDispatchDominationRule> rule)
    {
        domination_rules.push_back(std::move(rule));
    }

    void clearDominationRules()
    {
        domination_rules.clear();
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
                out_order = ordered;
                iter++;
                return true;
            }

            if (selection_at_pos[pos] == 0)
            {
                if (delegate)
                {
                    delegate->push_state();
                }
            }

            bool chosen = false;
            while (selection_at_pos[pos] < current_ready.size())
            {
                uint32_t choice_idx = selection_at_pos[pos];
                selection_at_pos[pos] = choice_idx + 1;
                uint32_t choice = choice_idx;

                if (delegate)
                {
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
                        f.graph = g;

                        features.push_back(f);
                    }
                    std::vector<uint32_t> custom_order = delegate->order_dispatch(features);
                    if (choice_idx < custom_order.size())
                    {
                        choice = custom_order[choice_idx];
                    }
                }

                EClassId node = current_ready[choice];

                if (!domination_rules.empty())
                {
                    DispatchContext ctx{egraph, selection_map, enodeInfos, ordered, current_ready, pos};
                    bool dominated = false;
                    for (const auto &rule : domination_rules)
                    {
                        if (rule->is_dominated(node, choice, ctx))
                        {
                            dominated = true;
                            break;
                        }
                    }
                    if (dominated)
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
    size_t num_nodes_in_selection = 0;
    std::vector<EClassId> ordered;
    std::vector<int32_t> current_in_degree;
    std::vector<std::vector<EClassId>> dependents;

    std::vector<EClassId> current_ready;
    std::vector<std::vector<EClassId>> added_nodes_at_pos;
    std::vector<EClassId> chosen_at_pos;
    std::vector<uint32_t> choice_at_pos;
    std::vector<uint32_t> selection_at_pos;

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

struct Extractor
{
  private:
    std::vector<std::unique_ptr<ISelectionValidator>> validators;

  public:
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
    EClassId target_backtrack_eclass = EClassId{UINT32_MAX};
    uint64_t numClasses;

    Extractor(const EGraph &_egraph, EClassId root_eclass_id, const std::vector<ENodeInfo> &_enodeInfos,
              std::shared_ptr<SearchDelegate> _delegate = nullptr)
        : egraph(_egraph), enodeInfos(_enodeInfos), delegate(_delegate), numClasses(_egraph.classes.size()),
          to_process({root_eclass_id}), in_path(_egraph.classes.size(), false), path_pos(_egraph.classes.size(), -1),
          has_options(_egraph.classes.size(), false)
    {
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

    bool getNextSelection()
    {
        LOG(DEBUG) << "getNextSelection";
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

            if (sel == 0 && delegate)
            {
                delegate->push_state();
            }

            const auto &enodes = egraph.getEClass(current).enodes;

            bool found_valid = false;
            int max_conflict_path_pos = -1;
            std::vector<EClassId> aggregate_conflicts;

            for (; sel < enodes.size(); ++sel)
            {
                uint32_t chosen_sel = sel;
                if (delegate)
                {
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
                        f.graph = g;

                        features.push_back(f);
                    }
                    std::vector<uint32_t> custom_order = delegate->order_enodes(features);
                    if (sel < custom_order.size())
                    {
                        chosen_sel = custom_order[sel];
                    }
                }

                ENodeId enode_id = enodes[chosen_sel];
                selection_map[current] = chosen_sel;

                bool step_valid = true;
                std::vector<EClassId> conflict_nodes;

                for (const auto &validator : validators)
                {
                    if (!validator->validateStep(current, enode_id, selection_map, conflict_nodes))
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
                    aggregate_conflicts.insert(aggregate_conflicts.end(), conflict_nodes.begin(), conflict_nodes.end());
                    for (EClassId c_node : conflict_nodes)
                    {
                        if (c_node != current && path_pos[c_node.value] != -1)
                        {
                            if (path_pos[c_node.value] > max_conflict_path_pos)
                            {
                                max_conflict_path_pos = path_pos[c_node.value];
                            }
                        }
                    }
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

                int best_conflict_pos = -1;
                for (EClassId c_node : aggregate_conflicts)
                {
                    int pos = path_pos[c_node.value];
                    if (pos != -1 && c_node != current)
                    {
                        auto sel_it = selection_map.find(c_node);
                        if (sel_it != selection_map.end())
                        {
                            uint32_t sel_idx = sel_it->second;
                            if (sel_idx + 1 < egraph.getEClass(c_node).enodes.size())
                            {
                                if (pos > best_conflict_pos)
                                {
                                    best_conflict_pos = pos;
                                }
                            }
                        }
                    }
                }

                if (best_conflict_pos != -1)
                {
                    target_backtrack_eclass = path[best_conflict_pos];
                }
                else
                {
                    if (max_conflict_path_pos != -1)
                        target_backtrack_eclass = path[max_conflict_path_pos];
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
        bool skip_increment = (target_backtrack_eclass != EClassId{UINT32_MAX});

        while (!path.empty())
        {
            EClassId current = path.back();
            path.pop_back();
            in_path[current.value] = false;

            if (selection_map.find(current) == selection_map.end())
                continue;

            if (skip_increment && current == target_backtrack_eclass)
            {
                LOG(DEBUG) << "skipped back to path size " << std::to_string(path.size()) << std::endl;
                skip_increment = false;
            }

            uint32_t chosen_sel = selection_map[current];
            const auto &enodes = egraph.getEClass(current).enodes;

            uint32_t iteration_index = chosen_sel;
            if (delegate)
            {
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
                        g.allocateNode(enode.getOpType(), enode.getOpName(), enode.getDType(), inIds, enode.getShape(),
                                       enode.getStrides(), "");
                    }
                    f.graph = g;

                    features.push_back(f);
                }
                std::vector<uint32_t> custom_order = delegate->order_enodes(features);
                auto it = std::find(custom_order.begin(), custom_order.end(), chosen_sel);
                if (it != custom_order.end())
                {
                    iteration_index = static_cast<uint32_t>(std::distance(custom_order.begin(), it));
                }
            }

            ENodeId enode_id = enodes[chosen_sel];
            const ENode &node = egraph.getENode(enode_id);

            if (!skip_increment && iteration_index + 1 < enodes.size())
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