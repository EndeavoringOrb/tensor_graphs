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
// Each rule is a plain struct. It may define any subset of:
//   init(ctx)                    -- BEFORE DFS: one-time precomputation
//   on_push(node, ctx)/on_pop()  -- DURING DFS: incremental state maintenance
//   check(candidate, idx, ctx)   -- DURING DFS: per-candidate pruning
//   validate_leaf(ctx)           -- AFTER DFS: whole-order validation
// Rules are registered at compile time (DispatchIterator<R1, R2, ...>), so
// every hook is a direct inlinable call -- no vtables, no indirect calls.

// Packed per-selected-node constants shared by several dispatch rules.
// Built once in init() (i.e. before the DFS) so hot loops never touch
// unordered_map lookups.
struct DispatchNodeMeta
{
    static constexpr uint64_t kNoKey = ~uint64_t{0};

    // (type << 32) | idx  of engines[0]; kNoKey if the enode has no engines.
    std::vector<uint64_t> eng_key;
    // (type << 32) | idx  of the enode memory space.
    std::vector<uint64_t> ms_key;
    std::vector<uint8_t> has_eng;
    std::vector<uint8_t> in_selection;
    std::vector<uint8_t> input_like; // OpType::INPUT or OpType::CACHE
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

// =============================================================================
// Rule A: Multi-Engine Commutativity (Partial Order Reduction)
// =============================================================================
// If candidate node `u` and another ready node `v` execute on different engines
// with disjoint memory spaces and are independent, their discrete relative order
// does not affect asynchronous concurrency or makespan. Enforce a canonical
// order (e.g. Engine index/type priority) to break symmetric interleavings.
class MultiEngineCommutativityRule
{
  public:
    const char *name() const
    {
        return "MultiEngineCommutativityRule";
    }

    void init(const DispatchContext &ctx)
    {
        meta.initFrom(ctx);
    }

    bool check(EClassId candidate, size_t candidate_idx, const DispatchContext &ctx) const
    {
        if (candidate_idx == 0 || ctx.current_ready.size() <= 1)
            return false;

        if (!meta.has_eng[candidate.value])
            return false;
        const uint64_t cand_e = meta.eng_key[candidate.value];
        const uint64_t cand_m = meta.ms_key[candidate.value];

        // Check if there is an earlier ready node `v` with a lower canonical engine ID
        // that runs on a disjoint engine and disjoint memory space
        for (size_t i = 0; i < candidate_idx; ++i)
        {
            const EClassId other = ctx.current_ready[i];
            if (!meta.has_eng[other.value])
                continue;

            const uint64_t oth_e = meta.eng_key[other.value];
            if (oth_e != cand_e && meta.ms_key[other.value] != cand_m && oth_e < cand_e)
            {
                return true; // candidate `u` is dominated; `other` must be dispatched first
            }
        }
        return false;
    }

  private:
    DispatchNodeMeta meta;
};

// =============================================================================
// Rule B: Disjoint Subgraph Symmetry Breaking
// =============================================================================
// If independent subgraphs run on disjoint memory spaces, once execution of a
// subgraph's component has started, avoid switching to a disjoint subgraph if
// the current component still has ready work. This collapses O((|A|+|B|)!/(|A|!|B|!))
// interleavings without altering makespan or peak memory.
class DisjointSubgraphSymmetryRule
{
  public:
    const char *name() const
    {
        return "DisjointSubgraphSymmetryRule";
    }

    void init(const DispatchContext &ctx)
    {
        meta.initFrom(ctx);
    }

    bool check(EClassId candidate, size_t candidate_idx, const DispatchContext &ctx) const
    {
        if (ctx.ordered.empty() || ctx.current_ready.size() <= 1)
            return false;

        // Determine the memory space of the most recently dispatched operation
        const EClassId last_dispatched = ctx.ordered.back();
        if (last_dispatched.value >= meta.in_selection.size() || !meta.in_selection[last_dispatched.value])
            return false;
        const uint64_t active_ms = meta.ms_key[last_dispatched.value];

        // If the active memory space still has ready nodes, candidate is dominated if it belongs
        // to a disjoint memory space
        if (meta.ms_key[candidate.value] == active_ms)
            return false;

        bool has_active_ready = false;
        for (EClassId ready_node : ctx.current_ready)
        {
            if (meta.ms_key[ready_node.value] == active_ms)
            {
                has_active_ready = true;
                break;
            }
        }
        return has_active_ready;
    }

  private:
    DispatchNodeMeta meta;
};

// =============================================================================
// Rule C: Last-Reader Buffer-Free Dominance
// =============================================================================
// If ready node `u` is the LAST remaining reader of its inputs (dispatching it
// immediately frees those input buffers), while candidate `v` on the same engine
// with equal cost does NOT free any buffer, candidate `v` is dominated by `u`.
//
// The last-reader relation is maintained INCREMENTALLY: remaining[c] counts the
// selection nodes that read child c and are not dispatched yet. on_push /
// on_pop keep it in sync with the DFS stack, so a "does this node free a
// buffer?" test is O(#children) instead of a full selection-map scan.
class LastReaderBufferFreeDominationRule
{
  public:
    const char *name() const
    {
        return "LastReaderBufferFreeDominationRule";
    }

    void init(const DispatchContext &ctx)
    {
        meta.initFrom(ctx);

        const EGraph &g = ctx.egraph;
        const size_t n = g.getClasses().size();

        remaining.assign(n, 0);
        node_children.clear();
        node_children.resize(n);

        for (const auto &kv : ctx.selection_map)
        {
            EClassId node = g.findConst(kv.first);
            if (node.value >= n)
                continue;
            const ENodeId eid = g.getEClass(node).enodes[kv.second];
            const ENode &en = g.getENode(eid);

            auto &kids = node_children[node.value];
            kids.clear();
            for (EClassId child : en.getChildren())
            {
                EClassId canon_child = g.findConst(child);
                if (canon_child.value >= n || canon_child == node)
                    continue;
                if (std::find(kids.begin(), kids.end(), canon_child) == kids.end())
                {
                    kids.push_back(canon_child);
                    remaining[canon_child.value]++;
                }
            }
        }
    }

    void on_push(EClassId node, const DispatchContext &)
    {
        for (EClassId child : node_children[node.value])
            --remaining[child.value];
    }

    void on_pop(EClassId node, const DispatchContext &)
    {
        for (EClassId child : node_children[node.value])
            ++remaining[child.value];
    }

    bool check(EClassId candidate, size_t candidate_idx, const DispatchContext &ctx) const
    {
        if (ctx.current_ready.size() <= 1)
            return false;

        if (!meta.has_eng[candidate.value])
            return false;
        const uint64_t cand_e = meta.eng_key[candidate.value];
        const float cand_cost = meta.cost[candidate.value];

        if (frees_buffer(candidate))
            return false; // Candidate frees memory; not dominated

        // Check if there is another ready node `other` on the same engine that frees memory with <= cost
        for (size_t i = 0; i < ctx.current_ready.size(); ++i)
        {
            if (i == candidate_idx)
                continue;
            const EClassId other = ctx.current_ready[i];
            if (!meta.has_eng[other.value] || meta.eng_key[other.value] != cand_e)
                continue;

            if (meta.cost[other.value] <= cand_cost + 1e-6f && frees_buffer(other))
            {
                return true; // `candidate` is dominated by `other`
            }
        }

        return false;
    }

  private:
    DispatchNodeMeta meta;

    // remaining[c] = number of undispatched selection nodes reading child c.
    std::vector<uint32_t> remaining;
    // Unique canonical children (inside the e-graph) per selection node.
    std::vector<std::vector<EClassId>> node_children;

    // A node frees a buffer iff it is the LAST undispatched reader of some child,
    // i.e. some child has exactly one undispatched reader left (the node itself).
    bool frees_buffer(EClassId node) const
    {
        for (EClassId child : node_children[node.value])
        {
            if (remaining[child.value] == 1)
            {
                return true;
            }
        }
        return false;
    }
};

// =============================================================================
// Rule D: Single-Engine Generalization
// =============================================================================
// When every dispatchable node executes on one single engine, all topological
// orders are cost-equivalent (execution is sequential), so every candidate
// except the first can be pruned. Whether that invariant holds depends only on
// the selection map, so it is computed ONCE in init() instead of rescanning the
// whole selection at every DFS position.
class SingleEngineDispatchDominationRule
{
  public:
    const char *name() const
    {
        return "SingleEngineDispatchDomination";
    }

    void init(const DispatchContext &ctx)
    {
        meta.initFrom(ctx);

        const size_t n = meta.eng_key.size();
        engine_ok.assign(n, 0);

        // Find the common engine of all non-input-like selection nodes. If they do
        // not agree (or one has no engine), the rule can never fire anywhere.
        bool have_target = false;
        bool any_non_input = false;
        uint64_t target = DispatchNodeMeta::kNoKey;

        for (const auto &kv : ctx.selection_map)
        {
            EClassId node = ctx.egraph.findConst(kv.first);
            if (node.value >= n || !meta.in_selection[node.value])
                continue;
            if (meta.input_like[node.value])
                continue;

            any_non_input = true;
            if (!meta.has_eng[node.value])
                return; // can never fire
            const uint64_t e = meta.eng_key[node.value];
            if (!have_target)
            {
                target = e;
                have_target = true;
            }
            else if (e != target)
            {
                return; // can never fire
            }
        }

        if (!have_target && !any_non_input)
        {
            // Degenerate selection with no dispatchable nodes: nothing to prune.
            return;
        }

        if (have_target)
        {
            mode = Mode::Pinned;
            target_key = target;
            has_input_like = false;
            for (const auto &kv : ctx.selection_map)
            {
                EClassId node = ctx.egraph.findConst(kv.first);
                if (node.value >= n || !meta.in_selection[node.value])
                    continue;
                has_input_like = has_input_like || meta.input_like[node.value] != 0;
                engine_ok[node.value] = (meta.has_eng[node.value] && meta.eng_key[node.value] == target) ? 1 : 0;
            }
        }
        else
        {
            // Selection made only of INPUT/CACHE nodes: the original rule would
            // canonicalize whenever the current ready set happens to share one
            // engine. Replicate that exactly.
            mode = Mode::AllInputLike;
            has_input_like = true;
        }
    }

    bool check(EClassId candidate, size_t candidate_idx, const DispatchContext &ctx) const
    {
        if (mode == Mode::Inactive || candidate_idx == 0 || ctx.current_ready.size() <= 1)
            return false;

        if (mode == Mode::AllInputLike)
        {
            const EClassId first = ctx.current_ready[0];
            if (!meta.has_eng[first.value])
                return false;
            const uint64_t first_e = meta.eng_key[first.value];
            for (EClassId r : ctx.current_ready)
            {
                if (!meta.has_eng[r.value] || meta.eng_key[r.value] != first_e)
                    return false;
            }
            return true;
        }

        // Mode::Pinned -- every non-input-like node is on target_key. Only
        // input-like nodes in the ready set can break the invariant.
        if (!has_input_like)
            return true;

        for (EClassId r : ctx.current_ready)
        {
            if (!engine_ok[r.value])
                return false;
        }
        return true;
    }

  private:
    enum class Mode
    {
        Inactive,
        Pinned,      // all dispatchable nodes pinned to target_key
        AllInputLike // selection contains only INPUT/CACHE nodes
    };

    DispatchNodeMeta meta;
    Mode mode = Mode::Inactive;
    uint64_t target_key = DispatchNodeMeta::kNoKey;
    bool has_input_like = false;
    std::vector<uint8_t> engine_ok; // "would keep the single-engine invariant" per node
};

// =============================================================================
// DispatchIterator
// =============================================================================
// Iterates topological dispatch orders of a fixed selection. Pruning rules are
// registered at compile time as template parameters; the iterator drives them
// through the unified hook protocol (see core/plan/pruning.hpp):
//
//   rules.init(ctx)             once, before the DFS
//   rules.is_pruned(cand, ...)  per candidate, during the DFS
//   rules.on_push(node, ctx)    after a dispatch choice is committed
//   rules.on_pop(node, ctx)     before the choice is undone (balanced, LIFO)
//   rules.validate_leaf(ctx)    when a complete order is reached
//
// With an empty rule set (DispatchIterator<>) every hook vanishes at compile
// time and the search behaves exactly like the unconstrained iterator.
template <typename... Rules> struct DispatchIterator
{
  public:
    prune::PruningRuleSet<Rules...> rules;

    DispatchIterator(const EGraph &_egraph, const std::unordered_map<EClassId, uint32_t> &selection_map,
                     const std::vector<ENodeInfo> &_enode_infos, std::shared_ptr<SearchDelegate> _delegate,
                     Rules &&..._rules)
        : egraph(_egraph), enodeInfos(_enode_infos), delegate(std::move(_delegate)),
          rules(std::forward<Rules>(_rules)...)
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
                // Rejected at the leaf: treat as a dead end and keep searching.
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

// Convenience factory: creates a DispatchIterator without a search delegate.
// Rule types are deduced from the arguments, e.g.
//   auto it = makeDispatchIterator(egraph, sel, infos,
//                                  MultiEngineCommutativityRule{},
//                                  DisjointSubgraphSymmetryRule{});
template <typename... Rules>
DispatchIterator<std::decay_t<Rules>...> makeDispatchIterator(
    const EGraph &egraph, const std::unordered_map<EClassId, uint32_t> &selection_map,
    const std::vector<ENodeInfo> &enodeInfos, Rules &&...rules)
{
    return DispatchIterator<std::decay_t<Rules>...>(egraph, selection_map, enodeInfos, nullptr,
                                                    std::forward<Rules>(rules)...);
}

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
