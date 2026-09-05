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
#include "core/rewrite.hpp"
#include "core/settings.hpp"
#include "core/shape_propagator.hpp"
#include "core/timer.hpp"
#include "core/types.hpp"

struct ENodeInfo
{
    float cost = TGConstants::INF;
    bool is_view = false;
    float dp_cost = TGConstants::INF;
    float dp_cp_cost = TGConstants::INF;
    float rev_cp_cost = 0.0f;
    float dp_mem = TGConstants::INF;
};

// =============================================================================
// Updated DispatchContext with best_cost pointer
// =============================================================================
struct DispatchContext
{
    const EGraph &egraph;
    const std::unordered_map<EClassId, uint32_t> &selection_map;
    const std::vector<ENodeInfo> &enodeInfos;
    const std::vector<EClassId> &ordered;
    const std::vector<EClassId> &current_ready;
    uint32_t pos;
    const std::unordered_map<MemSpace, uint64_t> &mem_caps = empty_mem_caps();
    const float *best_cost = nullptr;

    static const std::unordered_map<MemSpace, uint64_t> &empty_mem_caps()
    {
        static const std::unordered_map<MemSpace, uint64_t> empty;
        return empty;
    }
};

// =============================================================================
// Dispatch Pruning Rule: Makespan / Workload & Delivery Tail Pruning
// =============================================================================
class DispatchCostPruningRule
{
  public:
    TG_PRUNING_RULE(DispatchCostPruningRule)
    DispatchCostPruningRule(bool en = true) : enabled(en)
    {
    }

  private:
    std::unordered_map<Engine, float> engine_finish;
    std::unordered_map<Engine, float> remaining_work_per_engine;
    std::vector<float> node_finish;
    std::vector<float> tail_q;

    struct UndoState
    {
        EClassId node;
        float prev_node_finish;
        std::vector<std::pair<Engine, float>> prev_engine_finish;
        std::vector<std::pair<Engine, float>> prev_rem_work;
    };
    std::vector<UndoState> undo_stack;

  public:
    void init(const DispatchContext &ctx)
    {
        uint32_t max_classes = static_cast<uint32_t>(ctx.egraph.getClasses().size());
        node_finish.assign(max_classes, 0.0f);
        tail_q.assign(max_classes, 0.0f);
        engine_finish.clear();
        remaining_work_per_engine.clear();
        undo_stack.clear();

        for (const auto &kv : ctx.selection_map)
        {
            EClassId canon = ctx.egraph.findConst(kv.first);
            uint32_t sel = kv.second;
            ENodeId enode_id = ctx.egraph.getEClass(canon).enodes[sel];
            const ENode &enode = ctx.egraph.getENode(enode_id);
            float cost = (enode_id.value < ctx.enodeInfos.size()) ? ctx.enodeInfos[enode_id.value].cost : 0.0f;
            if (cost == TGConstants::INF)
                continue;

            for (const auto &eng : enode.getEngines())
            {
                remaining_work_per_engine[eng] += cost;
            }
        }

        // Compute reverse topological delivery tails (q)
        std::vector<std::vector<EClassId>> dependents(max_classes);
        std::vector<uint32_t> out_degree(max_classes, 0);

        for (const auto &kv : ctx.selection_map)
        {
            EClassId parent = ctx.egraph.findConst(kv.first);
            uint32_t sel = kv.second;
            ENodeId enode_id = ctx.egraph.getEClass(parent).enodes[sel];
            const ENode &enode = ctx.egraph.getENode(enode_id);
            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = ctx.egraph.findConst(child);
                if (canon_child != parent && ctx.selection_map.count(canon_child))
                {
                    dependents[canon_child.value].push_back(parent);
                    out_degree[canon_child.value]++;
                }
            }
        }

        std::vector<EClassId> q_worklist;
        for (const auto &kv : ctx.selection_map)
        {
            EClassId node = ctx.egraph.findConst(kv.first);
            if (out_degree[node.value] == 0)
            {
                q_worklist.push_back(node);
                tail_q[node.value] = 0.0f;
            }
        }

        while (!q_worklist.empty())
        {
            EClassId node = q_worklist.back();
            q_worklist.pop_back();

            uint32_t sel = ctx.selection_map.at(node);
            ENodeId enode_id = ctx.egraph.getEClass(node).enodes[sel];
            const ENode &enode = ctx.egraph.getENode(enode_id);
            float cost = (enode_id.value < ctx.enodeInfos.size()) ? ctx.enodeInfos[enode_id.value].cost : 0.0f;
            if (cost == TGConstants::INF)
                cost = 0.0f;

            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = ctx.egraph.findConst(child);
                if (canon_child != node && ctx.selection_map.count(canon_child))
                {
                    tail_q[canon_child.value] = std::max(tail_q[canon_child.value], tail_q[node.value] + cost);
                    out_degree[canon_child.value]--;
                    if (out_degree[canon_child.value] == 0)
                    {
                        q_worklist.push_back(canon_child);
                    }
                }
            }
        }
    }

    bool check(EClassId cand, size_t /*cand_idx*/, const DispatchContext &ctx) const
    {
        if (!enabled || !ctx.best_cost)
            return false;

        float best_c = *ctx.best_cost;
        if (best_c >= TGConstants::INF)
            return false;

        uint32_t sel = ctx.selection_map.at(cand);
        ENodeId enode_id = ctx.egraph.getEClass(cand).enodes[sel];
        const ENode &enode = ctx.egraph.getENode(enode_id);
        float cost = (enode_id.value < ctx.enodeInfos.size()) ? ctx.enodeInfos[enode_id.value].cost : 0.0f;
        if (cost == TGConstants::INF)
            return true;

        float children_finish = 0.0f;
        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = ctx.egraph.findConst(child);
            if (canon_child.value < node_finish.size())
                children_finish = std::max(children_finish, node_finish[canon_child.value]);
        }

        auto engines = enode.getEngines();
        float engine_ready = 0.0f;
        for (const auto &eng : engines)
        {
            auto it = engine_finish.find(eng);
            if (it != engine_finish.end())
                engine_ready = std::max(engine_ready, it->second);
        }

        float start_time = std::max(children_finish, engine_ready);
        float finish_time = start_time + cost;

        // 1. Delivery tail bound
        float lb_tail = finish_time + tail_q[cand.value];
        if (lb_tail >= best_c)
            return true;

        // 2. Scheduled engine remaining workload bound
        for (const auto &eng : engines)
        {
            auto it = remaining_work_per_engine.find(eng);
            float rem_work = (it != remaining_work_per_engine.end()) ? it->second : 0.0f;
            float eng_lb = finish_time + (rem_work - cost);
            if (eng_lb >= best_c)
                return true;
        }

        return false;
    }

    void on_push(EClassId node, const DispatchContext &ctx)
    {
        if (!enabled)
            return;

        uint32_t sel = ctx.selection_map.at(node);
        ENodeId enode_id = ctx.egraph.getEClass(node).enodes[sel];
        const ENode &enode = ctx.egraph.getENode(enode_id);
        float cost = (enode_id.value < ctx.enodeInfos.size()) ? ctx.enodeInfos[enode_id.value].cost : 0.0f;
        if (cost == TGConstants::INF)
            cost = 0.0f;

        float children_finish = 0.0f;
        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = ctx.egraph.findConst(child);
            if (canon_child.value < node_finish.size())
                children_finish = std::max(children_finish, node_finish[canon_child.value]);
        }

        auto engines = enode.getEngines();
        float engine_ready = 0.0f;
        for (const auto &eng : engines)
        {
            auto it = engine_finish.find(eng);
            if (it != engine_finish.end())
                engine_ready = std::max(engine_ready, it->second);
        }

        float start_time = std::max(children_finish, engine_ready);
        float finish_time = start_time + cost;

        UndoState undo;
        undo.node = node;
        undo.prev_node_finish = node_finish[node.value];
        node_finish[node.value] = finish_time;

        for (const auto &eng : engines)
        {
            undo.prev_engine_finish.push_back({eng, engine_finish[eng]});
            engine_finish[eng] = finish_time;

            undo.prev_rem_work.push_back({eng, remaining_work_per_engine[eng]});
            remaining_work_per_engine[eng] -= cost;
        }

        undo_stack.push_back(std::move(undo));
    }

    void on_pop(EClassId /*node*/, const DispatchContext & /*ctx*/)
    {
        if (!enabled || undo_stack.empty())
            return;

        UndoState undo = std::move(undo_stack.back());
        undo_stack.pop_back();

        node_finish[undo.node.value] = undo.prev_node_finish;
        for (const auto &p : undo.prev_engine_finish)
        {
            engine_finish[p.first] = p.second;
        }
        for (const auto &p : undo.prev_rem_work)
        {
            remaining_work_per_engine[p.first] = p.second;
        }
    }
};

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
  public:
    TG_PRUNING_RULE(InputDispatchDominationRule)
    InputDispatchDominationRule(bool en = true) : enabled(en)
    {
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

class UnifiedMemoryExchangeableDispatchRule
{
  public:
    TG_PRUNING_RULE(UnifiedMemoryExchangeableDispatchRule)
    UnifiedMemoryExchangeableDispatchRule(bool en = true) : enabled(en)
    {
    }

  private:
    std::vector<uint32_t> remaining_users; // Tracks R(P) for each EClass

  public:
    // 1. BEFORE DFS: Initialize remaining user counts from selection_map
    void init(const DispatchContext &ctx)
    {
        uint32_t max_class_id = static_cast<uint32_t>(ctx.egraph.getClasses().size());
        remaining_users.assign(max_class_id, 0);

        for (const auto &kv : ctx.selection_map)
        {
            EClassId node = ctx.egraph.findConst(kv.first);
            uint32_t sel = kv.second;
            ENodeId enode_id = ctx.egraph.getEClass(node).enodes[sel];
            const ENode &enode = ctx.egraph.getENode(enode_id);

            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = ctx.egraph.findConst(child);
                if (canon_child.value < max_class_id)
                {
                    remaining_users[canon_child.value]++;
                }
            }
        }
    }

    // 2. DURING DFS: O(1) decrements when a node is committed
    void on_push(EClassId node, const DispatchContext &ctx)
    {
        uint32_t sel = ctx.selection_map.at(node);
        ENodeId enode_id = ctx.egraph.getEClass(node).enodes[sel];
        const ENode &enode = ctx.egraph.getENode(enode_id);

        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = ctx.egraph.findConst(child);
            if (canon_child.value < remaining_users.size())
            {
                remaining_users[canon_child.value]--;
            }
        }
    }

    // 3. DURING DFS: O(1) increments when backtracking
    void on_pop(EClassId node, const DispatchContext &ctx)
    {
        uint32_t sel = ctx.selection_map.at(node);
        ENodeId enode_id = ctx.egraph.getEClass(node).enodes[sel];
        const ENode &enode = ctx.egraph.getENode(enode_id);

        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = ctx.egraph.findConst(child);
            if (canon_child.value < remaining_users.size())
            {
                remaining_users[canon_child.value]++;
            }
        }
    }

    // Computes the multiset of input buffer sizes that die on this step
    std::vector<std::pair<MemSpace, uint64_t>> get_freed_inputs(const ENode &node, EClassId sibling_node,
                                                                const DispatchContext &ctx) const
    {
        std::vector<std::pair<MemSpace, uint64_t>> freed;

        // Find sibling's children to check for shared parents
        uint32_t sib_sel = ctx.selection_map.at(sibling_node);
        const ENode &sib_enode = ctx.egraph.getENode(ctx.egraph.getEClass(sibling_node).enodes[sib_sel]);
        const auto &sib_children = sib_enode.getChildren();

        for (EClassId child : node.getChildren())
        {
            EClassId canon_child = ctx.egraph.findConst(child);

            // If the parent is shared with the sibling, it won't die on Step 1 in either order
            bool is_shared = false;
            for (EClassId sib_child : sib_children)
            {
                if (ctx.egraph.findConst(sib_child) == canon_child)
                {
                    is_shared = true;
                    break;
                }
            }
            if (is_shared)
                continue;

            // If this is the last consumer (R(P) == 1), this buffer dies immediately
            if (canon_child.value < remaining_users.size() && remaining_users[canon_child.value] == 1)
            {
                const EClass &cCls = ctx.egraph.getEClass(canon_child);
                uint64_t size = getSizeBytes(cCls.shape, cCls.dtype);
                freed.push_back({cCls.mem_space, size});
            }
        }
        std::sort(freed.begin(), freed.end());
        return freed;
    }

    // 4. PER-CANDIDATE PRUNING CHECK
    bool check(EClassId cand, size_t /*cand_idx*/, const DispatchContext &ctx) const
    {
        if (!enabled || ctx.current_ready.size() <= 1)
            return false;

        uint32_t sel_cand = ctx.selection_map.at(cand);
        ENodeId enode_cand_id = ctx.egraph.getEClass(cand).enodes[sel_cand];
        const ENode &enode_cand = ctx.egraph.getENode(enode_cand_id);

        uint64_t cand_out_size = getSizeBytes(enode_cand.getShape(), enode_cand.getDType());
        MemSpace cand_ms = enode_cand.getMemSpace();

        for (EClassId other : ctx.current_ready)
        {
            // Canonical tie-breaker: only prune cand if another exchangeable node has smaller ID
            if (other.value >= cand.value)
                continue;

            uint32_t sel_other = ctx.selection_map.at(other);
            ENodeId enode_other_id = ctx.egraph.getEClass(other).enodes[sel_other];
            const ENode &enode_other = ctx.egraph.getENode(enode_other_id);

            // 1. Hardware target equivalence
            if (enode_other.getMemSpace() != cand_ms)
                continue;
            if (enode_other.getEngines() != enode_cand.getEngines())
                continue;

            // 2. Output allocation equivalence
            if (getSizeBytes(enode_other.getShape(), enode_other.getDType()) != cand_out_size)
                continue;

            // 3. Immediate deallocation equivalence
            auto freed_cand = get_freed_inputs(enode_cand, other, ctx);
            auto freed_other = get_freed_inputs(enode_other, cand, ctx);

            if (freed_cand == freed_other)
            {
                return true; // Symmetric transition profile -> Prune candidate
            }
        }

        return false;
    }
};

class MemoryPressureDispatchRule
{
  public:
    TG_PRUNING_RULE(MemoryPressureDispatchRule)
    MemoryPressureDispatchRule(bool en = true) : enabled(en)
    {
    }

  private:
    std::unordered_map<MemSpace, uint64_t> current_live_mem;
    std::vector<uint32_t> remaining_users;

    struct UndoState
    {
        EClassId node;
        MemSpace ms;
        uint64_t allocated_bytes;
        std::vector<EClassId> decremented_children;
        std::vector<std::pair<EClassId, uint64_t>> freed_children;
        std::unordered_map<MemSpace, uint64_t> previous_live_mem;
        std::unordered_set<MemSpace> previously_absent_live_mem;

        bool operator==(const UndoState &other) const
        {
            return node == other.node && ms == other.ms && allocated_bytes == other.allocated_bytes &&
                   decremented_children == other.decremented_children && freed_children == other.freed_children &&
                   previous_live_mem == other.previous_live_mem &&
                   previously_absent_live_mem == other.previously_absent_live_mem;
        }
    };
    std::vector<UndoState> undo_stack;

    // Checks if the ENode is guaranteed to require a fresh buffer allocation
    bool can_execute_inplace(const ENode &enode, const ENodeInfo &info, const DispatchContext &ctx) const
    {
        if (info.is_view)
            return true;

        if (enode.getKernelId().value == 0 || !KernelRegistry::get().hasKernel(enode.getKernelId()))
            return false;

        const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
        if (kernel.safe_inplace_idxs.empty())
            return false;

        MemSpace ms = enode.getMemSpace();
        uint64_t out_size = getSizeBytes(enode.getShape(), enode.getDType());

        for (uint32_t inplace_idx : kernel.safe_inplace_idxs)
        {
            if (inplace_idx < enode.getChildren().size())
            {
                EClassId child = ctx.egraph.findConst(enode.getChildren()[inplace_idx]);

                // An in-place reuse requires this to be the parent's last use (R(P) == 1)
                if (child.value < remaining_users.size() && remaining_users[child.value] == 1)
                {
                    const EClass &cCls = ctx.egraph.getEClass(child);
                    if (cCls.mem_space == ms)
                    {
                        uint64_t in_size = getSizeBytes(cCls.shape, cCls.dtype);
                        if (in_size >= out_size)
                        {
                            return true; // Parent buffer can be directly reused in-place
                        }
                    }
                }
            }
        }
        return false;
    }

  public:
    bool operator==(const MemoryPressureDispatchRule &other) const
    {
        return enabled == other.enabled && current_live_mem == other.current_live_mem &&
               remaining_users == other.remaining_users && undo_stack == other.undo_stack;
    }

    bool operator!=(const MemoryPressureDispatchRule &other) const
    {
        return !(*this == other);
    }

    void init(const DispatchContext &ctx)
    {
        current_live_mem.clear();
        undo_stack.clear();
        uint32_t max_class_id = static_cast<uint32_t>(ctx.egraph.getClasses().size());
        remaining_users.assign(max_class_id, 0);

        for (const auto &kv : ctx.selection_map)
        {
            EClassId node = ctx.egraph.findConst(kv.first);
            uint32_t sel = kv.second;
            ENodeId enode_id = ctx.egraph.getEClass(node).enodes[sel];
            const ENode &enode = ctx.egraph.getENode(enode_id);

            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = ctx.egraph.findConst(child);
                if (canon_child.value < max_class_id)
                {
                    remaining_users[canon_child.value]++;
                }
            }
        }
    }

    bool check(EClassId cand, size_t /*cand_idx*/, const DispatchContext &ctx) const
    {
        if (!enabled)
            return false;

        uint32_t sel = ctx.selection_map.at(cand);
        ENodeId enode_id = ctx.egraph.getEClass(cand).enodes[sel];
        const ENode &enode = ctx.egraph.getENode(enode_id);
        const ENodeInfo &info = ctx.enodeInfos[enode_id.value];
        MemSpace ms = enode.getMemSpace();

        if (ms.type == HandleType::STORAGE || enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
            return false;

        // If it's a view or can run in-place, the lower bound on added memory is 0
        if (can_execute_inplace(enode, info, ctx))
            return false;

        auto cap_it = ctx.mem_caps.find(ms);
        if (cap_it == ctx.mem_caps.end() || cap_it->second == std::numeric_limits<uint64_t>::max())
            return false;

        uint64_t cap = cap_it->second;
        uint64_t out_size = (getSizeBytes(enode.getShape(), enode.getDType()) + 4095) & ~4095ULL;

        auto live_it = current_live_mem.find(ms);
        uint64_t cur_mem = (live_it != current_live_mem.end()) ? live_it->second : 0;

        // Unconditionally prune only if it MUST allocate out-of-place and violates cap
        if (cur_mem + out_size > cap)
        {
            return true;
        }

        return false;
    }

    void on_push(EClassId node, const DispatchContext &ctx)
    {
        if (!enabled)
            return;

        uint32_t sel = ctx.selection_map.at(node);
        ENodeId enode_id = ctx.egraph.getEClass(node).enodes[sel];
        const ENode &enode = ctx.egraph.getENode(enode_id);
        const ENodeInfo &info = ctx.enodeInfos[enode_id.value];
        MemSpace ms = enode.getMemSpace();

        UndoState state;
        state.node = node;
        state.ms = ms;
        state.allocated_bytes = 0;

        auto save_live_mem_state = [&](MemSpace touched_ms) {
            if (state.previous_live_mem.find(touched_ms) != state.previous_live_mem.end() ||
                state.previously_absent_live_mem.find(touched_ms) != state.previously_absent_live_mem.end())
                return;
            auto it = current_live_mem.find(touched_ms);
            if (it == current_live_mem.end())
                state.previously_absent_live_mem.insert(touched_ms);
            else
                state.previous_live_mem.emplace(touched_ms, it->second);
        };

        if (ms.type != HandleType::STORAGE && enode.getOpType() != OpType::INPUT &&
            enode.getOpType() != OpType::CACHE && !can_execute_inplace(enode, info, ctx))
        {
            state.allocated_bytes = (getSizeBytes(enode.getShape(), enode.getDType()) + 4095) & ~4095ULL;
            save_live_mem_state(ms);
            current_live_mem[ms] += state.allocated_bytes;
        }

        // Decrement remaining users and free dead parents
        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = ctx.egraph.findConst(child);
            if (canon_child.value < remaining_users.size())
            {
                remaining_users[canon_child.value]--;
                state.decremented_children.push_back(canon_child);
                if (remaining_users[canon_child.value] == 0)
                {
                    const EClass &cCls = ctx.egraph.getEClass(canon_child);
                    if (cCls.mem_space.type != HandleType::STORAGE)
                    {
                        uint64_t freed = (getSizeBytes(cCls.shape, cCls.dtype) + 4095) & ~4095ULL;
                        if (current_live_mem[cCls.mem_space] >= freed)
                        {
                            save_live_mem_state(cCls.mem_space);
                            current_live_mem[cCls.mem_space] -= freed;
                            state.freed_children.push_back({canon_child, freed});
                        }
                    }
                }
            }
        }
        undo_stack.push_back(std::move(state));
    }

    void on_pop(EClassId /*node*/, const DispatchContext &ctx)
    {
        if (!enabled || undo_stack.empty())
            return;

        UndoState state = std::move(undo_stack.back());
        undo_stack.pop_back();

        for (EClassId child : state.decremented_children)
            remaining_users[child.value]++;

        for (const auto &[ms, value] : state.previous_live_mem)
            current_live_mem[ms] = value;
        for (MemSpace ms : state.previously_absent_live_mem)
            current_live_mem.erase(ms);
    }
};

// =============================================================================
// Dispatch Pruning Rule: Cycle Detection
// =============================================================================
class DispatchCycleRule
{
  public:
    TG_PRUNING_RULE(DispatchCycleRule)
    DispatchCycleRule(bool en = true) : enabled(en)
    {
    }

  private:
    bool has_cycle = false;

    bool detectCycles(const DispatchContext &ctx) const
    {
        uint32_t max_class_id = static_cast<uint32_t>(ctx.egraph.getClasses().size());
        std::vector<int32_t> in_degree(max_class_id, 0);
        std::vector<std::vector<EClassId>> dependents(max_class_id);
        std::vector<EClassId> ready;
        ready.reserve(ctx.selection_map.size());

        std::vector<uint8_t> in_selection(max_class_id, 0);
        for (const auto &kv : ctx.selection_map)
        {
            EClassId canon_key = ctx.egraph.findConst(kv.first);
            if (canon_key.value < max_class_id)
            {
                in_selection[canon_key.value] = 1;
            }
        }

        std::vector<EClassId> unique_children;
        for (const auto &kv : ctx.selection_map)
        {
            EClassId node = ctx.egraph.findConst(kv.first);
            if (node.value >= max_class_id)
                continue;

            uint32_t sel = kv.second;
            ENodeId enode_id = ctx.egraph.getEClass(node).enodes[sel];
            const ENode &enode = ctx.egraph.getENode(enode_id);

            unique_children.clear();
            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = ctx.egraph.findConst(child);
                if (canon_child.value < max_class_id && in_selection[canon_child.value] && canon_child != node)
                {
                    if (std::find(unique_children.begin(), unique_children.end(), canon_child) == unique_children.end())
                    {
                        unique_children.push_back(canon_child);
                    }
                }
            }

            in_degree[node.value] = static_cast<int32_t>(unique_children.size());
            if (in_degree[node.value] == 0)
            {
                ready.push_back(node);
            }

            for (EClassId canon_child : unique_children)
            {
                dependents[canon_child.value].push_back(node);
            }
        }

        size_t processed = 0;
        while (!ready.empty())
        {
            EClassId curr = ready.back();
            ready.pop_back();
            processed++;

            for (EClassId dep : dependents[curr.value])
            {
                in_degree[dep.value]--;
                if (in_degree[dep.value] == 0)
                {
                    ready.push_back(dep);
                }
            }
        }

        return processed < ctx.selection_map.size();
    }

  public:
    void init(const DispatchContext &ctx)
    {
        if (enabled)
        {
            has_cycle = detectCycles(ctx);
        }
        else
        {
            has_cycle = false;
        }
    }

    bool check(EClassId /*cand*/, size_t /*cand_idx*/, const DispatchContext & /*ctx*/) const
    {
        if (!enabled)
            return false;
        return has_cycle;
    }
};

template <typename... Rules> struct DispatchIterator
{
  public:
    prune::PruningRuleSet<Rules...> rules;
    const float *best_cost = nullptr;
    TimeoutChecker *timeout = nullptr;
    const std::unordered_map<MemSpace, uint64_t> &mem_caps;

    template <typename... Rs>
    DispatchIterator(const EGraph &_egraph, const std::unordered_map<EClassId, uint32_t> &selection_map,
                     const std::vector<ENodeInfo> &_enode_infos, std::shared_ptr<SearchDelegate> _delegate,
                     const float *_best_cost, const std::unordered_map<MemSpace, uint64_t> *_mem_caps = nullptr,
                     TimeoutChecker *_timeout = nullptr, Rs &&..._rules)
        : rules(std::forward<Rs>(_rules)...), best_cost(_best_cost), timeout(_timeout),
          mem_caps(_mem_caps ? *_mem_caps : DispatchContext::empty_mem_caps()), egraph(_egraph),
          enodeInfos(_enode_infos), delegate(std::move(_delegate))
    {
        if (delegate && best_cost)
        {
            delegate->set_best_cost_ptr(best_cost);
        }
        selection_map_ref = &selection_map;
        initOrderState(selection_map);

        DispatchContext ctx{egraph, selection_map, enodeInfos, ordered, current_ready, 0, mem_caps, best_cost};
        rules.init(ctx);
    }

    bool can_abort()
    {
        return timeout && timeout->is_expired() && (best_cost != nullptr && *best_cost < TGConstants::INF);
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
            if (can_abort())
            {
                is_done = true;
                return false;
            }

            uint32_t pos = static_cast<uint32_t>(ordered.size());
            if (pos > max_pos_reached)
            {
                max_pos_reached = pos;
            }

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_report_time).count() >= 3.0)
            {
                last_report_time = now;
                std::cout << "\n";
                render_progress(pos, total_nodes);
                std::cout << "\n";
                prune::printPruningProfileSummary();
            }

            if (pos == total_nodes)
            {
                DispatchContext leaf_ctx{egraph,        selection_map, enodeInfos, ordered,
                                         current_ready, pos,           mem_caps,   best_cost};
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

            // Collect only unexplored nodes among current_ready
            std::vector<uint32_t> unexplored_indices;
            unexplored_indices.reserve(current_ready.size());
            for (uint32_t i = 0; i < current_ready.size(); ++i)
            {
                EClassId node = current_ready[i];
                if (std::find(tried_nodes_at_pos[pos].begin(), tried_nodes_at_pos[pos].end(), node) ==
                    tried_nodes_at_pos[pos].end())
                {
                    unexplored_indices.push_back(i);
                }
            }

            if (unexplored_indices.empty())
            {
                tried_nodes_at_pos[pos].clear();
                if (delegate)
                    delegate->pop_state();
                if (!ascend())
                {
                    is_done = true;
                    return false;
                }
                continue;
            }

            std::vector<uint32_t> relative_order;
            if (delegate)
            {
                delegate->push_state();

                std::vector<ActionFeatureExtractDispatch> features;
                features.reserve(unexplored_indices.size());
                for (uint32_t idx : unexplored_indices)
                {
                    EClassId id = current_ready[idx];
                    ActionFeatureExtractDispatch f;
                    uint32_t sel = selection_map.at(id);
                    ENodeId enodeId = egraph.getEClass(id).enodes[sel];
                    const ENode &enode = egraph.getENode(enodeId);

                    f.cost = (enodeId.value < enodeInfos.size()) ? enodeInfos[enodeId.value].cost : 0.0f;
                    f.dp_cost = (enodeId.value < enodeInfos.size()) ? enodeInfos[enodeId.value].dp_cost : 0.0f;
                    f.min_dp_cp_cost =
                        (enodeId.value < enodeInfos.size()) ? enodeInfos[enodeId.value].dp_cp_cost : 0.0f;
                    f.rev_cp_cost = (enodeId.value < enodeInfos.size()) ? enodeInfos[enodeId.value].rev_cp_cost : 0.0f;
                    f.dp_mem = (enodeId.value < enodeInfos.size()) ? enodeInfos[enodeId.value].dp_mem : 0.0f;
                    f.size = countElements(enode.getShape()) * getDTypeSize(enode.getDType());
                    f.mem_space = enode.getMemSpace();
                    auto cap_it = mem_caps.find(enode.getMemSpace());
                    f.mem_cap = (cap_it != mem_caps.end()) ? cap_it->second : 0;
                    for (const auto &eng : enode.getEngines())
                    {
                        f.engine_idxs.push_back(eng.idx);
                    }
                    features.push_back(f);
                }
                relative_order = delegate->order_dispatch(features);
            }
            else
            {
                relative_order.resize(unexplored_indices.size());
                std::iota(relative_order.begin(), relative_order.end(), 0u);
            }

            bool chosen = false;
            for (uint32_t rel_idx : relative_order)
            {
                if (can_abort())
                {
                    is_done = true;
                    return false;
                }

                uint32_t choice = unexplored_indices[rel_idx];
                EClassId node = current_ready[choice];
                tried_nodes_at_pos[pos].push_back(node);

                {
                    DispatchContext ctx{egraph,        selection_map, enodeInfos, ordered,
                                        current_ready, pos,           mem_caps,   best_cost};
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
                    DispatchContext push_ctx{egraph,        selection_map, enodeInfos, ordered,
                                             current_ready, pos,           mem_caps,   best_cost};
                    rules.on_push(node, push_ctx);
                }
                chosen = true;
                break;
            }

            if (!chosen)
            {
                tried_nodes_at_pos[pos].clear();
                if (delegate)
                    delegate->pop_state();
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

    bool ascend_to(uint32_t target_pos)
    {
        while (ordered.size() > target_pos)
        {
            if (!ascend())
                return false;
        }
        return true;
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
    std::vector<std::vector<EClassId>> tried_nodes_at_pos;

    bool is_done = false;
    bool first_yield = true;
    uint32_t iter = 0;
    std::chrono::steady_clock::time_point last_report_time = std::chrono::steady_clock::now();
    uint32_t max_pos_reached = 0;

    void render_progress(uint32_t pos, uint32_t total)
    {
        int bar_width = 30;
        float progress = total > 0 ? static_cast<float>(pos) / total : 0.0f;
        int filled = static_cast<int>(progress * bar_width);
        std::string bar = "[";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < filled)
                bar += "=";
            else if (i == filled)
                bar += ">";
            else
                bar += " ";
        }
        bar += "]";
        std::cout << "\r[Dispatch] " << bar << " " << pos << "/" << total << " (" << std::fixed << std::setprecision(1)
                  << (progress * 100.0f) << "%) max: " << max_pos_reached << "/" << total << std::flush;
    }

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
        tried_nodes_at_pos.clear();
        tried_nodes_at_pos.resize(num_nodes_in_selection + 1);

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
        uint32_t pos = static_cast<uint32_t>(ordered.size());
        if (ordered.empty())
            return false;

        EClassId undone = ordered.back();
        {
            DispatchContext pop_ctx{egraph,  *selection_map_ref, enodeInfos, ordered, current_ready,
                                    pos - 1, mem_caps,           best_cost};
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
    const std::vector<ENodeInfo> &enodeInfos, const std::unordered_map<MemSpace, uint64_t> *mem_caps = nullptr,
    TimeoutChecker *timeout = nullptr, Rules &&...rules)
{
    return DispatchIterator<std::decay_t<Rules>...>(egraph, selection_map, enodeInfos, nullptr, nullptr, mem_caps,
                                                    timeout, std::forward<Rules>(rules)...);
}

template <typename... Rules>
DispatchIterator<std::decay_t<Rules>...> makeDispatchIteratorWithDelegate(
    const EGraph &egraph, const std::unordered_map<EClassId, uint32_t> &selection_map,
    const std::vector<ENodeInfo> &enodeInfos, std::shared_ptr<SearchDelegate> delegate, const float *best_cost,
    const std::unordered_map<MemSpace, uint64_t> *mem_caps = nullptr, TimeoutChecker *timeout = nullptr,
    Rules &&...rules)
{
    return DispatchIterator<std::decay_t<Rules>...>(egraph, selection_map, enodeInfos, std::move(delegate), best_cost,
                                                    mem_caps, timeout, std::forward<Rules>(rules)...);
}

template <typename... Rules>
DispatchIterator<std::decay_t<Rules>...> makeDispatchIteratorWithDelegate(
    const EGraph &egraph, const std::unordered_map<EClassId, uint32_t> &selection_map,
    const std::vector<ENodeInfo> &enodeInfos, std::shared_ptr<SearchDelegate> delegate,
    const std::unordered_map<MemSpace, uint64_t> *mem_caps = nullptr, TimeoutChecker *timeout = nullptr,
    Rules &&...rules)
{
    return DispatchIterator<std::decay_t<Rules>...>(egraph, selection_map, enodeInfos, std::move(delegate), nullptr,
                                                    mem_caps, timeout, std::forward<Rules>(rules)...);
}

using AllDispatchRuleTypes = std::tuple<InputDispatchDominationRule, UnifiedMemoryExchangeableDispatchRule,
                                        MemoryPressureDispatchRule, DispatchCostPruningRule, DispatchCycleRule>;

template <typename BoolTuple>
inline auto makeConfiguredDispatchIteratorFromBools(const EGraph &egraph,
                                                    const std::unordered_map<EClassId, uint32_t> &selection_map,
                                                    const std::vector<ENodeInfo> &enodeInfos,
                                                    std::shared_ptr<SearchDelegate> delegate,
                                                    const BoolTuple &bool_flags, const float *best_cost = nullptr,
                                                    const std::unordered_map<MemSpace, uint64_t> *mem_caps = nullptr,
                                                    TimeoutChecker *timeout = nullptr)
{
    return std::apply(
        [&](auto &&...rs) {
            return makeDispatchIteratorWithDelegate(egraph, selection_map, enodeInfos, std::move(delegate), best_cost,
                                                    mem_caps, timeout, rs...);
        },
        prune::instantiate_from_bools<AllDispatchRuleTypes>(bool_flags));
}

inline auto makeConfiguredDispatchIterator(const EGraph &egraph,
                                           const std::unordered_map<EClassId, uint32_t> &selection_map,
                                           const std::vector<ENodeInfo> &enodeInfos,
                                           std::shared_ptr<SearchDelegate> delegate, const Settings &settings,
                                           const float *best_cost = nullptr,
                                           const std::unordered_map<MemSpace, uint64_t> *mem_caps = nullptr,
                                           TimeoutChecker *timeout = nullptr)
{
    settings.validate_dispatch_rules();
    if (mem_caps == nullptr)
        mem_caps = &settings.mem_caps;
    auto bool_flags = prune::extract_enabled_states<AllDispatchRuleTypes>("dispatch", settings);
    return makeConfiguredDispatchIteratorFromBools(egraph, selection_map, enodeInfos, std::move(delegate), bool_flags,
                                                   best_cost, mem_caps, timeout);
}

inline auto makeConfiguredDispatchIterator(const EGraph &egraph,
                                           const std::unordered_map<EClassId, uint32_t> &selection_map,
                                           const std::vector<ENodeInfo> &enodeInfos, const Settings &settings,
                                           const float *best_cost = nullptr, TimeoutChecker *timeout = nullptr)
{
    return makeConfiguredDispatchIterator(egraph, selection_map, enodeInfos, nullptr, settings, best_cost, nullptr,
                                          timeout);
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
    const std::vector<EClassId> *to_process = nullptr;
    const float *best_cost = nullptr;
    const std::unordered_map<MemSpace, uint64_t> *mem_caps = nullptr;
};

// =============================================================================
// Extractor pruning rules (plain structs; conform to prune::PruningRuleSet)
// =============================================================================

class ExtractorDynamicMinCutRule
{
  public:
    TG_PRUNING_RULE(ExtractorDynamicMinCutRule)
    ExtractorDynamicMinCutRule(bool en = true) : enabled(en)
    {
    }

  private:
    struct UndoFrame
    {
        EClassId current;
        MemSpace current_ms;
        bool was_open;
        std::vector<std::pair<MemSpace, EClassId>> newly_opened_children;
    };

    std::vector<uint64_t> class_sizes;
    std::vector<MemSpace> class_mem_spaces;
    std::unordered_map<MemSpace, uint64_t> open_bytes_per_ms;
    std::unordered_map<MemSpace, std::unordered_map<EClassId, uint32_t>> open_tensors_per_ms;
    std::vector<UndoFrame> undo_stack;

    bool is_open(EClassId node, MemSpace ms) const
    {
        auto it_ms = open_tensors_per_ms.find(ms);
        if (it_ms == open_tensors_per_ms.end())
            return false;
        auto it = it_ms->second.find(node);
        return it != it_ms->second.end() && it->second > 0;
    }

  public:
    void init(const ExtractContext &ctx)
    {
        open_bytes_per_ms.clear();
        open_tensors_per_ms.clear();
        undo_stack.clear();

        uint32_t num_classes = static_cast<uint32_t>(ctx.egraph.getClasses().size());
        class_sizes.assign(num_classes, 0);
        class_mem_spaces.assign(num_classes, MemSpace{1, HandleType::CPP});

        for (uint32_t i = 0; i < num_classes; ++i)
        {
            EClassId canon = ctx.egraph.findConst(EClassId{i});
            if (canon.value < num_classes)
            {
                const EClass &cls = ctx.egraph.getEClass(canon);
                uint64_t sz = (getSizeBytes(cls.shape, cls.dtype) + 4095) & ~4095ULL;
                class_sizes[i] = sz;
                class_mem_spaces[i] = cls.mem_space;
            }
        }
    }

    bool check(ENodeId cand, size_t /*cand_idx*/, const ExtractContext &ctx) const
    {
        if (!enabled || !ctx.mem_caps)
            return false;

        const ENode &enode = ctx.egraph.getENode(cand);
        MemSpace ms = enode.getMemSpace();

        if (ms.type == HandleType::STORAGE)
            return false;

        auto cap_it = ctx.mem_caps->find(ms);
        if (cap_it == ctx.mem_caps->end() || cap_it->second == std::numeric_limits<uint64_t>::max())
            return false;
        uint64_t cap = cap_it->second;

        uint64_t out_size = (getSizeBytes(enode.getShape(), enode.getDType()) + 4095) & ~4095ULL;

        bool is_view = (cand.value < ctx.enodeInfos.size()) ? ctx.enodeInfos[cand.value].is_view : false;
        bool can_be_inplace = is_view;

        if (!can_be_inplace && enode.getKernelId().value != 0 && KernelRegistry::get().hasKernel(enode.getKernelId()))
        {
            const auto &k_entry = KernelRegistry::get().getKernel(enode.getKernelId());
            for (uint32_t inplace_idx : k_entry.safe_inplace_idxs)
            {
                if (inplace_idx < enode.getChildren().size())
                {
                    EClassId child = ctx.egraph.findConst(enode.getChildren()[inplace_idx]);
                    if (class_mem_spaces[child.value] == ms)
                    {
                        uint64_t in_size = class_sizes[child.value];
                        if (out_size <= in_size)
                        {
                            can_be_inplace = true;
                            break;
                        }
                    }
                }
            }
        }

        uint64_t input_sum_in_ms = 0;
        std::unordered_set<EClassId> seen_children;
        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = ctx.egraph.findConst(child);
            if (seen_children.insert(canon_child).second)
            {
                if (class_mem_spaces[canon_child.value] == ms)
                {
                    input_sum_in_ms += class_sizes[canon_child.value];
                }
            }
        }

        // 1. Single Node Execution Peak Memory
        uint64_t local_peak = (can_be_inplace ? 0 : out_size) + input_sum_in_ms;
        if (local_peak > cap)
        {
            LOG(INFO) << "OOM at path size " << ctx.path.size() << " ("
                      << (enode.getDebugOrigin().empty() ? "unknown" : enode.getDebugOrigin()) << ")";
            return true;
        }

        // 2. Cut Memory Lower Bound (Active Bypass Frontier + Current Node Demand)
        uint64_t live_bypass_bytes = 0;
        auto open_it = open_bytes_per_ms.find(ms);
        if (open_it != open_bytes_per_ms.end())
        {
            live_bypass_bytes = open_it->second;
            if (ctx.current.value != UINT32_MAX)
            {
                EClassId canon_curr = ctx.egraph.findConst(ctx.current);
                if (class_mem_spaces[canon_curr.value] == ms && is_open(canon_curr, ms))
                {
                    if (live_bypass_bytes >= class_sizes[canon_curr.value])
                    {
                        live_bypass_bytes -= class_sizes[canon_curr.value];
                    }
                }
            }
        }

        uint64_t total_cut_memory = live_bypass_bytes + local_peak;
        if (total_cut_memory > cap)
        {
            LOG(INFO) << "OOM at path size " << ctx.path.size() << " ("
                      << (enode.getDebugOrigin().empty() ? "unknown" : enode.getDebugOrigin()) << ")";
            return true;
        }

        return false;
    }

    void on_push(ENodeId enode_id, const ExtractContext &ctx)
    {
        if (!enabled)
            return;

        UndoFrame frame;
        frame.current = EClassId{UINT32_MAX};
        frame.was_open = false;

        if (ctx.current.value != UINT32_MAX)
        {
            EClassId canon_curr = ctx.egraph.findConst(ctx.current);
            frame.current = canon_curr;
            frame.current_ms = class_mem_spaces[canon_curr.value];
            frame.was_open = is_open(canon_curr, frame.current_ms);

            // If canon_curr was open, close it (its definition is now expanded)
            if (frame.was_open)
            {
                open_tensors_per_ms[frame.current_ms][canon_curr]--;
                if (open_tensors_per_ms[frame.current_ms][canon_curr] == 0)
                {
                    open_bytes_per_ms[frame.current_ms] -= class_sizes[canon_curr.value];
                }
            }
        }

        const ENode &enode = ctx.egraph.getENode(enode_id);
        std::unordered_set<EClassId> seen_children;

        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = ctx.egraph.findConst(child);
            if (seen_children.insert(canon_child).second)
            {
                MemSpace child_ms = class_mem_spaces[canon_child.value];
                if (child_ms.type != HandleType::STORAGE)
                {
                    if (open_tensors_per_ms[child_ms][canon_child] == 0)
                    {
                        open_bytes_per_ms[child_ms] += class_sizes[canon_child.value];
                        frame.newly_opened_children.push_back({child_ms, canon_child});
                    }
                    open_tensors_per_ms[child_ms][canon_child]++;
                }
            }
        }

        undo_stack.push_back(std::move(frame));
    }

    void on_pop(ENodeId enode_id, const ExtractContext &ctx)
    {
        if (!enabled || undo_stack.empty())
            return;

        UndoFrame frame = std::move(undo_stack.back());
        undo_stack.pop_back();

        // Revert newly opened children
        for (const auto &p : frame.newly_opened_children)
        {
            open_bytes_per_ms[p.first] -= class_sizes[p.second.value];
        }

        const ENode &enode = ctx.egraph.getENode(enode_id);
        std::unordered_set<EClassId> seen_children;
        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = ctx.egraph.findConst(child);
            if (seen_children.insert(canon_child).second)
            {
                MemSpace child_ms = class_mem_spaces[canon_child.value];
                if (child_ms.type != HandleType::STORAGE)
                {
                    if (open_tensors_per_ms[child_ms][canon_child] > 0)
                    {
                        open_tensors_per_ms[child_ms][canon_child]--;
                    }
                }
            }
        }

        // Restore current node if it was open
        if (frame.current.value != UINT32_MAX && frame.was_open)
        {
            if (open_tensors_per_ms[frame.current_ms][frame.current] == 0)
            {
                open_bytes_per_ms[frame.current_ms] += class_sizes[frame.current.value];
            }
            open_tensors_per_ms[frame.current_ms][frame.current]++;
        }
    }
};

// =============================================================================
// Helper: Fast O(T log T) Schrage Preemptive Bound (Jackson's Preemptive Schedule)
// =============================================================================
struct SchrageTask
{
    float r; // release time (earliest start from inputs)
    float c; // processing duration
    float q; // delivery tail (earliest duration to root output)
};

inline float computeSchragePreemptiveBound(const std::vector<SchrageTask> &tasks, std::vector<uint32_t> &sorted_by_r,
                                           std::vector<std::pair<float, float>> &heap) // max-heap of (q, rem_c)
{
    size_t n = tasks.size();
    if (n == 0)
        return 0.0f;
    if (n == 1)
        return tasks[0].r + tasks[0].c + tasks[0].q;

    sorted_by_r.resize(n);
    std::iota(sorted_by_r.begin(), sorted_by_r.end(), 0u);
    std::sort(sorted_by_r.begin(), sorted_by_r.end(), [&](uint32_t a, uint32_t b) {
        if (tasks[a].r != tasks[b].r)
            return tasks[a].r < tasks[b].r;
        return tasks[a].q > tasks[b].q;
    });

    heap.clear();

    float t = tasks[sorted_by_r[0]].r;
    float max_cmax = 0.0f;
    size_t r_idx = 0;

    auto heap_comp = [](const std::pair<float, float> &a, const std::pair<float, float> &b) {
        return a.first < b.first; // max-heap prioritized by delivery tail q
    };

    while (r_idx < n || !heap.empty())
    {
        if (heap.empty() && r_idx < n && t < tasks[sorted_by_r[r_idx]].r)
        {
            t = tasks[sorted_by_r[r_idx]].r;
        }

        while (r_idx < n && tasks[sorted_by_r[r_idx]].r <= t)
        {
            uint32_t idx = sorted_by_r[r_idx++];
            if (tasks[idx].c > 0.0f)
            {
                heap.push_back({tasks[idx].q, tasks[idx].c});
                std::push_heap(heap.begin(), heap.end(), heap_comp);
            }
            else
            {
                max_cmax = std::max(max_cmax, t + tasks[idx].q);
            }
        }

        if (heap.empty())
            continue;

        std::pop_heap(heap.begin(), heap.end(), heap_comp);
        auto cur = heap.back();
        heap.pop_back();

        float cur_q = cur.first;
        float cur_rem_c = cur.second;

        float next_r = (r_idx < n) ? tasks[sorted_by_r[r_idx]].r : std::numeric_limits<float>::infinity();
        float time_to_next = next_r - t;

        if (cur_rem_c <= time_to_next)
        {
            t += cur_rem_c;
            max_cmax = std::max(max_cmax, t + cur_q);
        }
        else
        {
            cur_rem_c -= time_to_next;
            t = next_r;
            heap.push_back({cur_q, cur_rem_c});
            std::push_heap(heap.begin(), heap.end(), heap_comp);
        }
    }

    return max_cmax;
}

// =============================================================================
// Extractor Pruning Rule: ExtractorJacksonCarlierRule (Optimized)
// =============================================================================
class ExtractorJacksonCarlierRule
{
  public:
    TG_PRUNING_RULE(ExtractorJacksonCarlierRule)
    ExtractorJacksonCarlierRule(bool en = true) : enabled(en)
    {
    }

  private:
    struct EngineState
    {
        Engine engine;
        float selected_work = 0.0f;
        float max_r = 0.0f;
        float max_q = 0.0f;
        std::vector<SchrageTask> tasks;
    };

    std::vector<float> node_q;
    std::vector<float> class_min_cp;
    std::vector<EngineState> engines_state;
    std::unordered_map<Engine, uint32_t> engine_map;

    struct QTrailEntry
    {
        EClassId node;
        float old_q;
    };

    struct UndoFrame
    {
        EClassId current;
        float prev_q;
        uint32_t q_trail_start;
        std::vector<uint32_t> eng_indices;
    };

    std::vector<QTrailEntry> q_trail;
    std::vector<UndoFrame> undo_stack;

    mutable std::vector<uint32_t> tmp_sorted_by_r;
    mutable std::vector<std::pair<float, float>> tmp_heap;

  public:
    void init(const ExtractContext &ctx)
    {
        uint32_t num_classes = static_cast<uint32_t>(ctx.egraph.getClasses().size());
        node_q.assign(num_classes, 0.0f);
        class_min_cp.assign(num_classes, TGConstants::INF);
        q_trail.clear();
        undo_stack.clear();
        engines_state.clear();
        engine_map.clear();

        // 1. Precompute class_min_cp for O(1) critical-path queries
        for (uint32_t i = 0; i < num_classes; ++i)
        {
            EClassId canon = ctx.egraph.findConst(EClassId{i});
            if (canon.value != i)
                continue;
            const EClass &cls = ctx.egraph.getEClass(canon);
            float min_cp = TGConstants::INF;
            for (ENodeId eid : cls.enodes)
            {
                if (eid.value < ctx.enodeInfos.size())
                {
                    min_cp = std::min(min_cp, ctx.enodeInfos[eid.value].dp_cp_cost);
                }
            }
            class_min_cp[i] = min_cp;
        }
        for (uint32_t i = 0; i < num_classes; ++i)
        {
            EClassId canon = ctx.egraph.findConst(EClassId{i});
            if (canon.value != i)
            {
                class_min_cp[i] = class_min_cp[canon.value];
            }
        }

        // 2. Discover engines and initialize flat EngineState array
        for (const auto &enode : ctx.egraph.getENodes())
        {
            for (const auto &eng : enode.getEngines())
            {
                if (engine_map.find(eng) == engine_map.end())
                {
                    uint32_t idx = static_cast<uint32_t>(engines_state.size());
                    engine_map[eng] = idx;
                    EngineState es;
                    es.engine = eng;
                    engines_state.push_back(std::move(es));
                }
            }
        }
    }

    bool check(ENodeId cand, size_t /*cand_idx*/, const ExtractContext &ctx) const
    {
        if (!enabled || !ctx.best_cost)
            return false;

        float best_c = *ctx.best_cost;
        if (best_c >= TGConstants::INF)
            return false;

        const ENode &cand_enode = ctx.egraph.getENode(cand);
        float cand_cost = (cand.value < ctx.enodeInfos.size()) ? ctx.enodeInfos[cand.value].cost : 0.0f;
        if (cand_cost == TGConstants::INF)
            return true;

        EClassId current = ctx.current;
        float current_q = (current.value < node_q.size()) ? node_q[current.value] : 0.0f;

        // 1. Candidate Critical Path Bound
        float cand_cp = (cand.value < ctx.enodeInfos.size()) ? ctx.enodeInfos[cand.value].dp_cp_cost : 0.0f;
        if (current_q + cand_cp >= best_c)
        {
            return true;
        }

        // 2. Unselected Frontier Critical Path Bound
        if (ctx.to_process)
        {
            for (EClassId frontier_cls : *ctx.to_process)
            {
                EClassId canon_f = ctx.egraph.findConst(frontier_cls);
                if (canon_f == current || ctx.selection_map.count(canon_f))
                    continue;

                float f_q = (canon_f.value < node_q.size()) ? node_q[canon_f.value] : 0.0f;
                float min_f_cp = (canon_f.value < class_min_cp.size()) ? class_min_cp[canon_f.value] : TGConstants::INF;
                if (min_f_cp != TGConstants::INF && f_q + min_f_cp >= best_c)
                {
                    return true;
                }
            }
        }

        // 3. Engine Workload Bound
        const auto &engines = cand_enode.getEngines();
        for (const auto &eng : engines)
        {
            auto it = engine_map.find(eng);
            if (it != engine_map.end())
            {
                float sel_w = engines_state[it->second].selected_work;
                if (sel_w + cand_cost >= best_c)
                {
                    return true;
                }
            }
        }

        // 4. Jackson / Carlier Window Relaxation on Selected Tasks + Candidate
        for (const auto &eng : engines)
        {
            auto it = engine_map.find(eng);
            if (it == engine_map.end())
                continue;

            auto &es = const_cast<EngineState &>(engines_state[it->second]);
            if (es.tasks.empty())
                continue;

            float cand_r = 0.0f;
            for (EClassId child : cand_enode.getChildren())
            {
                EClassId canon_child = ctx.egraph.findConst(child);
                if (canon_child.value < class_min_cp.size())
                {
                    float min_child_dp = class_min_cp[canon_child.value];
                    if (min_child_dp != TGConstants::INF)
                        cand_r = std::max(cand_r, min_child_dp);
                }
            }

            // O(1) Upper-Bound Filter: skip running full Schrage if theoretical max < best_c
            float total_w = es.selected_work + cand_cost;
            float max_r = std::max(es.max_r, cand_r);
            float max_q = std::max(es.max_q, current_q);
            if (max_r + total_w + max_q < best_c)
            {
                continue;
            }

            // Execute Schrage in-place without heap allocations
            es.tasks.push_back({cand_r, cand_cost, current_q});
            float jps = computeSchragePreemptiveBound(es.tasks, tmp_sorted_by_r, tmp_heap);
            es.tasks.pop_back();

            if (jps >= best_c)
                return true;
        }

        return false;
    }

    void on_push(ENodeId enode_id, const ExtractContext &ctx)
    {
        if (!enabled)
            return;

        const ENode &enode = ctx.egraph.getENode(enode_id);
        float cost = (enode_id.value < ctx.enodeInfos.size()) ? ctx.enodeInfos[enode_id.value].cost : 0.0f;
        if (cost == TGConstants::INF)
            cost = 0.0f;

        EClassId current = ctx.current;
        float current_q = (current.value < node_q.size()) ? node_q[current.value] : 0.0f;

        UndoFrame frame;
        frame.current = current;
        frame.prev_q = current_q;
        frame.q_trail_start = static_cast<uint32_t>(q_trail.size());

        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = ctx.egraph.findConst(child);
            if (canon_child.value < node_q.size())
            {
                float new_q = current_q + cost;
                if (new_q > node_q[canon_child.value])
                {
                    q_trail.push_back({canon_child, node_q[canon_child.value]});
                    node_q[canon_child.value] = new_q;
                }
            }
        }

        float cand_r = 0.0f;
        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = ctx.egraph.findConst(child);
            if (canon_child.value < class_min_cp.size())
            {
                float min_child_dp = class_min_cp[canon_child.value];
                if (min_child_dp != TGConstants::INF)
                    cand_r = std::max(cand_r, min_child_dp);
            }
        }

        for (const auto &eng : enode.getEngines())
        {
            auto it = engine_map.find(eng);
            if (it != engine_map.end())
            {
                uint32_t eidx = it->second;
                auto &es = engines_state[eidx];
                es.selected_work += cost;
                es.max_r = std::max(es.max_r, cand_r);
                es.max_q = std::max(es.max_q, current_q);
                es.tasks.push_back({cand_r, cost, current_q});
                frame.eng_indices.push_back(eidx);
            }
        }

        undo_stack.push_back(std::move(frame));
    }

    void on_pop(ENodeId /*enode_id*/, const ExtractContext & /*ctx*/)
    {
        if (!enabled || undo_stack.empty())
            return;

        UndoFrame frame = std::move(undo_stack.back());
        undo_stack.pop_back();

        while (q_trail.size() > frame.q_trail_start)
        {
            const auto &entry = q_trail.back();
            node_q[entry.node.value] = entry.old_q;
            q_trail.pop_back();
        }

        for (uint32_t eidx : frame.eng_indices)
        {
            auto &es = engines_state[eidx];
            if (!es.tasks.empty())
            {
                float popped_c = es.tasks.back().c;
                es.tasks.pop_back();
                es.selected_work -= popped_c;

                es.max_r = 0.0f;
                es.max_q = 0.0f;
                for (const auto &t : es.tasks)
                {
                    es.max_r = std::max(es.max_r, t.r);
                    es.max_q = std::max(es.max_q, t.q);
                }
            }
        }
    }
};

// Rule: Skip ENodes whose cost has already been marked INF by the cost model /
// pre-extraction domination rules. Reduces dead-branch exploration.
class InfiniteCostSkipRule
{
  public:
    TG_PRUNING_RULE(InfiniteCostSkipRule)
    InfiniteCostSkipRule(bool en = true) : enabled(en)
    {
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

// =============================================================================
// Extractor Pruning Rule: Dynamic Incremental Cycle Detection (Pearce-Kelly)
// =============================================================================
class ExtractorCycleStepRule
{
  public:
    TG_PRUNING_RULE(ExtractorCycleStepRule)
    ExtractorCycleStepRule(bool en = true) : enabled(en)
    {
    }

  private:
    std::vector<uint32_t> ord;                  // Node -> topological position index
    std::vector<EClassId> node_at_pos;          // Topological position index -> Node
    std::vector<std::vector<EClassId>> adj;     // v -> consumers u (v is input to u)
    std::vector<std::vector<EClassId>> rev_adj; // u -> inputs v

    mutable std::vector<uint32_t> visited_marker;
    mutable uint32_t visited_gen = 0;

    struct UndoEntry
    {
        EClassId node;
        uint32_t old_ord;
    };

    struct UndoPosEntry
    {
        uint32_t pos;
        EClassId old_node;
    };

    struct PushUndoFrame
    {
        EClassId u;
        std::vector<EClassId> added_inputs;
        std::vector<UndoEntry> ord_undos;
        std::vector<UndoPosEntry> pos_undos;
    };

    std::vector<PushUndoFrame> undo_stack;

    // Checks reachability from start to target bounded by max_ord
    bool is_reachable_bounded(EClassId start, EClassId target, uint32_t max_ord) const
    {
        if (start == target)
            return true;

        if (++visited_gen == 0)
        {
            std::fill(visited_marker.begin(), visited_marker.end(), 0);
            visited_gen = 1;
        }

        std::vector<EClassId> stack;
        stack.push_back(start);
        visited_marker[start.value] = visited_gen;

        while (!stack.empty())
        {
            EClassId curr = stack.back();
            stack.pop_back();

            for (EClassId next : adj[curr.value])
            {
                if (next == target)
                    return true;

                if (next.value < ord.size() && ord[next.value] <= max_ord)
                {
                    if (visited_marker[next.value] != visited_gen)
                    {
                        visited_marker[next.value] = visited_gen;
                        stack.push_back(next);
                    }
                }
            }
        }
        return false;
    }

    void collect_forward(EClassId start, uint32_t max_ord, std::vector<EClassId> &F)
    {
        if (++visited_gen == 0)
        {
            std::fill(visited_marker.begin(), visited_marker.end(), 0);
            visited_gen = 1;
        }

        std::vector<EClassId> stack;
        stack.push_back(start);
        visited_marker[start.value] = visited_gen;
        F.push_back(start);

        while (!stack.empty())
        {
            EClassId curr = stack.back();
            stack.pop_back();

            for (EClassId next : adj[curr.value])
            {
                if (next.value < ord.size() && ord[next.value] <= max_ord)
                {
                    if (visited_marker[next.value] != visited_gen)
                    {
                        visited_marker[next.value] = visited_gen;
                        F.push_back(next);
                        stack.push_back(next);
                    }
                }
            }
        }
    }

    void collect_backward(EClassId start, uint32_t min_ord, std::vector<EClassId> &B)
    {
        if (++visited_gen == 0)
        {
            std::fill(visited_marker.begin(), visited_marker.end(), 0);
            visited_gen = 1;
        }

        std::vector<EClassId> stack;
        stack.push_back(start);
        visited_marker[start.value] = visited_gen;
        B.push_back(start);

        while (!stack.empty())
        {
            EClassId curr = stack.back();
            stack.pop_back();

            for (EClassId prev : rev_adj[curr.value])
            {
                if (prev.value < ord.size() && ord[prev.value] >= min_ord)
                {
                    if (visited_marker[prev.value] != visited_gen)
                    {
                        visited_marker[prev.value] = visited_gen;
                        B.push_back(prev);
                        stack.push_back(prev);
                    }
                }
            }
        }
    }

    // Pearce-Kelly topological reordering within window [ord[u], ord[v]]
    void pk_reorder(EClassId v, EClassId u, PushUndoFrame &frame)
    {
        uint32_t l = ord[u.value];
        uint32_t r = ord[v.value];
        if (l > r)
            return;

        std::vector<EClassId> F;
        collect_forward(u, r, F);

        std::vector<EClassId> B;
        collect_backward(v, l, B);

        std::sort(F.begin(), F.end(), [&](EClassId a, EClassId b) { return ord[a.value] < ord[b.value]; });
        std::sort(B.begin(), B.end(), [&](EClassId a, EClassId b) { return ord[a.value] < ord[b.value]; });

        std::vector<uint32_t> P;
        P.reserve(B.size() + F.size());
        for (EClassId b_node : B)
            P.push_back(ord[b_node.value]);
        for (EClassId f_node : F)
            P.push_back(ord[f_node.value]);

        std::sort(P.begin(), P.end());

        size_t p_idx = 0;
        for (EClassId b_node : B)
        {
            uint32_t new_pos = P[p_idx++];
            if (ord[b_node.value] != new_pos)
            {
                frame.ord_undos.push_back({b_node, ord[b_node.value]});
                ord[b_node.value] = new_pos;
            }
            if (node_at_pos[new_pos] != b_node)
            {
                frame.pos_undos.push_back({new_pos, node_at_pos[new_pos]});
                node_at_pos[new_pos] = b_node;
            }
        }

        for (EClassId f_node : F)
        {
            uint32_t new_pos = P[p_idx++];
            if (ord[f_node.value] != new_pos)
            {
                frame.ord_undos.push_back({f_node, ord[f_node.value]});
                ord[f_node.value] = new_pos;
            }
            if (node_at_pos[new_pos] != f_node)
            {
                frame.pos_undos.push_back({new_pos, node_at_pos[new_pos]});
                node_at_pos[new_pos] = f_node;
            }
        }
    }

  public:
    void init(const ExtractContext &ctx)
    {
        uint32_t n = static_cast<uint32_t>(ctx.egraph.getClasses().size());
        ord.resize(n);
        node_at_pos.resize(n);
        for (uint32_t i = 0; i < n; ++i)
        {
            ord[i] = i;
            node_at_pos[i] = EClassId{i};
        }
        adj.assign(n, {});
        rev_adj.assign(n, {});
        visited_marker.assign(n, 0);
        visited_gen = 0;
        undo_stack.clear();
    }

    bool check(ENodeId cand, size_t /*cand_idx*/, const ExtractContext &ctx) const
    {
        if (!enabled)
            return false;

        EClassId current = ctx.current;
        if (current.value == UINT32_MAX)
            return false;

        EClassId u = ctx.egraph.findConst(current);
        if (u.value >= ord.size())
            return false;

        const ENode &enode = ctx.egraph.getENode(cand);

        for (EClassId child : enode.getChildren())
        {
            EClassId v = ctx.egraph.findConst(child);
            if (v == u)
                return true; // Direct self-loop

            if (v.value >= ord.size())
                continue;

            // If ord[v] < ord[u], adding edge v -> u is forward in topological order; cycle is impossible.
            if (ord[v.value] < ord[u.value])
                continue;

            // If ord[v] >= ord[u], check reachability strictly bounded to [ord[u], ord[v]].
            if (is_reachable_bounded(u, v, ord[v.value]))
            {
                return true;
            }
        }
        return false;
    }

    void on_push(ENodeId enode_id, const ExtractContext &ctx)
    {
        if (!enabled)
            return;

        EClassId current = ctx.current;
        if (current.value == UINT32_MAX)
            return;

        EClassId u = ctx.egraph.findConst(current);
        if (u.value >= ord.size())
            return;

        const ENode &enode = ctx.egraph.getENode(enode_id);

        PushUndoFrame frame;
        frame.u = u;

        std::vector<EClassId> unique_children;
        for (EClassId child : enode.getChildren())
        {
            EClassId v = ctx.egraph.findConst(child);
            if (v == u || v.value >= ord.size())
                continue;
            if (std::find(unique_children.begin(), unique_children.end(), v) == unique_children.end())
            {
                unique_children.push_back(v);
            }
        }

        for (EClassId v : unique_children)
        {
            adj[v.value].push_back(u);
            rev_adj[u.value].push_back(v);
            frame.added_inputs.push_back(v);

            if (ord[v.value] >= ord[u.value])
            {
                pk_reorder(v, u, frame);
            }
        }

        undo_stack.push_back(std::move(frame));
    }

    void on_pop(ENodeId /*enode_id*/, const ExtractContext & /*ctx*/)
    {
        if (!enabled || undo_stack.empty())
            return;

        PushUndoFrame frame = std::move(undo_stack.back());
        undo_stack.pop_back();

        for (auto it = frame.ord_undos.rbegin(); it != frame.ord_undos.rend(); ++it)
        {
            ord[it->node.value] = it->old_ord;
        }
        for (auto it = frame.pos_undos.rbegin(); it != frame.pos_undos.rend(); ++it)
        {
            node_at_pos[it->pos] = it->old_node;
        }

        EClassId u = frame.u;
        for (EClassId v : frame.added_inputs)
        {
            auto &v_adj = adj[v.value];
            auto it1 = std::find(v_adj.begin(), v_adj.end(), u);
            if (it1 != v_adj.end())
                v_adj.erase(it1);

            auto &u_rev = rev_adj[u.value];
            auto it2 = std::find(u_rev.begin(), u_rev.end(), v);
            if (it2 != u_rev.end())
                u_rev.erase(it2);
        }
    }
};

// =============================================================================
// Extractor<Rules...> -- zero-overhead, rules inlined via std::tuple
// =============================================================================
template <typename... Rules> struct Extractor
{
  public:
    prune::PruningRuleSet<Rules...> rules;
    const float *best_cost = nullptr;
    TimeoutChecker *timeout = nullptr;
    const std::unordered_map<MemSpace, uint64_t> *mem_caps = nullptr;

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
    std::vector<std::vector<uint8_t>> tried_enodes;
    uint64_t numClasses;

    std::chrono::steady_clock::time_point last_report_time = std::chrono::steady_clock::now();
    uint32_t max_path_reached = 0;

    void render_progress(uint32_t path_len, uint32_t total_classes)
    {
        int bar_width = 30;
        float progress = total_classes > 0 ? static_cast<float>(path_len) / total_classes : 0.0f;
        int filled = static_cast<int>(progress * bar_width);
        std::string bar = "[";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < filled)
                bar += "=";
            else if (i == filled)
                bar += ">";
            else
                bar += " ";
        }
        bar += "]";
        std::cout << "\r[Extract] " << bar << " " << path_len << "/" << total_classes << " (" << std::fixed
                  << std::setprecision(1) << (progress * 100.0f) << "%) max: " << max_path_reached << "/"
                  << total_classes << std::flush;
    }

    template <typename... Rs>
    Extractor(const EGraph &_egraph, EClassId root_eclass_id, const std::vector<ENodeInfo> &_enodeInfos,
              std::shared_ptr<SearchDelegate> _delegate, const float *_best_cost,
              const std::unordered_map<MemSpace, uint64_t> *_mem_caps, TimeoutChecker *_timeout = nullptr,
              Rs &&..._rules)
        : rules(std::forward<Rs>(_rules)...), best_cost(_best_cost), timeout(_timeout), mem_caps(_mem_caps),
          egraph(_egraph), enodeInfos(_enodeInfos), delegate(std::move(_delegate)), numClasses(_egraph.classes.size()),
          to_process({root_eclass_id}), in_path(_egraph.classes.size(), false), path_pos(_egraph.classes.size(), -1),
          has_options(_egraph.classes.size(), false), tried_enodes(_egraph.classes.size())
    {
        if (delegate && best_cost)
        {
            delegate->set_best_cost_ptr(best_cost);
        }
        ExtractContext ctx{egraph, enodeInfos,  selection_map, path,    EClassId{UINT32_MAX},
                           0,      &to_process, best_cost,     mem_caps};
        rules.init(ctx);
    }

    bool is_done() const
    {
        return to_process.empty() && path.empty();
    }

    bool can_abort()
    {
        return timeout && timeout->is_expired() && (best_cost != nullptr && *best_cost < TGConstants::INF);
    }

    bool getNextSelection()
    {
        if (is_done())
            return false;

        while (!to_process.empty())
        {
            if (can_abort())
            {
                return false;
            }

            uint32_t cur_path_len = static_cast<uint32_t>(path.size());
            if (cur_path_len > max_path_reached)
            {
                max_path_reached = cur_path_len;
            }

            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_report_time).count() >= 3.0)
            {
                last_report_time = now;
                std::cout << "\n";
                render_progress(cur_path_len, static_cast<uint32_t>(numClasses));
                std::cout << "\n";
                prune::printPruningProfileSummary();
            }

            if (to_process.size() > 1 && delegate)
            {
                std::vector<ActionFeatureFrontier> features;
                features.reserve(to_process.size());

                for (EClassId cid : to_process)
                {
                    const auto &cls = egraph.getEClass(cid);
                    ActionFeatureFrontier f;
                    f.eclass_id = cid.value;
                    f.num_enodes = static_cast<uint32_t>(cls.enodes.size());
                    f.size = getSizeBytes(cls.shape, cls.dtype);
                    f.dtype = cls.dtype;
                    f.mem_space = cls.mem_space;
                    if (mem_caps)
                    {
                        auto cap_it = mem_caps->find(cls.mem_space);
                        f.mem_cap = (cap_it != mem_caps->end()) ? cap_it->second : 0;
                    }

                    float min_cp = TGConstants::INF;
                    float min_dp = TGConstants::INF;
                    float min_mem = TGConstants::INF;
                    for (ENodeId eid : cls.enodes)
                    {
                        if (eid.value < enodeInfos.size())
                        {
                            min_cp = std::min(min_cp, enodeInfos[eid.value].dp_cp_cost);
                            min_dp = std::min(min_dp, enodeInfos[eid.value].dp_cost);
                            min_mem = std::min(min_mem, enodeInfos[eid.value].dp_mem);
                        }
                    }
                    f.min_dp_cp_cost = (min_cp == TGConstants::INF) ? 0.0f : min_cp;
                    f.min_dp_cost = (min_dp == TGConstants::INF) ? 0.0f : min_dp;
                    f.min_dp_mem = (min_mem == TGConstants::INF) ? 0.0f : min_mem;
                    features.push_back(f);
                }

                std::vector<uint32_t> order = delegate->order_frontier(features);
                if (!order.empty() && order[0] < to_process.size())
                {
                    std::swap(to_process[order[0]], to_process.back());
                }
            }

            EClassId current = to_process.back();
            to_process.pop_back();

            if (selection_map.find(current) != selection_map.end())
            {
                continue;
            }

            const auto &enodes = egraph.getEClass(current).enodes;
            if (tried_enodes[current.value].size() != enodes.size())
            {
                tried_enodes[current.value].assign(enodes.size(), 0);
            }

            std::vector<uint32_t> unexplored;
            unexplored.reserve(enodes.size());
            for (uint32_t i = 0; i < enodes.size(); ++i)
            {
                if (!tried_enodes[current.value][i])
                {
                    unexplored.push_back(i);
                }
            }

            if (unexplored.empty())
            {
                tried_enodes[current.value].clear();
                if (delegate)
                    delegate->pop_state();
                if (delegate && delegate->fast_fail())
                {
                    to_process.clear();
                    return false;
                }
                return false;
            }

            path.push_back(current);
            in_path[current.value] = true;
            path_pos[current.value] = path.size() - 1;

            std::vector<uint32_t> relative_order;
            if (delegate)
            {
                delegate->push_state();

                std::vector<ActionFeatureExtractDispatch> features;
                features.reserve(unexplored.size());
                for (uint32_t unexp_idx : unexplored)
                {
                    ENodeId enodeId = enodes[unexp_idx];
                    const ENode &enode = egraph.getENode(enodeId);
                    ActionFeatureExtractDispatch f;
                    f.cost = enodeInfos[enodeId.value].cost;
                    f.dp_cost = enodeInfos[enodeId.value].dp_cost;
                    f.min_dp_cp_cost = enodeInfos[enodeId.value].dp_cp_cost;
                    f.rev_cp_cost = enodeInfos[enodeId.value].rev_cp_cost;
                    f.dp_mem = enodeInfos[enodeId.value].dp_mem;
                    f.size = (float)countElements(enode.getShape()) * getDTypeSize(enode.getDType());
                    f.mem_space = enode.getMemSpace();
                    if (mem_caps)
                    {
                        auto cap_it = mem_caps->find(enode.getMemSpace());
                        f.mem_cap = (cap_it != mem_caps->end()) ? cap_it->second : 0;
                    }
                    for (const auto &eng : enode.getEngines())
                    {
                        f.engine_idxs.push_back(eng.idx);
                    }
                    features.push_back(f);
                }
                relative_order = delegate->order_enodes(features);
            }
            else
            {
                relative_order.resize(unexplored.size());
                std::iota(relative_order.begin(), relative_order.end(), 0u);
            }

            bool found_valid = false;
            for (uint32_t rel_idx : relative_order)
            {
                if (can_abort())
                {
                    return false;
                }

                uint32_t chosen_sel = unexplored[rel_idx];
                tried_enodes[current.value][chosen_sel] = 1;
                ENodeId enode_id = enodes[chosen_sel];

                ExtractContext pctx{egraph,     enodeInfos,  selection_map, path,    current,
                                    chosen_sel, &to_process, best_cost,     mem_caps};
                if (rules.is_pruned(enode_id, static_cast<size_t>(chosen_sel), pctx))
                {
                    continue;
                }

                selection_map[current] = chosen_sel;
                rules.on_push(enode_id, pctx);
                found_valid = true;
                break;
            }

            if (!found_valid)
            {
                path.pop_back();
                in_path[current.value] = false;
                path_pos[current.value] = -1;

                if (delegate)
                    delegate->pop_state();

                if (delegate && delegate->fast_fail())
                {
                    to_process.clear();
                    return false;
                }

                return false;
            }

            uint32_t remaining_unexplored = 0;
            for (uint32_t i = 0; i < enodes.size(); ++i)
            {
                if (!tried_enodes[current.value][i])
                    remaining_unexplored++;
            }
            if (remaining_unexplored > 0)
            {
                if (!has_options[current.value])
                {
                    has_options[current.value] = true;
                    active_options++;
                }
            }
            else
            {
                if (has_options[current.value])
                {
                    has_options[current.value] = false;
                    active_options--;
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
        while (!path.empty())
        {
            EClassId current = path.back();
            path.pop_back();
            in_path[current.value] = false;
            path_pos[current.value] = -1;

            if (selection_map.find(current) == selection_map.end())
                continue;

            uint32_t chosen_sel = selection_map[current];
            const auto &enodes = egraph.getEClass(current).enodes;
            ENodeId popped_enode = enodes[chosen_sel];

            ExtractContext pop_ctx{egraph,     enodeInfos,  selection_map, path,    current,
                                   chosen_sel, &to_process, best_cost,     mem_caps};
            rules.on_pop(popped_enode, pop_ctx);
            selection_map.erase(current);

            bool has_unexplored = false;
            for (uint32_t i = 0; i < enodes.size(); ++i)
            {
                if (!tried_enodes[current.value][i])
                {
                    has_unexplored = true;
                    break;
                }
            }

            if (has_unexplored)
            {
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
                tried_enodes[current.value].clear();
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
                                                const std::vector<ENodeInfo> &enodeInfos,
                                                const std::unordered_map<MemSpace, uint64_t> *mem_caps = nullptr,
                                                TimeoutChecker *timeout = nullptr, Rules &&...rules)
{
    return Extractor<std::decay_t<Rules>...>(egraph, root_eclass_id, enodeInfos, nullptr, nullptr, mem_caps, timeout,
                                             std::forward<Rules>(rules)...);
}

template <typename... Rules>
Extractor<std::decay_t<Rules>...> makeExtractorWithDelegate(
    const EGraph &egraph, EClassId root_eclass_id, const std::vector<ENodeInfo> &enodeInfos,
    std::shared_ptr<SearchDelegate> delegate, const float *best_cost,
    const std::unordered_map<MemSpace, uint64_t> *mem_caps = nullptr, TimeoutChecker *timeout = nullptr,
    Rules &&...rules)
{
    return Extractor<std::decay_t<Rules>...>(egraph, root_eclass_id, enodeInfos, std::move(delegate), best_cost,
                                             mem_caps, timeout, std::forward<Rules>(rules)...);
}

template <typename... Rules>
Extractor<std::decay_t<Rules>...> makeExtractorWithDelegate(
    const EGraph &egraph, EClassId root_eclass_id, const std::vector<ENodeInfo> &enodeInfos,
    std::shared_ptr<SearchDelegate> delegate, const std::unordered_map<MemSpace, uint64_t> *mem_caps = nullptr,
    TimeoutChecker *timeout = nullptr, Rules &&...rules)
{
    return Extractor<std::decay_t<Rules>...>(egraph, root_eclass_id, enodeInfos, std::move(delegate), nullptr, mem_caps,
                                             timeout, std::forward<Rules>(rules)...);
}

using AllExtractRuleTypes =
    std::tuple<InfiniteCostSkipRule, ExtractorCycleStepRule, ExtractorJacksonCarlierRule, ExtractorDynamicMinCutRule>;

template <typename BoolTuple>
inline auto makeConfiguredExtractorFromBools(const EGraph &egraph, EClassId root_eclass_id,
                                             const std::vector<ENodeInfo> &enodeInfos,
                                             std::shared_ptr<SearchDelegate> delegate, const BoolTuple &bool_flags,
                                             const float *best_cost = nullptr,
                                             const std::unordered_map<MemSpace, uint64_t> *mem_caps = nullptr,
                                             TimeoutChecker *timeout = nullptr)
{
    return std::apply(
        [&](auto &&...rs) {
            return makeExtractorWithDelegate(egraph, root_eclass_id, enodeInfos, std::move(delegate), best_cost,
                                             mem_caps, timeout, rs...);
        },
        prune::instantiate_from_bools<AllExtractRuleTypes>(bool_flags));
}

inline auto makeConfiguredExtractor(const EGraph &egraph, EClassId root_eclass_id,
                                    const std::vector<ENodeInfo> &enodeInfos, std::shared_ptr<SearchDelegate> delegate,
                                    const Settings &settings, const float *best_cost = nullptr,
                                    const std::unordered_map<MemSpace, uint64_t> *mem_caps = nullptr,
                                    TimeoutChecker *timeout = nullptr)
{
    settings.validate_rules("extract");
    if (mem_caps == nullptr)
        mem_caps = &settings.mem_caps;
    auto bool_flags = prune::extract_enabled_states<AllExtractRuleTypes>("extract", settings);
    return makeConfiguredExtractorFromBools(egraph, root_eclass_id, enodeInfos, std::move(delegate), bool_flags,
                                            best_cost, mem_caps, timeout);
}

inline auto makeConfiguredExtractor(const EGraph &egraph, EClassId root_eclass_id,
                                    const std::vector<ENodeInfo> &enodeInfos, const Settings &settings,
                                    const float *best_cost = nullptr, TimeoutChecker *timeout = nullptr)
{
    return makeConfiguredExtractor(egraph, root_eclass_id, enodeInfos, nullptr, settings, best_cost, nullptr, timeout);
}
