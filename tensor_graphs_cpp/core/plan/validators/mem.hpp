// File: tensor_graphs_cpp/core/plan/validators/mem.hpp
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
#include "core/plan/extractor.hpp"
#include "core/plan/pruning.hpp"
#include "core/plan/search_delegate.hpp"
#include "core/plan/validators/validator.hpp"
#include "core/rewrite.hpp"
#include "core/settings.hpp"
#include "core/shape_propagator.hpp"
#include "core/timer.hpp"
#include "core/types.hpp"

inline float get_cost(const std::vector<EClassId> &ordered, const EGraph &egraph,
                      const std::unordered_map<EClassId, uint32_t> &selection_map,
                      const std::vector<ENodeInfo> &enodeInfos, bool print_utilization = false)
{
    std::unordered_map<EClassId, float> birth_times;
    std::unordered_map<Engine, float> engine_finish;
    std::unordered_map<Engine, float> engine_active_time;

    for (EClassId eclass : ordered)
    {
        auto sel_it = selection_map.find(eclass);
        if (sel_it == selection_map.end())
            continue;
        uint32_t sel = sel_it->second;
        ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
        const ENode &node = egraph.getENode(enode_id);

        if (node.getOpType() == OpType::INPUT || node.getOpType() == OpType::CACHE)
        {
            birth_times[eclass] = 0.0f;
            continue;
        }

        float cost = enodeInfos[enode_id.value].cost;

        float children_finish = 0.0f;
        for (EClassId child : node.getChildren())
        {
            child = egraph.findConst(child);
            auto c_sel_it = selection_map.find(child);
            if (c_sel_it == selection_map.end())
                continue;
            uint32_t child_sel = c_sel_it->second;
            ENodeId child_enode_id = egraph.getEClass(child).enodes[child_sel];
            const ENode &child_node = egraph.getENode(child_enode_id);

            for (const auto &engine : child_node.getEngines())
            {
                auto it = engine_finish.find(engine);
                float child_finish = (it != engine_finish.end()) ? it->second : 0.0f;
                children_finish = std::max(children_finish, child_finish);
            }
        }

        float engine_free = 0.0f;
        for (const auto &engine : node.getEngines())
        {
            auto it = engine_finish.find(engine);
            if (it != engine_finish.end())
            {
                engine_free = std::max(engine_free, it->second);
            }
        }

        float birth = std::max(children_finish, engine_free);
        birth_times[eclass] = birth;

        for (const auto &engine : node.getEngines())
        {
            engine_finish[engine] = birth + cost;
            engine_active_time[engine] += cost;
        }
    }

    float total_cost = 0.0f;
    for (const auto &kv : engine_finish)
    {
        total_cost = std::max(total_cost, kv.second);
    }

    if (print_utilization && total_cost > 0.0f)
    {
        std::cout << "Total Makespan (Cost): " << total_cost << " ms\n";
        for (const auto &kv : engine_active_time)
        {
            Engine eng = kv.first;
            float active_duration = kv.second;
            float percentage = (active_duration / total_cost) * 100.0f;

            std::cout << "  - Engine " << eng.idx << " (" << toString(eng.type) << "): " << std::fixed
                      << std::setprecision(2) << percentage << "% "
                      << "(" << active_duration << " ms active)\n";
        }
    }

    return total_cost;
}

inline EClassId resolve_view_alias(EClassId id, const EGraph &egraph,
                                   const std::unordered_map<EClassId, uint32_t> &selection_map,
                                   const std::vector<ENodeInfo> &enodeInfos)
{
    EClassId curr = egraph.findConst(id);
    while (true)
    {
        auto sel_it = selection_map.find(curr);
        if (sel_it == selection_map.end())
            break;
        uint32_t sel = sel_it->second;
        ENodeId enode_id = egraph.getEClass(curr).enodes[sel];
        if (enodeInfos[enode_id.value].is_view)
        {
            const ENode &node = egraph.getENode(enode_id);
            if (!node.getChildren().empty())
            {
                curr = egraph.findConst(node.getChildren()[0]);
            }
            else
            {
                break;
            }
        }
        else
        {
            break;
        }
    }
    return curr;
}

// =============================================================================
// BufferizeContext -- view into BufferizeIterator's state at check() time
// =============================================================================
struct BufferizeContext
{
    const std::vector<EClassId> &ordered;
    const EGraph &egraph;
    const std::unordered_map<EClassId, uint32_t> &selection_map;
    const std::vector<ENodeInfo> &enodeInfos;
    const std::unordered_map<EClassId, uint32_t> &birth_times;
    const std::unordered_map<EClassId, uint32_t> &death_times;
    const std::unordered_map<EClassId, EClassId> &inplace_alias;
    const std::vector<int> &current_choices;
    uint32_t k;
    const std::unordered_map<MemSpace, uint64_t> &mem_caps;

    EClassId get_inplace_alias(EClassId id) const
    {
        auto it = inplace_alias.find(id);
        while (it != inplace_alias.end())
        {
            id = it->second;
            it = inplace_alias.find(id);
        }
        return id;
    }
};

// =============================================================================
// Bufferize pruning rules
// =============================================================================

class PeakMemoryPruningRule
{
  public:
    TG_PRUNING_RULE(PeakMemoryPruningRule)
    PeakMemoryPruningRule(bool en = true) : enabled(en)
    {
    }

  private:
    std::unordered_map<MemSpace, std::vector<uint64_t>> mem_usage;
    std::unordered_map<EClassId, uint32_t> max_death_time;

    struct UndoState
    {
        int choice;
        EClassId target_base;
        uint32_t old_death;
        uint32_t new_death;
    };
    std::vector<UndoState> undo_stack;

  public:
    void init(const BufferizeContext &ctx)
    {
        mem_usage.clear();
        max_death_time.clear();
        undo_stack.clear();

        uint32_t T = ctx.ordered.size();
        for (const auto &kv : ctx.mem_caps)
        {
            if (kv.first.type == HandleType::STORAGE)
                continue;
            mem_usage[kv.first].assign(T + 2, 0);
        }
    }

    bool check(int candidate_choice, size_t, const BufferizeContext &ctx) const
    {
        if (!enabled)
            return false;

        EClassId eclass = ctx.ordered[ctx.k];
        uint32_t sel = ctx.selection_map.at(eclass);
        ENodeId enode_id = ctx.egraph.getEClass(eclass).enodes[sel];
        const ENode &node = ctx.egraph.getENode(enode_id);
        MemSpace ms = node.getMemSpace();

        // 1. STORAGE is file-backed and does not consume RAM/VRAM capacity
        if (ms.type == HandleType::STORAGE)
            return false;

        // 2. INPUT and CACHE nodes are preallocated in reserved memory
        if (node.getOpType() == OpType::INPUT || node.getOpType() == OpType::CACHE)
            return false;

        auto cap_it = ctx.mem_caps.find(ms);
        if (cap_it == ctx.mem_caps.end())
            return false;
        uint64_t cap = cap_it->second;
        if (cap == std::numeric_limits<uint64_t>::max())
            return false;

        auto usage_it = mem_usage.find(ms);
        if (usage_it == mem_usage.end())
            return false;
        const auto &usage = usage_it->second;

        uint64_t size = getSizeBytes(node.getShape(), node.getDType());
        size = (size + 4095) & ~4095ULL;

        uint32_t d_time = ctx.death_times.count(eclass) ? ctx.death_times.at(eclass) : 1;

        if (candidate_choice == -1)
        {
            uint32_t b_time = ctx.birth_times.count(eclass) ? ctx.birth_times.at(eclass) : 0;
            for (uint32_t t = b_time; t <= d_time && t < usage.size(); ++t)
            {
                if (usage[t] + size > cap)
                {
                    LOG(DEBUG) << "OOM k=" << ctx.k << "/" << ctx.ordered.size();
                    return true;
                }
            }
        }
        else
        {
            EClassId child = ctx.egraph.findConst(node.getChildren()[candidate_choice]);
            EClassId child_base = resolve_view_alias(child, ctx.egraph, ctx.selection_map, ctx.enodeInfos);
            EClassId target_base = ctx.get_inplace_alias(child_base);

            auto death_it = max_death_time.find(target_base);
            uint32_t old_death = (death_it != max_death_time.end()) ? death_it->second : 0xFFFFFFFF;

            if (old_death != 0xFFFFFFFF && d_time > old_death)
            {
                for (uint32_t t = old_death; t <= d_time && t < usage.size(); ++t)
                {
                    if (usage[t] + size > cap)
                    {
                        LOG(DEBUG) << "OOM k=" << ctx.k << "/" << ctx.ordered.size();
                        return true;
                    }
                }
            }
        }

        return false;
    }

    void on_push(int choice, const BufferizeContext &ctx)
    {
        if (!enabled)
            return;

        EClassId eclass = ctx.ordered[ctx.k];
        uint32_t sel = ctx.selection_map.at(eclass);
        ENodeId enode_id = ctx.egraph.getEClass(eclass).enodes[sel];
        const ENode &node = ctx.egraph.getENode(enode_id);
        MemSpace ms = node.getMemSpace();

        if (ms.type == HandleType::STORAGE || node.getOpType() == OpType::INPUT || node.getOpType() == OpType::CACHE)
        {
            undo_stack.push_back({choice, EClassId{UINT32_MAX}, 0, 0});
            return;
        }

        uint64_t size = getSizeBytes(node.getShape(), node.getDType());
        size = (size + 4095) & ~4095ULL;

        uint32_t d_time = ctx.death_times.count(eclass) ? ctx.death_times.at(eclass) : 1;

        UndoState state{choice, EClassId{UINT32_MAX}, 0, 0};

        if (choice == -1)
        {
            uint32_t b_time = ctx.birth_times.count(eclass) ? ctx.birth_times.at(eclass) : 0;
            state.target_base = eclass;
            state.old_death = b_time;
            state.new_death = d_time;
            max_death_time[eclass] = d_time;

            if (mem_usage.count(ms))
            {
                for (uint32_t t = b_time; t <= d_time && t < mem_usage[ms].size(); ++t)
                {
                    mem_usage[ms][t] += size;
                }
            }
        }
        else
        {
            EClassId child = ctx.egraph.findConst(node.getChildren()[choice]);
            EClassId child_base = resolve_view_alias(child, ctx.egraph, ctx.selection_map, ctx.enodeInfos);
            EClassId target_base = ctx.get_inplace_alias(child_base);

            auto death_it = max_death_time.find(target_base);
            if (death_it != max_death_time.end())
            {
                uint32_t old_death = death_it->second;
                if (d_time > old_death)
                {
                    state.target_base = target_base;
                    state.old_death = old_death;
                    state.new_death = d_time;
                    max_death_time[target_base] = d_time;

                    if (mem_usage.count(ms))
                    {
                        for (uint32_t t = old_death; t <= d_time && t < mem_usage[ms].size(); ++t)
                        {
                            mem_usage[ms][t] += size;
                        }
                    }
                }
            }
        }
        undo_stack.push_back(state);
    }

    void on_pop(int choice, const BufferizeContext &ctx)
    {
        if (!enabled)
            return;

        UndoState state = undo_stack.back();
        undo_stack.pop_back();

        if (state.target_base.value != UINT32_MAX)
        {
            EClassId eclass = ctx.ordered[ctx.k];
            uint32_t sel = ctx.selection_map.at(eclass);
            ENodeId enode_id = ctx.egraph.getEClass(eclass).enodes[sel];
            const ENode &node = ctx.egraph.getENode(enode_id);
            MemSpace ms = node.getMemSpace();

            uint64_t size = getSizeBytes(node.getShape(), node.getDType());
            size = (size + 4095) & ~4095ULL;

            if (state.choice == -1)
            {
                max_death_time.erase(state.target_base);
            }
            else
            {
                max_death_time[state.target_base] = state.old_death;
            }

            if (mem_usage.count(ms))
            {
                for (uint32_t t = state.old_death; t <= state.new_death && t < mem_usage[ms].size(); ++t)
                {
                    mem_usage[ms][t] -= size;
                }
            }
        }
    }
};

class MemSpaceMismatchInplaceRule
{
  public:
    TG_PRUNING_RULE(MemSpaceMismatchInplaceRule)
    MemSpaceMismatchInplaceRule(bool en = true) : enabled(en)
    {
    }

    bool check(int candidate_choice, size_t /*candidate_choice_idx*/, const BufferizeContext &ctx) const
    {
        if (!enabled)
            return false;
        if (candidate_choice < 0)
            return false;

        EClassId eclass = ctx.ordered[ctx.k];
        uint32_t sel = ctx.selection_map.at(eclass);
        ENodeId enode_id = ctx.egraph.getEClass(eclass).enodes[sel];
        const ENode &node = ctx.egraph.getENode(enode_id);

        if (static_cast<size_t>(candidate_choice) >= node.getChildren().size())
            return false;

        EClassId child = ctx.egraph.findConst(node.getChildren()[candidate_choice]);
        EClassId child_base = resolve_view_alias(child, ctx.egraph, ctx.selection_map, ctx.enodeInfos);
        EClassId target_base = ctx.get_inplace_alias(child_base);

        auto base_sel_it = ctx.selection_map.find(target_base);
        if (base_sel_it == ctx.selection_map.end())
            return false;

        uint32_t base_sel = base_sel_it->second;
        ENodeId base_enode_id = ctx.egraph.getEClass(target_base).enodes[base_sel];
        const ENode &base_node = ctx.egraph.getENode(base_enode_id);

        return base_node.getMemSpace() != node.getMemSpace();
    }
};

class LinearChainInplaceDominationRule
{
  public:
    TG_PRUNING_RULE(LinearChainInplaceDominationRule)
    LinearChainInplaceDominationRule(bool en = true) : enabled(en)
    {
    }

    bool check(int candidate_choice, size_t /*candidate_choice_idx*/, const BufferizeContext &ctx) const
    {
        if (!enabled)
            return false;
        if (candidate_choice != -1)
            return false;

        EClassId eclass = ctx.ordered[ctx.k];
        uint32_t sel = ctx.selection_map.at(eclass);
        ENodeId enode_id = ctx.egraph.getEClass(eclass).enodes[sel];
        const ENode &node = ctx.egraph.getENode(enode_id);
        uint64_t out_size = getSizeBytes(node.getShape(), node.getDType());

        for (int choice : ctx.current_choices)
        {
            if (choice < 0)
                continue;

            if (static_cast<size_t>(choice) >= node.getChildren().size())
                continue;

            EClassId child = ctx.egraph.findConst(node.getChildren()[choice]);
            EClassId child_base = resolve_view_alias(child, ctx.egraph, ctx.selection_map, ctx.enodeInfos);

            auto death_it = ctx.death_times.find(child_base);
            if (death_it == ctx.death_times.end() || death_it->second != ctx.k)
                continue;

            EClassId target_base = ctx.get_inplace_alias(child_base);
            auto base_sel_it = ctx.selection_map.find(target_base);
            if (base_sel_it == ctx.selection_map.end())
                continue;

            uint32_t base_sel = base_sel_it->second;
            const ENode &base_node = ctx.egraph.getENode(ctx.egraph.getEClass(target_base).enodes[base_sel]);

            if (base_node.getOpType() == OpType::INPUT || base_node.getOpType() == OpType::CACHE)
                continue;

            if (base_node.getMemSpace() != node.getMemSpace())
                continue;

            uint64_t in_size = getSizeBytes(base_node.getShape(), base_node.getDType());
            if (out_size == in_size)
            {
                return true;
            }
        }

        return false;
    }
};

class IntervalSubsetDominationRule
{
  public:
    TG_PRUNING_RULE(IntervalSubsetDominationRule)
    IntervalSubsetDominationRule(bool en = true) : enabled(en)
    {
    }

    bool check(int candidate_choice, size_t /*candidate_choice_idx*/, const BufferizeContext &ctx) const
    {
        if (!enabled)
            return false;
        if (candidate_choice < 0)
            return false;

        EClassId eclass = ctx.ordered[ctx.k];
        uint32_t sel = ctx.selection_map.at(eclass);
        ENodeId enode_id = ctx.egraph.getEClass(eclass).enodes[sel];
        const ENode &node = ctx.egraph.getENode(enode_id);

        if (static_cast<size_t>(candidate_choice) >= node.getChildren().size())
            return false;

        EClassId cand_child = ctx.egraph.findConst(node.getChildren()[candidate_choice]);
        EClassId cand_child_base = resolve_view_alias(cand_child, ctx.egraph, ctx.selection_map, ctx.enodeInfos);
        EClassId cand_target_base = ctx.get_inplace_alias(cand_child_base);

        auto cand_sel_it = ctx.selection_map.find(cand_target_base);
        if (cand_sel_it == ctx.selection_map.end())
            return false;

        const ENode &cand_base_node =
            ctx.egraph.getENode(ctx.egraph.getEClass(cand_target_base).enodes[cand_sel_it->second]);
        uint64_t cand_size = getSizeBytes(cand_base_node.getShape(), cand_base_node.getDType());
        MemSpace cand_ms = cand_base_node.getMemSpace();

        auto cand_birth_it = ctx.birth_times.find(cand_target_base);
        if (cand_birth_it == ctx.birth_times.end())
            return false;
        uint32_t cand_birth = cand_birth_it->second;

        for (int other_choice : ctx.current_choices)
        {
            if (other_choice < 0 || other_choice == candidate_choice)
                continue;

            if (static_cast<size_t>(other_choice) >= node.getChildren().size())
                continue;

            EClassId other_child = ctx.egraph.findConst(node.getChildren()[other_choice]);
            EClassId other_child_base = resolve_view_alias(other_child, ctx.egraph, ctx.selection_map, ctx.enodeInfos);
            EClassId other_target_base = ctx.get_inplace_alias(other_child_base);

            auto other_sel_it = ctx.selection_map.find(other_target_base);
            if (other_sel_it == ctx.selection_map.end())
                continue;

            const ENode &other_base_node =
                ctx.egraph.getENode(ctx.egraph.getEClass(other_target_base).enodes[other_sel_it->second]);
            uint64_t other_size = getSizeBytes(other_base_node.getShape(), other_base_node.getDType());
            MemSpace other_ms = other_base_node.getMemSpace();

            if (other_ms != cand_ms || other_size != cand_size)
                continue;

            auto other_birth_it = ctx.birth_times.find(other_target_base);
            if (other_birth_it == ctx.birth_times.end())
                continue;
            uint32_t other_birth = other_birth_it->second;

            if (other_birth > cand_birth)
            {
                return true;
            }
        }

        return false;
    }
};

class CommutativeInplaceSymmetryRule
{
  public:
    TG_PRUNING_RULE(CommutativeInplaceSymmetryRule)
    CommutativeInplaceSymmetryRule(bool en = true) : enabled(en)
    {
    }

    bool check(int candidate_choice, size_t candidate_choice_idx, const BufferizeContext &ctx) const
    {
        if (!enabled)
            return false;
        if (candidate_choice < 0)
            return false;

        EClassId eclass = ctx.ordered[ctx.k];
        uint32_t sel = ctx.selection_map.at(eclass);
        ENodeId enode_id = ctx.egraph.getEClass(eclass).enodes[sel];
        const ENode &node = ctx.egraph.getENode(enode_id);

        if (static_cast<size_t>(candidate_choice) >= node.getChildren().size())
            return false;

        EClassId cand_child = ctx.egraph.findConst(node.getChildren()[candidate_choice]);
        EClassId cand_child_base = resolve_view_alias(cand_child, ctx.egraph, ctx.selection_map, ctx.enodeInfos);
        EClassId cand_target_base = ctx.get_inplace_alias(cand_child_base);

        auto cand_sel_it = ctx.selection_map.find(cand_target_base);
        if (cand_sel_it == ctx.selection_map.end())
            return false;

        const ENode &cand_base_node =
            ctx.egraph.getENode(ctx.egraph.getEClass(cand_target_base).enodes[cand_sel_it->second]);
        uint64_t cand_size = getSizeBytes(cand_base_node.getShape(), cand_base_node.getDType());
        MemSpace cand_ms = cand_base_node.getMemSpace();

        auto cand_birth_it = ctx.birth_times.find(cand_target_base);
        if (cand_birth_it == ctx.birth_times.end())
            return false;
        uint32_t cand_birth = cand_birth_it->second;

        for (size_t i = 0; i < candidate_choice_idx; ++i)
        {
            int other_choice = ctx.current_choices[i];
            if (other_choice < 0)
                continue;

            if (static_cast<size_t>(other_choice) >= node.getChildren().size())
                continue;

            EClassId other_child = ctx.egraph.findConst(node.getChildren()[other_choice]);
            EClassId other_child_base = resolve_view_alias(other_child, ctx.egraph, ctx.selection_map, ctx.enodeInfos);
            EClassId other_target_base = ctx.get_inplace_alias(other_child_base);

            auto other_sel_it = ctx.selection_map.find(other_target_base);
            if (other_sel_it == ctx.selection_map.end())
                continue;

            const ENode &other_base_node =
                ctx.egraph.getENode(ctx.egraph.getEClass(other_target_base).enodes[other_sel_it->second]);
            uint64_t other_size = getSizeBytes(other_base_node.getShape(), other_base_node.getDType());
            MemSpace other_ms = other_base_node.getMemSpace();

            if (other_ms != cand_ms || other_size != cand_size)
                continue;

            auto other_birth_it = ctx.birth_times.find(other_target_base);
            if (other_birth_it == ctx.birth_times.end())
                continue;
            uint32_t other_birth = other_birth_it->second;

            if (other_birth == cand_birth)
            {
                return true;
            }
        }

        return false;
    }
};

class DeadBufferReuseDominationRule
{
  public:
    TG_PRUNING_RULE(DeadBufferReuseDominationRule)
    DeadBufferReuseDominationRule(bool en = true) : enabled(en)
    {
    }

    bool check(int candidate_choice, size_t /*candidate_choice_idx*/, const BufferizeContext &ctx) const
    {
        if (!enabled)
            return false;
        if (candidate_choice != -1)
            return false;

        EClassId eclass = ctx.ordered[ctx.k];
        uint32_t sel = ctx.selection_map.at(eclass);
        ENodeId enode_id = ctx.egraph.getEClass(eclass).enodes[sel];
        const ENode &node = ctx.egraph.getENode(enode_id);
        uint64_t out_size = getSizeBytes(node.getShape(), node.getDType());

        for (int choice : ctx.current_choices)
        {
            if (choice < 0)
                continue;
            if (static_cast<size_t>(choice) >= node.getChildren().size())
                continue;

            EClassId child = ctx.egraph.findConst(node.getChildren()[choice]);
            EClassId child_base = resolve_view_alias(child, ctx.egraph, ctx.selection_map, ctx.enodeInfos);

            auto death_it = ctx.death_times.find(child_base);
            if (death_it == ctx.death_times.end() || death_it->second != ctx.k)
                continue;

            EClassId target_base = ctx.get_inplace_alias(child_base);
            auto base_sel_it = ctx.selection_map.find(target_base);
            if (base_sel_it == ctx.selection_map.end())
                continue;

            uint32_t base_sel = base_sel_it->second;
            const ENode &base_node = ctx.egraph.getENode(ctx.egraph.getEClass(target_base).enodes[base_sel]);
            if (base_node.getMemSpace() != node.getMemSpace())
                continue;

            uint64_t in_size = getSizeBytes(base_node.getShape(), base_node.getDType());
            if (out_size <= in_size)
                return true;
        }
        return false;
    }
};

// =============================================================================
// BufferizeIterator<Rules...>
// =============================================================================
template <typename... Rules> struct BufferizeIterator
{
  public:
    prune::PruningRuleSet<Rules...> rules;

    const std::vector<EClassId> &ordered;
    const EGraph &egraph;
    const std::unordered_map<EClassId, uint32_t> &selection_map;
    const std::vector<ENodeInfo> &enodeInfos;
    std::shared_ptr<SearchDelegate> delegate;
    const float *best_cost = nullptr;
    TimeoutChecker *timeout = nullptr;

    std::unordered_map<EClassId, uint32_t> birth_times;
    std::unordered_map<EClassId, uint32_t> death_times;
    std::vector<std::vector<int>> valid_choices;

    int k = 0;
    bool is_done = false;
    bool first_yield = true;
    std::vector<int> state;
    std::vector<std::vector<uint32_t>> choice_orders;
    std::unordered_map<EClassId, EClassId> inplace_alias;
    const std::unordered_map<MemSpace, uint64_t> &mem_caps;

    template <typename... Rs>
    BufferizeIterator(const std::vector<EClassId> &_ordered, const EGraph &_egraph,
                      const std::unordered_map<EClassId, uint32_t> &_selection_map,
                      const std::vector<ENodeInfo> &_enodeInfos,
                      const std::unordered_map<MemSpace, uint64_t> &_mem_caps,
                      std::shared_ptr<SearchDelegate> _delegate, const float *_best_cost = nullptr,
                      TimeoutChecker *_timeout = nullptr, Rs &&..._rules)
        : rules(std::forward<Rs>(_rules)...), ordered(_ordered), egraph(_egraph), selection_map(_selection_map),
          enodeInfos(_enodeInfos), mem_caps(_mem_caps), delegate(std::move(_delegate)), best_cost(_best_cost),
          timeout(_timeout)
    {
        init();
        BufferizeContext ctx{ordered,       egraph,      selection_map,
                             enodeInfos,    birth_times, death_times,
                             inplace_alias, {},          static_cast<uint32_t>(0),
                             mem_caps};
        rules.init(ctx);
    }

    bool can_abort()
    {
        return timeout && timeout->is_expired() && (best_cost != nullptr && *best_cost < TGConstants::INF);
    }

    void init()
    {
        uint32_t N = ordered.size();
        state.assign(N, 0);
        choice_orders.resize(N);
        valid_choices.resize(N);

        for (uint32_t i = 0; i < N; ++i)
        {
            EClassId eclass = ordered[i];
            birth_times[eclass] = i;
            death_times[eclass] = i + 1;
        }
        for (uint32_t i = 0; i < N; ++i)
        {
            EClassId eclass = ordered[i];
            auto sel_it = selection_map.find(eclass);
            if (sel_it == selection_map.end())
                continue;
            uint32_t sel = sel_it->second;
            ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
            const ENode &node = egraph.getENode(enode_id);
            for (EClassId child : node.getChildren())
            {
                EClassId child_base = resolve_view_alias(child, egraph, selection_map, enodeInfos);
                death_times[child_base] = std::max(death_times[child_base], i);
            }
        }

        for (uint32_t i = 0; i < N; ++i)
        {
            EClassId eclass = ordered[i];
            auto sel_it = selection_map.find(eclass);
            if (sel_it == selection_map.end())
                continue;
            uint32_t sel = sel_it->second;
            ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
            const ENode &node = egraph.getENode(enode_id);
            const ENodeInfo &info = enodeInfos[enode_id.value];

            if (info.is_view)
            {
                continue;
            }
            if (node.getOpType() == OpType::INPUT || node.getOpType() == OpType::CACHE)
            {
                valid_choices[i].push_back(-1);
                continue;
            }

            valid_choices[i].push_back(-1);

            if (node.getKernelId().value != 0)
            {
                const KernelEntry &kernel = KernelRegistry::get().getKernel(node.getKernelId());
                if (!kernel.safe_inplace_idxs.empty())
                {
                    for (uint32_t idx : kernel.safe_inplace_idxs)
                    {
                        if (idx < node.getChildren().size())
                        {
                            EClassId child = egraph.findConst(node.getChildren()[idx]);
                            EClassId child_base = resolve_view_alias(child, egraph, selection_map, enodeInfos);

                            if (death_times[child_base] == i)
                            {
                                auto c_sel_it = selection_map.find(child_base);
                                if (c_sel_it == selection_map.end())
                                    continue;
                                uint32_t c_sel = c_sel_it->second;
                                const ENode &c_node = egraph.getENode(egraph.getEClass(child_base).enodes[c_sel]);
                                if (c_node.getOpType() != OpType::INPUT && c_node.getOpType() != OpType::CACHE)
                                {
                                    uint64_t out_size = getSizeBytes(node.getShape(), node.getDType());
                                    uint64_t in_size = getSizeBytes(c_node.getShape(), c_node.getDType());
                                    if (out_size <= in_size)
                                    {
                                        valid_choices[i].push_back((int)idx);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if (delegate)
        {
            std::vector<float> node_features;
            std::vector<uint32_t> edge_src;
            std::vector<uint32_t> edge_dst;

            std::unordered_map<EClassId, uint32_t> eclass_to_idx;
            for (uint32_t i = 0; i < N; ++i)
            {
                eclass_to_idx[ordered[i]] = i;
            }

            for (uint32_t i = 0; i < N; ++i)
            {
                EClassId eclass = ordered[i];
                auto sel_it = selection_map.find(eclass);
                if (sel_it == selection_map.end())
                {
                    node_features.push_back(0.0f);
                    node_features.push_back(0.0f);
                    node_features.push_back(0.0f);
                    node_features.push_back(0.0f);
                    node_features.push_back(0.0f);
                    continue;
                }
                uint32_t sel = sel_it->second;
                ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
                const ENode &node = egraph.getENode(enode_id);
                const ENodeInfo &info = enodeInfos[enode_id.value];

                node_features.push_back((float)getSizeBytes(node.getShape(), node.getDType()));
                node_features.push_back((float)birth_times[eclass]);
                node_features.push_back((float)death_times[eclass]);
                node_features.push_back(info.is_view ? 1.0f : 0.0f);
                node_features.push_back(
                    (node.getOpType() == OpType::INPUT || node.getOpType() == OpType::CACHE) ? 1.0f : 0.0f);

                for (EClassId child : node.getChildren())
                {
                    EClassId child_canon = egraph.findConst(child);
                    if (eclass_to_idx.count(child_canon))
                    {
                        edge_src.push_back(eclass_to_idx[child_canon]);
                        edge_dst.push_back(i);
                    }
                }
            }
            delegate->init_bufferize_graph(node_features, edge_src, edge_dst);
        }
    }

    EClassId get_inplace_alias(EClassId id) const
    {
        auto it = inplace_alias.find(id);
        while (it != inplace_alias.end())
        {
            id = it->second;
            it = inplace_alias.find(id);
        }
        return id;
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
            if (delegate && valid_choices[k].size() > 1)
            {
                delegate->pop_state();
            }

            EClassId eclass = ordered[k];

            // Pop the rule state!
            uint32_t choice_idx = choice_orders[k][state[k] - 1];
            int choice = valid_choices[k][choice_idx];
            BufferizeContext pop_ctx{ordered,       egraph,           selection_map,
                                     enodeInfos,    birth_times,      death_times,
                                     inplace_alias, valid_choices[k], static_cast<uint32_t>(k),
                                     mem_caps};
            rules.on_pop(choice, pop_ctx);

            inplace_alias.erase(eclass);

            if (state[k] < valid_choices[k].size())
            {
                return true;
            }
            state[k] = 0;
            k--;
        }
        return false;
    }

    bool getNextBufferization(std::vector<ParallelBuffer> &out_buffers,
                              std::unordered_map<EClassId, BufferId> &out_eclass_to_buf)
    {
        LOG(DEBUG) << "getNextBufferization";
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

        uint32_t N = ordered.size();
        while (k >= 0)
        {
            if (can_abort())
            {
                is_done = true;
                return false;
            }

            if (k == static_cast<int>(N))
            {
                build_buffers(out_buffers, out_eclass_to_buf);
                return true;
            }

            if (valid_choices[k].empty())
            {
                k++;
                continue;
            }

            EClassId eclass = ordered[k];
            uint32_t sel = selection_map.at(eclass);
            ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
            const ENode &node = egraph.getENode(enode_id);

            if (state[k] == 0)
            {
                if (delegate && valid_choices[k].size() > 1)
                {
                    delegate->push_state();

                    std::vector<ActionFeatureBufferize> features;
                    uint64_t out_size = getSizeBytes(node.getShape(), node.getDType());

                    for (int choice : valid_choices[k])
                    {
                        ActionFeatureBufferize f;
                        f.mem_space = node.getMemSpace();
                        auto cap_it = mem_caps.find(node.getMemSpace());
                        f.mem_cap = (cap_it != mem_caps.end()) ? cap_it->second : 0;

                        if (choice == -1)
                        {
                            f.is_new_buffer = 1.0f;
                            f.size = out_size;
                            f.parent_size = 0;
                            f.parent_birth_time = 0.0f;
                        }
                        else
                        {
                            f.is_new_buffer = 0.0f;
                            f.size = out_size;
                            EClassId child = egraph.findConst(node.getChildren()[choice]);
                            EClassId child_base = resolve_view_alias(child, egraph, selection_map, enodeInfos);
                            EClassId parent_actual_base = get_inplace_alias(child_base);

                            auto c_sel_it = selection_map.find(child_base);
                            if (c_sel_it != selection_map.end())
                            {
                                uint32_t c_sel = c_sel_it->second;
                                const ENode &c_node = egraph.getENode(egraph.getEClass(child_base).enodes[c_sel]);
                                f.parent_size = getSizeBytes(c_node.getShape(), c_node.getDType());
                            }
                            else
                            {
                                f.parent_size = 0;
                            }
                            f.parent_birth_time = (float)birth_times[parent_actual_base];
                        }
                        features.push_back(f);
                    }
                    choice_orders[k] = delegate->order_bufferize(features);
                }
                else
                {
                    choice_orders[k].resize(valid_choices[k].size());
                    std::iota(choice_orders[k].begin(), choice_orders[k].end(), 0u);
                }
            }

            bool chosen = false;
            while (state[k] < valid_choices[k].size())
            {
                uint32_t choice_idx = choice_orders[k][state[k]];
                int choice = valid_choices[k][choice_idx];
                state[k]++;

                BufferizeContext ctx{ordered,       egraph,           selection_map,
                                     enodeInfos,    birth_times,      death_times,
                                     inplace_alias, valid_choices[k], static_cast<uint32_t>(k),
                                     mem_caps};
                if (rules.is_pruned(choice, choice_idx, ctx))
                {
                    continue;
                }

                // Push the rule state
                rules.on_push(choice, ctx);

                if (choice != -1)
                {
                    EClassId child = egraph.findConst(node.getChildren()[choice]);
                    EClassId child_base = resolve_view_alias(child, egraph, selection_map, enodeInfos);
                    inplace_alias[eclass] = get_inplace_alias(child_base);
                }
                else
                {
                    inplace_alias.erase(eclass);
                }

                chosen = true;
                k++;
                break;
            }

            if (!chosen)
            {
                state[k] = 0;
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

    void build_buffers(std::vector<ParallelBuffer> &out_buffers,
                       std::unordered_map<EClassId, BufferId> &out_eclass_to_buf)
    {
        out_buffers.clear();
        out_eclass_to_buf.clear();

        std::unordered_map<EClassId, uint32_t> act_birth_times = birth_times;
        std::unordered_map<EClassId, uint32_t> act_death_times = death_times;

        for (uint32_t i = 0; i < ordered.size(); ++i)
        {
            EClassId eclass = ordered[i];
            auto sel_it = selection_map.find(eclass);
            if (sel_it == selection_map.end())
                continue;
            uint32_t sel = sel_it->second;
            ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
            const ENodeInfo &info = enodeInfos[enode_id.value];

            EClassId base = eclass;
            if (info.is_view)
            {
                base = resolve_view_alias(eclass, egraph, selection_map, enodeInfos);
            }
            EClassId target_base = get_inplace_alias(base);

            if (target_base != eclass)
            {
                if (act_birth_times.count(target_base))
                    act_birth_times[target_base] = std::min(act_birth_times[target_base], act_birth_times[eclass]);
                if (act_death_times.count(target_base))
                    act_death_times[target_base] = std::max(act_death_times[target_base], act_death_times[eclass]);
            }
        }

        std::unordered_map<EClassId, BufferId> base_to_buf;
        for (EClassId eclass : ordered)
        {
            auto sel_it = selection_map.find(eclass);
            if (sel_it == selection_map.end())
                continue;

            EClassId base = eclass;
            uint32_t sel = sel_it->second;
            ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
            if (enodeInfos[enode_id.value].is_view)
            {
                base = resolve_view_alias(eclass, egraph, selection_map, enodeInfos);
            }
            EClassId target_base = get_inplace_alias(base);
            if (selection_map.find(target_base) == selection_map.end())
            {
                target_base = base;
            }
            if (selection_map.find(target_base) == selection_map.end())
            {
                target_base = eclass;
            }

            if (base_to_buf.find(target_base) == base_to_buf.end())
            {
                auto base_sel_it = selection_map.find(target_base);
                if (base_sel_it == selection_map.end())
                    continue;

                BufferId buf_id = BufferId{(uint32_t)out_buffers.size()};
                base_to_buf[target_base] = buf_id;

                uint32_t base_sel = base_sel_it->second;
                ENodeId base_enode_id = egraph.getEClass(target_base).enodes[base_sel];
                const ENode &base_node = egraph.getENode(base_enode_id);

                uint64_t size_bytes = getSizeBytes(base_node.getShape(), base_node.getDType());
                if (size_bytes == 0)
                {
                    Error::throw_err("empty node");
                }
                size_bytes = (size_bytes + 4095) & ~4095ULL;

                uint32_t b_time = act_birth_times.count(target_base) ? act_birth_times[target_base] : 0;
                uint32_t d_time = act_death_times.count(target_base) ? act_death_times[target_base] : 1;

                ParallelBuffer buf = {buf_id, base_node.getMemSpace(), size_bytes, b_time, d_time, -1};
                out_buffers.push_back(std::move(buf));
            }
            out_eclass_to_buf[eclass] = base_to_buf[target_base];
        }
    }
};

template <typename... Rules>
BufferizeIterator<std::decay_t<Rules>...> makeBufferizeIterator(
    const std::vector<EClassId> &ordered, const EGraph &egraph,
    const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<ENodeInfo> &enodeInfos,
    const std::unordered_map<MemSpace, uint64_t> &mem_caps, const float *best_cost = nullptr,
    TimeoutChecker *timeout = nullptr, Rules &&...rules)
{
    return BufferizeIterator<std::decay_t<Rules>...>(ordered, egraph, selection_map, enodeInfos, mem_caps, nullptr,
                                                     best_cost, timeout, std::forward<Rules>(rules)...);
}

template <typename... Rules>
BufferizeIterator<std::decay_t<Rules>...> makeBufferizeIteratorWithDelegate(
    const std::vector<EClassId> &ordered, const EGraph &egraph,
    const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<ENodeInfo> &enodeInfos,
    const std::unordered_map<MemSpace, uint64_t> &mem_caps, std::shared_ptr<SearchDelegate> delegate,
    const float *best_cost = nullptr, TimeoutChecker *timeout = nullptr, Rules &&...rules)
{
    return BufferizeIterator<std::decay_t<Rules>...>(ordered, egraph, selection_map, enodeInfos, mem_caps,
                                                     std::move(delegate), best_cost, timeout,
                                                     std::forward<Rules>(rules)...);
}

using AllBufferizeRuleTypes =
    std::tuple<MemSpaceMismatchInplaceRule, LinearChainInplaceDominationRule, IntervalSubsetDominationRule,
               CommutativeInplaceSymmetryRule, DeadBufferReuseDominationRule, PeakMemoryPruningRule>;

template <typename BoolTuple>
inline auto makeConfiguredBufferizeIteratorFromBools(const std::vector<EClassId> &ordered, const EGraph &egraph,
                                                     const std::unordered_map<EClassId, uint32_t> &selection_map,
                                                     const std::vector<ENodeInfo> &enodeInfos,
                                                     const std::unordered_map<MemSpace, uint64_t> &mem_caps,
                                                     std::shared_ptr<SearchDelegate> delegate,
                                                     const BoolTuple &bool_flags, const float *best_cost = nullptr,
                                                     TimeoutChecker *timeout = nullptr)
{
    return std::apply(
        [&](auto &&...rs) {
            return makeBufferizeIteratorWithDelegate(ordered, egraph, selection_map, enodeInfos, mem_caps,
                                                     std::move(delegate), best_cost, timeout, rs...);
        },
        prune::instantiate_from_bools<AllBufferizeRuleTypes>(bool_flags));
}

inline auto makeConfiguredBufferizeIterator(const std::vector<EClassId> &ordered, const EGraph &egraph,
                                            const std::unordered_map<EClassId, uint32_t> &selection_map,
                                            const std::vector<ENodeInfo> &enodeInfos,
                                            const std::unordered_map<MemSpace, uint64_t> &mem_caps,
                                            std::shared_ptr<SearchDelegate> delegate, const Settings &settings,
                                            const float *best_cost = nullptr, TimeoutChecker *timeout = nullptr)
{
    settings.validate_rules("bufferize");
    auto bool_flags = prune::extract_enabled_states<AllBufferizeRuleTypes>("bufferize", settings);
    return makeConfiguredBufferizeIteratorFromBools(ordered, egraph, selection_map, enodeInfos, mem_caps,
                                                    std::move(delegate), bool_flags, best_cost, timeout);
}

inline auto makeConfiguredBufferizeIterator(const std::vector<EClassId> &ordered, const EGraph &egraph,
                                            const std::unordered_map<EClassId, uint32_t> &selection_map,
                                            const std::vector<ENodeInfo> &enodeInfos,
                                            const std::unordered_map<MemSpace, uint64_t> &mem_caps,
                                            const Settings &settings, const float *best_cost = nullptr,
                                            TimeoutChecker *timeout = nullptr)
{
    return makeConfiguredBufferizeIterator(ordered, egraph, selection_map, enodeInfos, mem_caps, nullptr, settings,
                                           best_cost, timeout);
}

// =============================================================================
// Malloc Context & Rules
// =============================================================================
struct MallocContext
{
    uint64_t mem_cap;
    const std::vector<ParallelBuffer> &unallocated;
    const std::vector<int64_t> &unallocated_sizes;
    const std::vector<int> &avail;
    const std::vector<int64_t> &current_offsets;
    const std::vector<std::vector<int>> &adj;
    const std::vector<int> &order;
    const std::vector<int64_t> &chosen_offset;
    const std::vector<int64_t> &global_offset_max;
    int k;
    int idx;
    int64_t offset;
    int64_t h_min;
};

class OffsetMonotoneRule
{
  public:
    TG_PRUNING_RULE(OffsetMonotoneRule)
    OffsetMonotoneRule(bool en = true) : enabled(en)
    {
    }
    bool check(int /*cand*/, size_t /*cand_idx*/, const MallocContext &c) const
    {
        if (!enabled)
            return false;
        return c.offset < c.global_offset_max[c.k];
    }
};

class IdMaxSymmetryRule
{
  public:
    TG_PRUNING_RULE(IdMaxSymmetryRule)
    IdMaxSymmetryRule(bool en = true) : enabled(en)
    {
    }
    bool check(int /*cand*/, size_t /*cand_idx*/, const MallocContext &c) const
    {
        if (!enabled)
            return false;
        BufferId id_max = BufferId{0};
        for (int d = c.k - 1; d >= 0; --d)
        {
            if (c.chosen_offset[d] == c.offset)
            {
                id_max = std::max(id_max, c.unallocated[c.order[d]].id);
            }
            else
            {
                break;
            }
        }
        return c.unallocated[c.avail[c.idx]].id < id_max;
    }
};

class CapRespectRule
{
  public:
    TG_PRUNING_RULE(CapRespectRule)
    CapRespectRule(bool en = true) : enabled(en)
    {
    }
    bool check(int /*cand*/, size_t /*cand_idx*/, const MallocContext &c) const
    {
        if (!enabled)
            return false;
        if (c.mem_cap == std::numeric_limits<uint64_t>::max())
            return false;
        return static_cast<uint64_t>(c.offset) + c.unallocated_sizes[c.avail[c.idx]] > c.mem_cap;
    }
};

class HMinBoundRule
{
  public:
    TG_PRUNING_RULE(HMinBoundRule)
    HMinBoundRule(bool en = true) : enabled(en)
    {
    }
    bool check(int /*cand*/, size_t /*cand_idx*/, const MallocContext &c) const
    {
        if (!enabled)
            return false;
        return c.offset >= c.h_min;
    }
};

// =============================================================================
// MallocIterator<Rules...>
// =============================================================================
template <typename... Rules> struct MallocIterator
{
  public:
    prune::PruningRuleSet<Rules...> rules;

    uint64_t mem_cap;
    const std::vector<ParallelBuffer> &unallocated;
    std::shared_ptr<SearchDelegate> delegate;
    const float *best_cost = nullptr;
    TimeoutChecker *timeout = nullptr;

    int N;
    std::vector<int64_t> unallocated_sizes;
    std::vector<std::vector<int>> adj;

    std::vector<int> avail;
    std::vector<int> state;
    std::vector<int> order;
    std::vector<int64_t> chosen_offset;
    std::vector<int64_t> global_offset_max;
    std::vector<int64_t> current_offsets;
    std::vector<std::vector<uint32_t>> choice_orders;

    struct Backup
    {
        int j;
        int64_t old_val;
    };
    std::vector<Backup> trail;
    std::vector<size_t> trail_starts;

    int k = 0;

    bool is_done = false;

    template <typename... Rs>
    MallocIterator(uint64_t _mem_cap, const std::vector<ParallelBuffer> &_unallocated,
                   std::shared_ptr<SearchDelegate> _delegate, const float *_best_cost = nullptr,
                   TimeoutChecker *_timeout = nullptr, Rs &&..._rules)
        : rules(std::forward<Rs>(_rules)...), mem_cap(_mem_cap), unallocated(_unallocated),
          delegate(std::move(_delegate)), best_cost(_best_cost), timeout(_timeout),
          N(static_cast<int>(_unallocated.size()))
    {
        unallocated_sizes.resize(N);
        for (int i = 0; i < N; ++i)
            unallocated_sizes[i] = unallocated[i].size;

        adj.resize(N);
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                if (i != j && overlapsBuf(unallocated[i], unallocated[j]))
                {
                    adj[i].push_back(j);
                }
            }
        }

        if (delegate)
        {
            std::vector<float> node_features;
            std::vector<uint32_t> edge_src;
            std::vector<uint32_t> edge_dst;

            for (int i = 0; i < N; ++i)
            {
                node_features.push_back((float)unallocated[i].size);
                node_features.push_back((float)unallocated[i].start);
                node_features.push_back((float)unallocated[i].end);

                for (int j : adj[i])
                {
                    edge_src.push_back(i);
                    edge_dst.push_back(j);
                }
            }
            delegate->init_malloc_graph(node_features, edge_src, edge_dst);
        }

        avail.resize(N);
        std::iota(avail.begin(), avail.end(), 0);
        state.assign(N, 0);
        state[0] = 0;
        order.assign(N, 0);
        chosen_offset.assign(N, 0);
        global_offset_max.assign(N + 1, 0);
        current_offsets.assign(N, 0);
        choice_orders.resize(N);
        trail.reserve(N * 50);
        trail_starts.assign(N + 1, 0);
    }

    bool can_abort()
    {
        return timeout && timeout->is_expired() && (best_cost != nullptr && *best_cost < TGConstants::INF);
    }

    bool getNextAllocation(std::vector<ParallelBuffer> &allocated)
    {
        if (is_done)
            return false;
        ProgressTimer t(0, "malloc", false, true);
        if (unallocated.empty())
            return true;

        while (k >= 0)
        {
            if (can_abort())
            {
                is_done = true;
                return false;
            }

            if (k % 100 == 0)
            {
                LOG(DEBUG) << "malloc k=" << std::to_string(k) << "/" << std::to_string(N);
            }

            if (k == N)
            {
                for (int d = 0; d < N; ++d)
                {
                    ParallelBuffer buf = unallocated[order[d]];
                    buf.offset = chosen_offset[d];
                    allocated.push_back(buf);
                }
                is_done = true;
                return true;
            }

            if (state[k] == (k == 0 ? 0 : k))
            {
                if (delegate)
                {
                    delegate->push_state();

                    std::vector<ActionFeatureMalloc> features;
                    for (int idx = k; idx < N; ++idx)
                    {
                        ActionFeatureMalloc f;
                        f.size = unallocated[avail[idx]].size;
                        f.start = unallocated[avail[idx]].start;
                        f.end = unallocated[avail[idx]].end;
                        f.mem_space = unallocated[avail[idx]].mem_space;
                        f.mem_cap = mem_cap;
                        features.push_back(f);
                    }
                    choice_orders[k] = delegate->order_malloc(features);
                }
                else
                {
                    choice_orders[k].resize(N - k);
                    std::iota(choice_orders[k].begin(), choice_orders[k].end(), 0u);
                }
            }

            bool advanced = false;

            int64_t h_min = std::numeric_limits<int64_t>::max();
            for (int idx = k; idx < N; ++idx)
            {
                int i = avail[idx];
                int64_t h = current_offsets[i] + unallocated_sizes[i];
                if (h < h_min)
                    h_min = h;
            }

            while (state[k] < N)
            {
                uint32_t sel_idx = state[k] - k;
                int mapped_idx = k + choice_orders[k][sel_idx];

                int idx = mapped_idx;
                state[k]++;
                int i = avail[idx];
                int64_t offset_i = current_offsets[i];

                MallocContext ctx{mem_cap,
                                  unallocated,
                                  unallocated_sizes,
                                  avail,
                                  current_offsets,
                                  adj,
                                  order,
                                  chosen_offset,
                                  global_offset_max,
                                  static_cast<int>(k),
                                  static_cast<int>(idx),
                                  offset_i,
                                  h_min};
                if (rules.is_pruned(/*cand=*/int{}, /*cand_idx=*/size_t{}, ctx))
                {
                    continue;
                }

                std::swap(avail[k], avail[idx]);
                order[k] = i;
                chosen_offset[k] = offset_i;
                global_offset_max[k + 1] = std::max(global_offset_max[k], offset_i);

                trail_starts[k] = trail.size();
                int64_t new_end = offset_i + unallocated_sizes[i];

                for (int j : adj[i])
                {
                    if (current_offsets[j] < new_end)
                    {
                        trail.push_back({j, current_offsets[j]});
                        current_offsets[j] = new_end;
                    }
                }

                advanced = true;
                break;
            }

            if (advanced)
            {
                k++;
                if (k < N)
                    state[k] = k;
            }
            else
            {
                if (delegate)
                    delegate->pop_state();
                k--;
                if (k >= 0)
                {
                    int idx = state[k] - 1;
                    std::swap(avail[k], avail[idx]);

                    size_t ts = trail_starts[k];
                    while (trail.size() > ts)
                    {
                        current_offsets[trail.back().j] = trail.back().old_val;
                        trail.pop_back();
                    }
                }
            }
        }
        return false;
    }
};

template <typename... Rules>
MallocIterator<std::decay_t<Rules>...> makeMallocIterator(uint64_t mem_cap,
                                                          const std::vector<ParallelBuffer> &unallocated,
                                                          const float *best_cost = nullptr,
                                                          TimeoutChecker *timeout = nullptr, Rules &&...rules)
{
    return MallocIterator<std::decay_t<Rules>...>(mem_cap, unallocated, nullptr, best_cost, timeout,
                                                  std::forward<Rules>(rules)...);
}

template <typename... Rules>
MallocIterator<std::decay_t<Rules>...> makeMallocIteratorWithDelegate(
    uint64_t mem_cap, const std::vector<ParallelBuffer> &unallocated, std::shared_ptr<SearchDelegate> delegate,
    const float *best_cost = nullptr, TimeoutChecker *timeout = nullptr, Rules &&...rules)
{
    return MallocIterator<std::decay_t<Rules>...>(mem_cap, unallocated, std::move(delegate), best_cost, timeout,
                                                  std::forward<Rules>(rules)...);
}

using AllMallocRuleTypes = std::tuple<OffsetMonotoneRule, IdMaxSymmetryRule, HMinBoundRule>;

template <typename BoolTuple>
inline auto makeConfiguredMallocIteratorFromBools(uint64_t mem_cap, const std::vector<ParallelBuffer> &unallocated,
                                                  std::shared_ptr<SearchDelegate> delegate, const BoolTuple &bool_flags,
                                                  const float *best_cost = nullptr, TimeoutChecker *timeout = nullptr)
{
    return std::apply(
        [&](auto &&...rs) {
            return makeMallocIteratorWithDelegate(mem_cap, unallocated, std::move(delegate), best_cost, timeout,
                                                  CapRespectRule(true), rs...);
        },
        prune::instantiate_from_bools<AllMallocRuleTypes>(bool_flags));
}

inline auto makeConfiguredMallocIterator(uint64_t mem_cap, const std::vector<ParallelBuffer> &unallocated,
                                         std::shared_ptr<SearchDelegate> delegate, const Settings &settings,
                                         const float *best_cost = nullptr, TimeoutChecker *timeout = nullptr)
{
    settings.validate_rules("malloc");
    auto bool_flags = prune::extract_enabled_states<AllMallocRuleTypes>("malloc", settings);
    return makeConfiguredMallocIteratorFromBools(mem_cap, unallocated, std::move(delegate), bool_flags, best_cost,
                                                 timeout);
}

static bool check_peak_memory(const std::vector<ParallelBuffer> &bufs, uint64_t mem_cap, BufferId &overflow)
{
    if (mem_cap == std::numeric_limits<uint64_t>::max())
        return true;

    struct Event
    {
        uint32_t time;
        int type;
        int64_t size;
        BufferId buffer_id;
    };

    std::vector<Event> events;
    events.reserve(bufs.size() * 2);
    for (const auto &b : bufs)
    {
        events.push_back({b.start, 1, static_cast<int64_t>(b.size), b.id});
        events.push_back({b.end, 0, static_cast<int64_t>(b.size), b.id});
    }

    std::sort(events.begin(), events.end(), [](const Event &a, const Event &b) {
        if (a.time != b.time)
            return a.time < b.time;
        return a.type < b.type;
    });

    int64_t current_mem = 0;
    for (const auto &ev : events)
    {
        if (ev.type == 1)
        {
            current_mem += ev.size;
            if (current_mem > static_cast<int64_t>(mem_cap))
            {
                overflow = ev.buffer_id;
                LOG(INFO) << "[check_peak_memory] OOM error at idx=" << ev.time << std::endl;
                return false;
            }
        }
        else
        {
            current_mem -= ev.size;
        }
    }
    return true;
}

static bool greedy_alloc(uint64_t mem_cap, const std::vector<ParallelBuffer> &unallocated,
                         std::vector<ParallelBuffer> &allocated, BufferId &overflow)
{
    std::vector<ParallelBuffer> bufs = unallocated;

    std::sort(bufs.begin(), bufs.end(), [](const ParallelBuffer &a, const ParallelBuffer &b) {
        if (a.size != b.size)
            return a.size > b.size;
        return a.id < b.id;
    });

    allocated.clear();
    allocated.reserve(bufs.size());

    for (auto &buf : bufs)
    {
        int64_t best_offset = 0;

        std::vector<const ParallelBuffer *> time_overlaps;
        for (const auto &alloc : allocated)
        {
            if (overlapsBuf(buf, alloc))
            {
                time_overlaps.push_back(&alloc);
            }
        }

        std::sort(time_overlaps.begin(), time_overlaps.end(),
                  [](const ParallelBuffer *a, const ParallelBuffer *b) { return a->offset < b->offset; });

        for (const auto *alloc : time_overlaps)
        {
            if (best_offset < alloc->offset + static_cast<int64_t>(alloc->size) &&
                best_offset + static_cast<int64_t>(buf.size) > alloc->offset)
            {
                best_offset = alloc->offset + alloc->size;
            }
        }

        if (mem_cap != std::numeric_limits<uint64_t>::max() &&
            best_offset + static_cast<int64_t>(buf.size) > static_cast<int64_t>(mem_cap))
        {
            overflow = buf.id;
            return false;
        }

        buf.offset = best_offset;
        allocated.push_back(buf);
    }
    return true;
}

static bool malloc_by_time_components(uint64_t mem_cap, const std::vector<ParallelBuffer> &unallocated,
                                      std::vector<ParallelBuffer> &allocated, BufferId &overflow,
                                      std::shared_ptr<SearchDelegate> delegate = nullptr,
                                      const Settings *settings_ptr = nullptr, const float *best_cost = nullptr,
                                      TimeoutChecker *timeout = nullptr)
{
    if (unallocated.empty())
        return true;

    std::vector<ParallelBuffer> sorted_bufs = unallocated;
    std::sort(sorted_bufs.begin(), sorted_bufs.end(), [](const ParallelBuffer &a, const ParallelBuffer &b) {
        if (a.start != b.start)
            return a.start < b.start;
        return a.end < b.end;
    });

    if (sorted_bufs.empty())
        return true;

    if (!check_peak_memory(sorted_bufs, mem_cap, overflow))
    {
        return false;
    }

    std::vector<ParallelBuffer> comp_allocated;
    if (greedy_alloc(mem_cap, sorted_bufs, comp_allocated, overflow))
    {
        allocated.insert(allocated.end(), comp_allocated.begin(), comp_allocated.end());
        return true;
    }

    std::sort(sorted_bufs.begin(), sorted_bufs.end(), [](const ParallelBuffer &a, const ParallelBuffer &b) {
        if (a.size != b.size)
            return a.size > b.size;
        return a.id < b.id;
    });

    const Settings &active_settings = settings_ptr ? *settings_ptr : Settings::get_default();
    auto iter = makeConfiguredMallocIterator(mem_cap, sorted_bufs, delegate, active_settings, best_cost, timeout);
    if (!iter.getNextAllocation(comp_allocated))
    {
        return false;
    }

    allocated.insert(allocated.end(), comp_allocated.begin(), comp_allocated.end());
    return true;
}

struct MemValidator : public ISelectionValidator
{
    const EGraph &egraph;
    const std::vector<ENodeInfo> &enodeInfos;
    const std::unordered_map<MemSpace, uint64_t> &mem_caps;
    const std::unordered_map<EClassId, LogicalId> &eclassToLogical;
    const std::unordered_map<LogicalId, ParallelBuffer> &preallocatedBuffers;
    std::shared_ptr<SearchDelegate> delegate;
    const Settings *settings_ptr = nullptr;
    const float *best_cost = nullptr;
    TimeoutChecker *timeout = nullptr;

    MemValidator(const EGraph &_egraph, const std::vector<ENodeInfo> &_enodeInfos,
                 const std::unordered_map<MemSpace, uint64_t> &_mem_caps,
                 const std::unordered_map<EClassId, LogicalId> &_eclassToLogical,
                 const std::unordered_map<LogicalId, ParallelBuffer> &_preallocatedBuffers,
                 std::shared_ptr<SearchDelegate> _delegate = nullptr, const Settings *_settings = nullptr,
                 const float *_best_cost = nullptr, TimeoutChecker *_timeout = nullptr)
        : egraph(_egraph), enodeInfos(_enodeInfos), mem_caps(_mem_caps), eclassToLogical(_eclassToLogical),
          preallocatedBuffers(_preallocatedBuffers), delegate(_delegate), settings_ptr(_settings),
          best_cost(_best_cost), timeout(_timeout)
    {
    }

    bool validate(const std::unordered_map<EClassId, uint32_t> &selection_map, const std::vector<EClassId> &order,
                  const std::vector<EClassId> &path, std::vector<ParallelBuffer> &buffers,
                  std::unordered_map<EClassId, BufferId> &eclass_to_buf, float &cost,
                  std::vector<EClassId> &conflict_nodes) override
    {
        cost = get_cost(order, egraph, selection_map, enodeInfos);

        const Settings &active_settings = settings_ptr ? *settings_ptr : Settings::get_default();
        auto buf_iter = makeConfiguredBufferizeIterator(order, egraph, selection_map, enodeInfos, mem_caps, delegate,
                                                        active_settings, best_cost, timeout);

        std::vector<ParallelBuffer> unallocated_buffers;
        std::unordered_map<EClassId, BufferId> eclass_to_buf_local;

        std::unordered_set<EClassId> all_conflict_nodes;
        bool any_alloc_ok = false;

        while (buf_iter.getNextBufferization(unallocated_buffers, eclass_to_buf_local))
        {
            std::unordered_set<BufferId> preallocated_buf_ids;
            std::unordered_map<BufferId, ParallelBuffer> preallocated_overrides;
            std::unordered_map<MemSpace, uint64_t> reserved_per_ms;
            for (EClassId eclass : order)
            {
                auto logicalIt = eclassToLogical.find(eclass);
                if (logicalIt == eclassToLogical.end())
                    continue;

                auto sel_it = selection_map.find(eclass);
                if (sel_it == selection_map.end())
                    continue;
                uint32_t sel = sel_it->second;
                ENodeId enode_id = egraph.getEClass(eclass).enodes[sel];
                const ENode &node = egraph.getENode(enode_id);
                if (node.getOpType() != OpType::INPUT && node.getOpType() != OpType::CACHE)
                    continue;

                auto preIt = preallocatedBuffers.find(logicalIt->second);
                if (preIt == preallocatedBuffers.end())
                    continue;

                BufferId buf_id = eclass_to_buf_local.at(eclass);
                preallocated_buf_ids.insert(buf_id);
                preallocated_overrides[buf_id] = preIt->second;

                const ParallelBuffer &pre = preIt->second;
                uint64_t extent = static_cast<uint64_t>(pre.offset) + pre.size;
                uint64_t &cur = reserved_per_ms[pre.mem_space];
                cur = std::max(cur, extent);
            }

            std::unordered_map<MemSpace, std::vector<ParallelBuffer>> buf_by_mem_space;
            for (auto &buf : unallocated_buffers)
            {
                if (buf.mem_space.type == HandleType::STORAGE)
                    continue;
                if (preallocated_buf_ids.count(buf.id))
                    continue;
                buf_by_mem_space[buf.mem_space].push_back(buf);
            }

            std::vector<ParallelBuffer> current_buffers;
            current_buffers.reserve(unallocated_buffers.size());

            for (auto &buf : unallocated_buffers)
            {
                if (buf.mem_space.type == HandleType::STORAGE)
                {
                    buf.offset = 0;
                    current_buffers.push_back(buf);
                }
            }

            for (auto &buf : unallocated_buffers)
            {
                if (preallocated_buf_ids.count(buf.id))
                {
                    ParallelBuffer pre = preallocated_overrides.at(buf.id);
                    buf.offset = pre.offset;
                    current_buffers.push_back(buf);
                }
            }

            bool alloc_ok = true;
            BufferId overflow;
            MemSpace failed_ms;
            uint64_t failed_reduced_cap = std::numeric_limits<uint64_t>::max();

            for (auto &kv : buf_by_mem_space)
            {
                MemSpace ms = kv.first;
                auto &bufs = kv.second;
                uint64_t cap = mem_caps.count(ms) ? mem_caps.at(ms) : std::numeric_limits<uint64_t>::max();
                uint64_t reserved = reserved_per_ms.count(ms) ? reserved_per_ms.at(ms) : 0;
                uint64_t reduced_cap =
                    (cap == std::numeric_limits<uint64_t>::max()) ? cap : (cap > reserved ? cap - reserved : 0);

                std::vector<ParallelBuffer> allocated;
                if (!malloc_by_time_components(reduced_cap, bufs, allocated, overflow, delegate, settings_ptr,
                                               best_cost, timeout))
                {
                    alloc_ok = false;
                    failed_ms = ms;
                    failed_reduced_cap = reduced_cap;
                    LOG(INFO) << "[MemValidator] OOM error in mem_space (" << ms.idx << ", " << (int)ms.type << ")"
                              << std::endl;
                    break;
                }

                for (auto &buf : allocated)
                {
                    buf.offset += static_cast<int64_t>(reserved);
                }

                current_buffers.insert(current_buffers.end(), std::make_move_iterator(allocated.begin()),
                                       std::make_move_iterator(allocated.end()));
            }

            if (alloc_ok)
            {
                buffers = std::move(current_buffers);
                eclass_to_buf = std::move(eclass_to_buf_local);
                any_alloc_ok = true;
                break;
            }

            const ParallelBuffer *overflow_buf_ptr = nullptr;
            for (const auto &b : unallocated_buffers)
            {
                if (b.id == overflow)
                {
                    overflow_buf_ptr = &b;
                    break;
                }
            }

            std::unordered_map<BufferId, uint64_t> overlapping_buf_sizes;
            std::unordered_set<BufferId> overflows;

            if (overflow_buf_ptr)
            {
                for (const auto &b : unallocated_buffers)
                {
                    if (b.mem_space == failed_ms && overlapsBuf(*overflow_buf_ptr, b))
                    {
                        overflows.insert(b.id);
                        overlapping_buf_sizes[b.id] = b.size;
                    }
                }
            }

            std::unordered_set<BufferId> seen_bufs;
            uint64_t running_sum = 0;

            for (EClassId node_in_path : path)
            {
                auto it = eclass_to_buf_local.find(node_in_path);
                if (it == eclass_to_buf_local.end())
                    continue;

                bool added = false;
                BufferId buf_id = it->second;
                auto size_it = overlapping_buf_sizes.find(buf_id);
                if (size_it != overlapping_buf_sizes.end())
                {
                    if (seen_bufs.insert(buf_id).second)
                    {
                        running_sum += size_it->second;
                        all_conflict_nodes.insert(node_in_path);
                        added = true;
                    }
                }

                if (!added)
                {
                    auto sel_it = selection_map.find(node_in_path);
                    if (sel_it != selection_map.end())
                    {
                        uint32_t sel = sel_it->second;
                        ENodeId enode_id = egraph.getEClass(node_in_path).enodes[sel];
                        const ENode &node = egraph.getENode(enode_id);
                        for (EClassId child : node.getChildren())
                        {
                            EClassId canon_child = egraph.findConst(child);
                            auto child_buf_it = eclass_to_buf_local.find(canon_child);
                            if (child_buf_it != eclass_to_buf_local.end() && overflows.count(child_buf_it->second))
                            {
                                all_conflict_nodes.insert(node_in_path);
                                break;
                            }
                        }
                    }
                }

                if (failed_reduced_cap != std::numeric_limits<uint64_t>::max() && running_sum > failed_reduced_cap)
                {
                    break;
                }
            }
        }

        if (any_alloc_ok)
        {
            return true;
        }

        conflict_nodes.insert(conflict_nodes.end(), all_conflict_nodes.begin(), all_conflict_nodes.end());
        return false;
    }
};