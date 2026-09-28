// tensor_graphs_cpp/core/plan/search_engine.hpp
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <array>
#ifdef TG_PROFILE
#include <iomanip>
#endif
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/constants.hpp"
#include "core/logging.hpp"
#include "core/plan/brancher.hpp"
#include "core/plan/domain.hpp"
#include "core/plan/propagator.hpp"
#include "core/plan/search_node.hpp"
#include "core/plan/search_state.hpp"
#include "core/plan/selector.hpp"
#include "core/timer.hpp"

namespace plan
{

struct ExtractionResult
{
    std::unordered_map<EClassId, uint32_t> selection_map;
    std::vector<EClassId> order;
    std::vector<ParallelBuffer> buffers;
    std::unordered_map<EClassId, BufferId> eclass_to_buf;
    float cost = TGConstants::INF;
    std::unordered_map<EClassId, float> eclass_to_cost;
};

class SearchEngine
{
  public:
    SearchState state;
    std::shared_ptr<Selector> selector;
    std::shared_ptr<Brancher> brancher;
    std::vector<std::unique_ptr<Propagator>> propagators;

    std::vector<std::shared_ptr<SearchNode>> all_nodes;
    uint32_t current_node_id = UINT32_MAX;

    std::vector<VarId> prop_worklist;
    std::vector<uint32_t> prop_queued_epoch;
    uint32_t current_prop_epoch = 1;
    std::vector<uint32_t> restore_current_path;
    std::vector<uint32_t> restore_target_path;

    float incumbent_best_cost = TGConstants::INF;
    std::vector<ExtractionResult> incumbent_extractions;
    std::unordered_set<BaseEClassId> incumbent_cached_nodes;

    SearchEngine(SearchState state, std::shared_ptr<Selector> selector = nullptr,
                 std::shared_ptr<Brancher> brancher = nullptr)
        : state(std::move(state)), selector(selector ? selector : std::make_shared<PriorityQueueSelector>()),
          brancher(brancher ? brancher : std::make_shared<HeuristicBrancher>())
    {
#ifdef TG_PROFILE
        search_start_time = std::chrono::steady_clock::now();
        next_propagator_report = search_start_time + std::chrono::seconds(2);
#endif
    }

    void addPropagator(std::unique_ptr<Propagator> prop)
    {
        propagators.push_back(std::move(prop));
#ifdef TG_PROFILE
        propagator_timings.emplace_back();
#endif
    }

    bool restoreNode(const std::shared_ptr<SearchNode> &target_node)
    {
        if (current_node_id == target_node->id)
            return true;

#ifdef TG_PROFILE
        search_timing.restore_node_calls++;
        auto lca_start = std::chrono::steady_clock::now();
#endif
        // Path from current_node up to root
        restore_current_path.clear();
        uint32_t curr = current_node_id;
        while (curr != UINT32_MAX)
        {
            restore_current_path.push_back(curr);
            curr = all_nodes[curr]->parent_id;
        }

        // Path from target_node up to root
        restore_target_path.clear();
        uint32_t tgt = target_node->id;
        while (tgt != UINT32_MAX)
        {
            restore_target_path.push_back(tgt);
            tgt = all_nodes[tgt]->parent_id;
        }

        // Find Lowest Common Ancestor (LCA)
        int i = static_cast<int>(restore_current_path.size()) - 1;
        int j = static_cast<int>(restore_target_path.size()) - 1;
        uint32_t lca = UINT32_MAX;
        while (i >= 0 && j >= 0 && restore_current_path[i] == restore_target_path[j])
        {
            lca = restore_current_path[i];
            i--;
            j--;
        }

        // Backtrack to LCA's trail marker
        size_t lca_marker = (lca != UINT32_MAX) ? all_nodes[lca]->trail_marker : 0;
        state.backtrackTo(lca_marker);
#ifdef TG_PROFILE
        search_timing.restore_node_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - lca_start).count());
#endif

        // Play decisions and propagate from child of LCA down to target_node
        for (int k = j; k >= 0; --k)
        {
            uint32_t nid = restore_target_path[k];
#ifdef TG_PROFILE
            auto set_start = std::chrono::steady_clock::now();
#endif
            state.setDomain(all_nodes[nid]->delta.first, all_nodes[nid]->delta.second);
#ifdef TG_PROFILE
            search_timing.restore_node_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - set_start).count());
#endif
            float lb = 0.0f;
            std::string conflict_reason;
            if (!runPropagators(lb, all_nodes[nid]->delta.first, &conflict_reason))
            {
#ifdef TG_PROFILE
                auto bt_start = std::chrono::steady_clock::now();
#endif
                LOG(DEBUG) << "[SearchEngine] Pruned hyperbox node " << nid
                           << " while restoring: " << conflict_reason;
                current_node_id = (k < static_cast<int>(restore_target_path.size()) - 1) ? restore_target_path[k + 1] : lca;
                state.backtrackTo(current_node_id == UINT32_MAX ? 0 : all_nodes[current_node_id]->trail_marker);
#ifdef TG_PROFILE
                search_timing.restore_node_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - bt_start).count());
#endif
                return false;
            }
            all_nodes[nid]->lower_bound = lb;
            all_nodes[nid]->trail_marker = state.getTrailMarker();
        }

        current_node_id = target_node->id;
        return true;
    }

    bool runPropagators(float &out_lower_bound, VarId changed, std::string *out_conflict_reason = nullptr)
    {
        if (out_conflict_reason)
            out_conflict_reason->clear();

#ifdef TG_PROFILE
        auto run_prop_start = std::chrono::steady_clock::now();
        uint64_t prop_and_lb_ns_in_call = 0;
        auto record_prop_overhead = [&]() {
            uint64_t total_call_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - run_prop_start).count());
            if (total_call_ns > prop_and_lb_ns_in_call)
                search_timing.run_prop_overhead_ns += (total_call_ns - prop_and_lb_ns_in_call);
        };
#endif

        if (prop_queued_epoch.size() < state.numVars())
            prop_queued_epoch.resize(state.numVars(), 0);

        prop_worklist.clear();
        ++current_prop_epoch;
        if (current_prop_epoch == 0)
        {
            std::fill(prop_queued_epoch.begin(), prop_queued_epoch.end(), 0);
            current_prop_epoch = 1;
        }

        auto enqueue = [&](VarId var_id) {
            if (var_id != kInvalidVarId && prop_queued_epoch[var_id] != current_prop_epoch)
            {
                prop_queued_epoch[var_id] = current_prop_epoch;
                prop_worklist.push_back(var_id);
            }
        };

        enqueue(changed);
        state.consumeDirtyDomains(enqueue);

        size_t worklist_head = 0;
        auto preservePendingWork = [&]() {
            for (size_t i = worklist_head - 1; i < prop_worklist.size(); ++i)
                state.schedulePropagation(prop_worklist[i]);
        };
        while (worklist_head < prop_worklist.size())
        {
            // LOG(DEBUG) << "propagator worklist pos: " << worklist_head << "/" << prop_worklist.size();
            const VarId next_changed = prop_worklist[worklist_head++];
            prop_queued_epoch[next_changed] = 0;

            for (size_t prop_idx = 0; prop_idx < propagators.size(); ++prop_idx)
            {
                auto &prop = propagators[prop_idx];
#ifdef TG_PROFILE
                auto propagate_start = std::chrono::steady_clock::now();
#endif
                bool propagated = prop->propagate(state, next_changed, prop_worklist);
#ifdef TG_PROFILE
                const uint64_t propagate_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - propagate_start)
                        .count());
                auto &timing = propagator_timings[prop_idx];
                timing.propagate_calls++;
                timing.propagate_ns += propagate_ns;
                timing.max_propagate_ns = std::max(timing.max_propagate_ns, propagate_ns);
                if (!propagated)
                    timing.contradictions++;
                propagator_calls_since_report++;
                prop_and_lb_ns_in_call += propagate_ns;
#endif
                if (!propagated)
                {
                    preservePendingWork();
                    if (out_conflict_reason)
                        *out_conflict_reason = prop->name();
#ifdef TG_PROFILE
                    record_prop_overhead();
                    maybeReportPropagatorTimings();
#endif
                    return false;
                }

                if (state.hasEmptyDomain())
                {
                    preservePendingWork();
                    if (out_conflict_reason)
                    {
                        VarId empty_var = state.getEmptyDomainVar();
                        *out_conflict_reason = prop->name() + " emptied ";
                        if (empty_var != kInvalidVarId && empty_var < state.var_infos.size())
                            *out_conflict_reason += state.var_infos[empty_var].name;
                        else
                            *out_conflict_reason += "a domain";
                    }
#ifdef TG_PROFILE
                    record_prop_overhead();
                    maybeReportPropagatorTimings();
#endif
                    return false;
                }
                state.consumeDirtyDomains(enqueue);
            }
        }

        out_lower_bound = 0.0f;
        for (size_t prop_idx = 0; prop_idx < propagators.size(); ++prop_idx)
        {
#ifdef TG_PROFILE
            auto lower_bound_start = std::chrono::steady_clock::now();
#endif
            out_lower_bound = std::max(out_lower_bound, propagators[prop_idx]->computeLowerBound(state));
#ifdef TG_PROFILE
            const uint64_t lower_bound_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - lower_bound_start)
                    .count());
            auto &timing = propagator_timings[prop_idx];
            timing.lower_bound_calls++;
            timing.lower_bound_ns += lower_bound_ns;
            timing.max_lower_bound_ns = std::max(timing.max_lower_bound_ns, lower_bound_ns);
            prop_and_lb_ns_in_call += lower_bound_ns;
#endif
        }
#ifdef TG_PROFILE
        record_prop_overhead();
        maybeReportPropagatorTimings();
#endif
        // An incumbent can improve without changing a domain (for example
        // after restoring an already-propagated node).
        if (out_lower_bound >= state.best_cost)
        {
            if (out_conflict_reason)
                *out_conflict_reason = "CostLowerBoundPropagator";
            return false;
        }
        return true;
    }

  public:
    float evaluateMakespan(const SearchState &st, uint32_t b) const
    {
        // Simulate the selected dispatch order, waiting for both data dependencies
        // and the engines required by each operation.
        std::unordered_map<Engine, float> engine_finish;
        std::unordered_map<EClassId, float> eclass_finish;
        std::vector<std::pair<int32_t, EClassId>> sorted_ops;

        const auto &cids = (b < st.reachable_cids.size() && !st.reachable_cids[b].empty())
                               ? st.reachable_cids[b]
                               : std::vector<EClassId>{};
        for (EClassId cid : cids)
        {
            auto it = st.selected_vars[b].find(cid);
            if (it == st.selected_vars[b].end())
                continue;
            VarId sel_v = it->second;
            if (st.domains[sel_v].isFixed() && st.domains[sel_v].fixedValue() > 0)
            {
                uint32_t en_idx = static_cast<uint32_t>(st.domains[sel_v].fixedValue() - 1);
                VarId st_v = st.start_vars[b].at(cid)[en_idx];
                int32_t st_val = st.domains[st_v].fixedValue();
                sorted_ops.push_back({st_val, cid});
            }
        }
        std::sort(sorted_ops.begin(), sorted_ops.end());

        for (const auto &item : sorted_ops)
        {
            EClassId cid = item.second;
            uint32_t en_idx = static_cast<uint32_t>(st.domains[st.selected_vars[b].at(cid)].fixedValue() - 1);
            const EClass &cls = st.bucket_egraphs[b].getEClass(cid);
            ENodeId en_id = cls.enodes[en_idx];
            const ENode &enode = st.bucket_egraphs[b].getENode(en_id);
            float cost = (en_id.value < st.bucket_enode_infos[b].size()) ? st.bucket_enode_infos[b][en_id.value].cost : 0.0f;
            if (cost == TGConstants::INF)
                cost = 1.0f;

            if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
            {
                eclass_finish[cid] = 0.0f;
                continue;
            }

            const bool is_view = en_id.value < st.bucket_enode_infos[b].size() &&
                                 st.bucket_enode_infos[b][en_id.value].is_view;
            const float duration = is_view ? 0.0f : cost;

            float children_finish = 0.0f;
            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = st.bucket_egraphs[b].findConst(child);
                auto finish_it = eclass_finish.find(canon_child);
                if (finish_it != eclass_finish.end())
                    children_finish = std::max(children_finish, finish_it->second);
            }

            float engine_free = 0.0f;
            const std::vector<Engine> &engines = enode.getEngines();
            if (!is_view && engines.empty())
            {
                // Compiled operations without an explicit engine run on CPU.
                auto finish_it = engine_finish.find(Engine{0, EngineType::CPU});
                if (finish_it != engine_finish.end())
                    engine_free = std::max(engine_free, finish_it->second);
            }
            else if (!is_view)
            {
                for (const Engine &eng : engines)
                {
                    auto finish_it = engine_finish.find(eng);
                    if (finish_it != engine_finish.end())
                        engine_free = std::max(engine_free, finish_it->second);
                }
            }

            const float start_time = std::max(children_finish, engine_free);
            const float finish_time = start_time + duration;
            eclass_finish[cid] = finish_time;

            if (!is_view)
            {
                if (engines.empty())
                    engine_finish[Engine{0, EngineType::CPU}] = finish_time;
                else
                {
                    for (const Engine &eng : engines)
                        engine_finish[eng] = finish_time;
                }
            }
        }

        float max_finish = 0.0f;
        for (const auto &pair : engine_finish)
            max_finish = std::max(max_finish, pair.second);
        return max_finish;
    }

    std::vector<ExtractionResult> extractSolution(const SearchState &st) const
    {
        std::vector<ExtractionResult> results(st.buckets.size());
        uint32_t buf_counter = 1;

        for (uint32_t b = 0; b < st.buckets.size(); ++b)
        {
            ExtractionResult &res = results[b];
            std::vector<std::pair<int32_t, EClassId>> sorted_ops;

            const auto &cids = (b < st.reachable_cids.size() && !st.reachable_cids[b].empty())
                                   ? st.reachable_cids[b]
                                   : std::vector<EClassId>{};
            for (EClassId cid : cids)
            {
                auto it = st.selected_vars[b].find(cid);
                if (it == st.selected_vars[b].end())
                    continue;
                VarId sel_v = it->second;
                if (st.domains[sel_v].isFixed() && st.domains[sel_v].fixedValue() > 0)
                {
                    uint32_t en_idx = static_cast<uint32_t>(st.domains[sel_v].fixedValue() - 1);
                    res.selection_map[cid] = en_idx;
                    VarId st_v = st.start_vars[b].at(cid)[en_idx];
                    int32_t st_val = st.domains[st_v].fixedValue();
                    sorted_ops.push_back({st_val, cid});
                }
            }
            std::sort(sorted_ops.begin(), sorted_ops.end());
            for (const auto &item : sorted_ops)
                res.order.push_back(item.second);

            // 1. Add preallocated buffers
            std::unordered_map<BaseEClassId, BufferId> base_to_prealloc;
            for (const auto &pair : st.preallocated_buffers)
            {
                res.buffers.push_back(pair.second);
                base_to_prealloc[pair.first] = pair.second.id;
                buf_counter = std::max(buf_counter, pair.second.id.value + 1);
            }

            // Helper to get or allocate buffer for a root non-view eclass
            auto get_or_allocate_buffer = [&](EClassId target_cid) -> BufferId {
                auto it = res.eclass_to_buf.find(target_cid);
                if (it != res.eclass_to_buf.end())
                    return it->second;

                const EClass &cls = st.bucket_egraphs[b].getEClass(target_cid);
                if (cls.base_eclass_id != BaseEClassId{} && base_to_prealloc.count(cls.base_eclass_id))
                {
                    BufferId bid = base_to_prealloc.at(cls.base_eclass_id);
                    res.eclass_to_buf[target_cid] = bid;
                    return bid;
                }

                if (cls.mem_space.type == HandleType::STORAGE)
                {
                    ParallelBuffer pb;
                    pb.id = BufferId{buf_counter++};
                    pb.mem_space = cls.mem_space;
                    pb.size = getSizeBytes(cls.shape, cls.dtype);
                    pb.offset = -1;
                    res.buffers.push_back(pb);
                    res.eclass_to_buf[target_cid] = pb.id;
                    return pb.id;
                }

                int64_t byte_offset = 0;
                auto off_it = st.offset_vars[b].find(target_cid);
                if (off_it != st.offset_vars[b].end())
                {
                    VarId off_v = off_it->second;
                    int32_t page_off = st.domains[off_v].fixedValue();
                    byte_offset = static_cast<int64_t>(page_off) * st.getPageAlignment(cls.mem_space);
                }

                ParallelBuffer pb;
                pb.id = BufferId{buf_counter++};
                pb.mem_space = cls.mem_space;
                pb.size = getSizeBytes(cls.shape, cls.dtype);
                pb.offset = byte_offset;
                res.buffers.push_back(pb);
                res.eclass_to_buf[target_cid] = pb.id;
                return pb.id;
            };

            // 2. Map buffers for all ordered eclasses
            for (EClassId cid : res.order)
            {
                EClassId base_cid = resolve_view_alias(cid, st.bucket_egraphs[b], res.selection_map, st.bucket_enode_infos[b]);
                BufferId bid = get_or_allocate_buffer(base_cid);
                res.eclass_to_buf[cid] = bid;
            }

            for (const auto &pair : res.selection_map)
            {
                EClassId cid = pair.first;
                uint32_t en_idx = pair.second;
                ENodeId en_id = st.bucket_egraphs[b].getEClass(cid).enodes[en_idx];
                float en_cost = (en_id.value < st.bucket_enode_infos[b].size())
                                    ? st.bucket_enode_infos[b][en_id.value].cost
                                    : 0.0f;
                res.eclass_to_cost[cid] = en_cost;
            }

            res.cost = evaluateMakespan(st, b);
        }
        return results;
    }

    bool solve(float timeout_seconds = 0.0f)
    {
        TimeoutChecker timer(timeout_seconds);
#ifdef TG_PROFILE
        search_start_time = std::chrono::steady_clock::now();
        next_propagator_report = search_start_time + std::chrono::seconds(2);
#endif

        LOG(DEBUG) << "[SearchEngine] Starting solve: num_vars=" << state.numVars()
                   << ", candidates=" << state.candidates.size()
                   << ", buckets=" << state.buckets.size()
                   << ", timeout=" << timeout_seconds << "s";

        // 1. Root node
        auto root_node = std::make_shared<SearchNode>(0, UINT32_MAX, std::make_pair(kInvalidVarId, Domain{}), 0.0f,
                                                      0.0f, 0);
        all_nodes.push_back(root_node);
        current_node_id = 0;

        if (incumbent_best_cost < TGConstants::INF)
        {
            selector->setIncumbent(incumbent_best_cost);
        }

        root_node->priority = root_node->lower_bound;
        root_node->trail_marker = state.getTrailMarker();
#ifdef TG_PROFILE
        {
            auto push_start = std::chrono::steady_clock::now();
            selector->push(root_node);
            search_timing.queue_push_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - push_start).count());
            search_timing.queue_push_calls++;
        }
#else
        selector->push(root_node);
#endif

        uint32_t iterations = 0;
        while (!selector->empty())
        {
            if (timer.is_expired() && incumbent_best_cost < TGConstants::INF)
            {
                LOG(INFO) << "[SearchEngine] Timeout reached with feasible incumbent best cost: "
                          << incumbent_best_cost << " at iteration " << iterations;
                break;
            }

            if (timeout_seconds <= 0.0f && incumbent_best_cost < TGConstants::INF)
            {
                LOG(INFO) << "[SearchEngine] Greedy solution found with cost: "
                          << incumbent_best_cost << " at iteration " << iterations
                          << " (min compile time = 0); completing search.";
                break;
            }

#ifdef TG_PROFILE
            auto pop_start = std::chrono::steady_clock::now();
            auto node = selector->pop();
            search_timing.queue_pop_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - pop_start).count());
            search_timing.queue_pop_calls++;
#else
            auto node = selector->pop();
#endif
            if (!node || node->lower_bound >= incumbent_best_cost)
                continue;

            iterations++;
            if (iterations < 50 || iterations % 50 == 0)
            {
                LOG(DEBUG) << "[SearchEngine] Iter " << iterations
                           << " | Queue: " << selector->size()
                           << " | Depth: " << node->depth
                           << " | Node LB: " << node->lower_bound
                           << " | Best: " << (incumbent_best_cost < TGConstants::INF ? std::to_string(incumbent_best_cost) : "inf");
            }

            if (!restoreNode(node))
            {
                if (iterations <= 20 || iterations % 50 == 0)
                {
                    LOG(DEBUG) << "[SearchEngine] Iter " << iterations << ": backtrack due to propagation conflict";
                }
                continue;
            }

            if (node->lower_bound >= incumbent_best_cost)
                continue;

            // Check if branching is needed or all variables are fixed
            BranchDecision decision;
#ifdef TG_PROFILE
            auto branch_start = std::chrono::steady_clock::now();
            bool can_branch = brancher->chooseBranch(state, decision);
            search_timing.choose_branch_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - branch_start).count());
#else
            bool can_branch = brancher->chooseBranch(state, decision);
#endif

            if (!can_branch)
            {
                // Leaf reached: evaluate complete plan
#ifdef TG_PROFILE
                auto leaf_start = std::chrono::steady_clock::now();
#endif
                float total_cost = 0.0f;
                for (uint32_t b = 0; b < state.buckets.size(); ++b)
                {
                    float w = (b < state.bucket_weights.size()) ? state.bucket_weights[b] : 1.0f;
                    total_cost += w * evaluateMakespan(state, b);
                }
#ifdef TG_PROFILE
                search_timing.leaf_eval_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - leaf_start).count());
                search_timing.leaf_eval_calls++;
#endif

                LOG(DEBUG) << "[SearchEngine] Iter " << iterations << ": Leaf reached at depth " << node->depth
                           << ", evaluated total cost=" << total_cost;

                if (total_cost < incumbent_best_cost)
                {
                    incumbent_best_cost = total_cost;
                    selector->setIncumbent(incumbent_best_cost);
                    incumbent_extractions = extractSolution(state);

                    incumbent_cached_nodes.clear();
                    for (const auto &pair : state.cached_vars)
                    {
                        if (state.domains[pair.second].isFixed() && state.domains[pair.second].fixedValue() == 1)
                        {
                            incumbent_cached_nodes.insert(pair.first);
                        }
                    }

                    LOG(INFO) << "[SearchEngine] New best cost: " << incumbent_best_cost
                              << " at iteration " << iterations;
                    state.best_cost = incumbent_best_cost;
                }
                continue;
            }

            if (iterations <= 25 || iterations % 50 == 0)
            {
                VarId branch_var = decision.left_delta.first;
                LOG(DEBUG) << "[SearchEngine] Iter " << iterations << ": branching on var " << branch_var
                            << " (" << state.var_infos[branch_var].name << ") [dom: " << state.domains[branch_var].toString()
                            << "] -> Left: " << decision.left_delta.second.toString()
                            << ", Right: " << decision.right_delta.second.toString();
            }

            // 1. Create Right Child (lazy alternative, pushed to queue without upfront propagation)
            uint32_t right_id = static_cast<uint32_t>(all_nodes.size());
            float right_lb = node->lower_bound;
            float right_prio = right_lb + 1.0f;
            auto right_node = std::make_shared<SearchNode>(
                right_id, node->id, decision.right_delta, right_lb,
                right_prio, node->depth + 1);
            all_nodes.push_back(right_node);
#ifdef TG_PROFILE
            {
                auto push_start = std::chrono::steady_clock::now();
                selector->push(right_node);
                search_timing.queue_push_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - push_start).count());
                search_timing.queue_push_calls++;
            }
#else
            selector->push(right_node);
#endif

            // 2. Create Left Child (dive immediately in place!)
            // TODO: should it go back to the selector to choose once left/right are added instead of always choosing left? 
            state.setDomain(decision.left_delta.first, decision.left_delta.second);
            float left_lb = 0.0f;
            std::string left_conflict;
            bool left_ok = runPropagators(left_lb, decision.left_delta.first, &left_conflict);
            if (left_ok && left_lb < incumbent_best_cost)
            {
                uint32_t left_id = static_cast<uint32_t>(all_nodes.size());
                float left_prio = left_lb - (node->depth + 1) * 0.01f; // Dive bias
                auto left_node = std::make_shared<SearchNode>(
                    left_id, node->id, decision.left_delta, left_lb,
                    left_prio, node->depth + 1, state.getTrailMarker());
                all_nodes.push_back(left_node);
                current_node_id = left_id; // Current state stays at left_node!
#ifdef TG_PROFILE
                {
                    auto push_start = std::chrono::steady_clock::now();
                    selector->push(left_node);
                    search_timing.queue_push_ns += static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - push_start).count());
                    search_timing.queue_push_calls++;
                }
#else
                selector->push(left_node);
#endif
            }
            else
            {
                if (!left_ok)
                {
                    LOG(DEBUG) << "[SearchEngine] Pruned left hyperbox from node " << node->id
                               << ": " << left_conflict;
                }
                else
                {
                    LOG(DEBUG) << "[SearchEngine] Pruned left hyperbox from node " << node->id
                               << " by lower bound " << left_lb;
                }
                // Left branch failed, backtrack in place to parent node
                state.backtrackTo(node->trail_marker);
                current_node_id = node->id;
            }
        }

        LOG(INFO) << "[SearchEngine] Search finished after " << iterations << " iterations ("
                  << all_nodes.size() << " total nodes generated). Best cost: "
                  << (incumbent_best_cost < TGConstants::INF ? std::to_string(incumbent_best_cost) : "none");

#ifdef TG_PROFILE
        reportPropagatorTimings(true);
#endif

        return incumbent_best_cost < TGConstants::INF;
    }

  private:
#ifdef TG_PROFILE
    struct PropagatorTiming
    {
        uint64_t propagate_calls = 0;
        uint64_t propagate_ns = 0;
        uint64_t max_propagate_ns = 0;
        uint64_t contradictions = 0;
        uint64_t lower_bound_calls = 0;
        uint64_t lower_bound_ns = 0;
        uint64_t max_lower_bound_ns = 0;
    };

    struct SearchTimingBreakdown
    {
        uint64_t choose_branch_ns = 0;
        uint64_t restore_node_ns = 0;
        uint64_t restore_node_calls = 0;
        uint64_t queue_pop_ns = 0;
        uint64_t queue_pop_calls = 0;
        uint64_t queue_push_ns = 0;
        uint64_t queue_push_calls = 0;
        uint64_t leaf_eval_ns = 0;
        uint64_t leaf_eval_calls = 0;
        uint64_t run_prop_overhead_ns = 0;
    };

    std::vector<PropagatorTiming> propagator_timings;
    SearchTimingBreakdown search_timing;
    uint64_t propagator_calls_since_report = 0;
    std::chrono::steady_clock::time_point next_propagator_report;
    std::chrono::steady_clock::time_point search_start_time;

    void maybeReportPropagatorTimings()
    {
        if (std::chrono::steady_clock::now() < next_propagator_report)
            return;

        reportPropagatorTimings(false);
    }

    void reportPropagatorTimings(bool force)
    {
        if (!force && propagator_calls_since_report == 0)
            return;

        const auto now = std::chrono::steady_clock::now();
        const double search_elapsed_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            now - search_start_time).count();

        uint64_t total_ns = 0;
        for (size_t i = 0; i < propagators.size(); ++i)
        {
            total_ns += propagator_timings[i].propagate_ns + propagator_timings[i].lower_bound_ns;
        }
        const double total_prop_ms = total_ns / 1.0e6;
        const double total_pct_search = (search_elapsed_ms > 0.0)
                                            ? (total_prop_ms / search_elapsed_ms) * 100.0
                                            : 0.0;
        const double other_search_ms = std::max(0.0, search_elapsed_ms - total_prop_ms);
        const double other_pct_search = (search_elapsed_ms > 0.0)
                                            ? (other_search_ms / search_elapsed_ms) * 100.0
                                            : 0.0;

        std::cout << "\n[SearchEngine] Propagator timing report" << (force ? " (final)" : "") << "\n";
        std::cout << std::left << std::setw(32) << "Propagator" << std::right
                  << std::setw(12) << "Prop (ms)"
                  << std::setw(12) << "Prop calls"
                  << std::setw(14) << "Prop avg (us)"
                  << std::setw(14) << "Prop max (ms)"
                  << std::setw(15) << "Contradictions"
                  << std::setw(10) << "LB (ms)"
                  << std::setw(10) << "LB calls"
                  << std::setw(12) << "LB max (ms)"
                  << std::setw(12) << "Total (ms)"
                  << std::setw(10) << "% Search"
                  << std::setw(10) << "% Prop"
                  << "\n";
        std::cout << std::string(151, '-') << "\n";

        for (size_t i = 0; i < propagators.size(); ++i)
        {
            const auto &timing = propagator_timings[i];
            const double propagate_ms = timing.propagate_ns / 1.0e6;
            const double lower_bound_ms = timing.lower_bound_ns / 1.0e6;
            const double total_ms = propagate_ms + lower_bound_ms;
            const double propagate_avg_us = timing.propagate_calls == 0
                                                ? 0.0
                                                : static_cast<double>(timing.propagate_ns) /
                                                      static_cast<double>(timing.propagate_calls) / 1.0e3;
            const double pct_search = (search_elapsed_ms > 0.0)
                                          ? (total_ms / search_elapsed_ms) * 100.0
                                          : 0.0;
            const double pct_prop = (total_prop_ms > 0.0)
                                        ? (total_ms / total_prop_ms) * 100.0
                                        : 0.0;
            std::cout << std::left << std::setw(32) << propagators[i]->name().substr(0, 31) << std::right
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << propagate_ms
                      << std::setw(12) << timing.propagate_calls
                      << std::setw(14) << propagate_avg_us
                      << std::setw(14) << timing.max_propagate_ns / 1.0e6
                      << std::setw(15) << timing.contradictions
                      << std::setw(10) << lower_bound_ms
                      << std::setw(10) << timing.lower_bound_calls
                      << std::setw(12) << timing.max_lower_bound_ns / 1.0e6
                      << std::setw(12) << total_ms
                      << std::setw(9) << pct_search << "%"
                      << std::setw(9) << pct_prop << "%"
                      << "\n";
        }
        std::cout << std::string(151, '-') << "\n";
        std::cout << std::fixed << std::setprecision(2)
                  << "Total Search Elapsed Time:            " << std::setw(10) << search_elapsed_ms << " ms\n"
                  << "Cumulative Propagator Time:          " << std::setw(10) << total_prop_ms << " ms ("
                  << total_pct_search << "% of total search time)\n"
                  << "Rest of Search (branching, queue...): " << std::setw(10) << other_search_ms << " ms ("
                  << other_pct_search << "% of total search time)\n";

        const double choose_branch_ms = search_timing.choose_branch_ns / 1.0e6;
        const double restore_node_ms = search_timing.restore_node_ns / 1.0e6;
        const double run_prop_overhead_ms = search_timing.run_prop_overhead_ns / 1.0e6;
        const double queue_pop_ms = search_timing.queue_pop_ns / 1.0e6;
        const double queue_push_ms = search_timing.queue_push_ns / 1.0e6;
        const double leaf_eval_ms = search_timing.leaf_eval_ns / 1.0e6;
        const double queue_total_ms = queue_pop_ms + queue_push_ms;
        const double accounted_other_ms = choose_branch_ms + restore_node_ms + run_prop_overhead_ms + queue_total_ms + leaf_eval_ms;
        const double remaining_other_ms = std::max(0.0, other_search_ms - accounted_other_ms);

        std::cout << "\n[SearchEngine] Non-Propagator Timing Breakdown:\n";
        std::cout << std::left << std::setw(32) << "Component" << std::right
                  << std::setw(12) << "Time (ms)"
                  << std::setw(12) << "Calls"
                  << std::setw(14) << "% Search"
                  << std::setw(14) << "% Rest"
                  << "\n";
        std::cout << std::string(84, '-') << "\n";

        auto print_component = [&](const std::string &name, double ms, uint64_t calls) {
            const double pct_search = (search_elapsed_ms > 0.0) ? (ms / search_elapsed_ms) * 100.0 : 0.0;
            const double pct_rest = (other_search_ms > 0.0) ? (ms / other_search_ms) * 100.0 : 0.0;
            std::cout << std::left << std::setw(32) << name << std::right
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << ms
                      << std::setw(12) << calls
                      << std::setw(13) << pct_search << "%"
                      << std::setw(13) << pct_rest << "%"
                      << "\n";
        };

        print_component("chooseBranch", choose_branch_ms, 0);
        print_component("restoreNode", restore_node_ms, search_timing.restore_node_calls);
        print_component("runPropagators Overhead", run_prop_overhead_ms, 0);
        print_component("Queue Pop", queue_pop_ms, search_timing.queue_pop_calls);
        print_component("Queue Push", queue_push_ms, search_timing.queue_push_calls);
        print_component("Leaf Evaluation", leaf_eval_ms, search_timing.leaf_eval_calls);
        print_component("Other / Engine Overhead", remaining_other_ms, 0);
        std::cout << std::string(84, '-') << "\n";

        if (brancher)
            brancher->reportBrancherTiming();

        std::cout << std::flush;

        propagator_calls_since_report = 0;
        next_propagator_report = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    }
#endif
};

} // namespace plan
