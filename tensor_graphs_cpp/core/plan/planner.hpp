// File: tensor_graphs_cpp/core/plan/planner.hpp
// TODO: Enhanced NaN protection during DP passes to avoid sorting UB

#pragma once
#include <algorithm>
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
#include "core/plan/extractor.hpp"
#include "core/plan/mem.hpp"
#include "core/plan/pruning.hpp"
#include "core/plan/search_delegate.hpp"
#include "core/rewrite.hpp"
#include "core/shape_propagator.hpp"
#include "core/timer.hpp"
#include "core/types.hpp"

struct ExtractionResult
{
    std::unordered_map<EClassId, uint32_t> selection_map;
    std::vector<EClassId> order;
    std::vector<ParallelBuffer> buffers;
    std::unordered_map<EClassId, BufferId> eclass_to_buf;
    float cost;
    std::unordered_map<EClassId, float> eclass_to_cost;
};

#include "core/plan/unified_search.hpp"

inline ExtractionResult UnifiedSearchPlanner::solve(float minCompileSeconds, bool onlyDive, bool stopOnFirstValid)
{
    float best_cost = TGConstants::INF;
    std::unordered_map<EClassId, uint32_t> best_selection_map;
    std::vector<EClassId> best_order;
    std::vector<ParallelBuffer> best_buffers;
    std::unordered_map<EClassId, BufferId> best_eclass_to_buf;

    auto start_time = std::chrono::steady_clock::now();
    TimeoutChecker timeout_checker(minCompileSeconds);
    auto is_time_expired = [&]() {
        if (minCompileSeconds <= 0.0f || best_cost == TGConstants::INF)
            return false;
        auto elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
        return elapsed >= minCompileSeconds;
    };

    auto extract_bools = prune::extract_enabled_states<AllExtractRuleTypes>("extract", settings);
    auto dispatch_bools = prune::extract_enabled_states<AllDispatchRuleTypes>("dispatch", settings);
    auto bufferize_bools = prune::extract_enabled_states<AllBufferizeRuleTypes>("bufferize", settings);
    (void)extract_bools;
    (void)dispatch_bools;
    (void)bufferize_bools;

    SearchState state(egraph, rootEClassId, enodeInfos, eclassToLogical, reducedCaps, settings, &best_cost,
                      cachedEClasses, cleanEClasses);
    // The agenda is deliberately unbounded: priority controls exploration
    // order, but no branch may be dropped if the search is to remain complete.
    // Agenda nodes hold shared persistent path prefixes, so this does not copy
    // a full decision vector for every sibling branch.
    PriorityQueue agenda;
    std::vector<AgendaNode> depth_first_agenda;
    uint64_t sequence_id = 0;
    uint64_t total_dives = 0;
    size_t extraction_dead_ends = 0;
    size_t dispatch_dead_ends = 0;
    size_t bufferize_dead_ends = 0;
    std::string last_dead_end;
    uint32_t last_extraction_class = UINT32_MAX;
    std::unordered_map<std::string, size_t> last_extraction_prunes;
    const bool stop_after_first = stopOnFirstValid && minCompileSeconds <= 0.0f;

    struct CandidateMove
    {
        Decision decision;
        float priority = TGConstants::INF;
        float lower_bound = TGConstants::INF;
    };

    auto finite_cost = [](float cost) {
        return cost < TGConstants::INF && std::isfinite(cost);
    };

    auto heuristic_cost = [&](ENodeId node_id) {
        const ENodeInfo &info = enodeInfos[node_id.value];
        if (finite_cost(info.dp_cp_cost))
            return info.dp_cp_cost;
        if (finite_cost(info.dp_cost))
            return info.dp_cost;
        return finite_cost(info.cost) ? info.cost : 0.0f;
    };

    auto get_sorted_candidates = [&]() -> std::vector<CandidateMove> {
        std::vector<CandidateMove> candidates;
        state.prepareNextPhase();

        if (!state.to_process.empty())
        {
            std::vector<EClassId> frontier;
            for (EClassId eclass : state.to_process)
            {
                if (!state.selection_map.count(eclass))
                {
                    frontier.push_back(eclass);
                }
            }
            std::vector<EClassId> frontier_order = frontier;
            if (delegate && frontier.size() > 1)
            {
                frontier_order.clear();
                std::vector<ActionFeatureFrontier> features;
                features.reserve(frontier.size());
                for (EClassId eclass : frontier)
                {
                    const EClass &class_info = egraph.getEClass(eclass);
                    ActionFeatureFrontier feature;
                    feature.eclass_id = eclass.value;
                    feature.num_enodes = static_cast<uint32_t>(class_info.enodes.size());
                    feature.size = getSizeBytes(class_info.shape, class_info.dtype);
                    feature.dtype = class_info.dtype;
                    feature.mem_space = class_info.mem_space;
                    feature.mem_cap = reducedCaps.count(class_info.mem_space) ? reducedCaps.at(class_info.mem_space) : 0;
                    feature.min_dp_cp_cost = TGConstants::INF;
                    feature.min_dp_cost = TGConstants::INF;
                    feature.min_dp_mem = TGConstants::INF;
                    for (ENodeId node_id : class_info.enodes)
                    {
                        feature.min_dp_cp_cost = std::min(feature.min_dp_cp_cost, enodeInfos[node_id.value].dp_cp_cost);
                        feature.min_dp_cost = std::min(feature.min_dp_cost, enodeInfos[node_id.value].dp_cost);
                        feature.min_dp_mem = std::min(feature.min_dp_mem, enodeInfos[node_id.value].dp_mem);
                    }
                    if (!finite_cost(feature.min_dp_cp_cost))
                        feature.min_dp_cp_cost = 0.0f;
                    if (!finite_cost(feature.min_dp_cost))
                        feature.min_dp_cost = 0.0f;
                    if (!finite_cost(feature.min_dp_mem))
                        feature.min_dp_mem = 0.0f;
                    features.push_back(feature);
                }
                std::vector<uint32_t> relative_order = delegate->order_frontier(features);
                std::vector<uint8_t> seen(frontier.size(), 0);
                for (uint32_t index : relative_order)
                {
                    if (index < frontier.size() && !seen[index])
                    {
                        frontier_order.push_back(frontier[index]);
                        seen[index] = 1;
                    }
                }
                for (size_t index = 0; index < frontier.size(); ++index)
                    if (!seen[index])
                        frontier_order.push_back(frontier[index]);
            }
            std::unordered_map<std::string, size_t> extraction_prunes;
            for (size_t fo_idx = 0; fo_idx < frontier_order.size(); ++fo_idx)
            {
                EClassId current = frontier_order[fo_idx];
                last_extraction_class = current.value;
                const auto &enodes = egraph.getEClass(current).enodes;
                for (uint32_t selection = 0; selection < enodes.size(); ++selection)
                {
                    ExtractContext ctx{egraph, enodeInfos, state.selection_map, state.extract_path, current,
                                       selection, &state.to_process, &best_cost, &reducedCaps, cachedEClasses,
                                       cleanEClasses};
                    if (state.extract_rules.is_pruned(enodes[selection], selection, ctx))
                    {
                        ++extraction_prunes[state.extract_rules.first_pruning_rule(enodes[selection], selection, ctx)];
                        continue;
                    }

                    float rule_lb = state.extract_rules.compute_lower_bound(enodes[selection], selection, ctx);
                    float node_cost = enodeInfos[enodes[selection].value].cost;
                    float op_lb = state.selected_operation_lower_bound;
                    if (finite_cost(node_cost))
                        op_lb = std::max(op_lb, node_cost);
                    float cand_lb = std::max(op_lb, rule_lb);

                    if (cand_lb >= best_cost)
                        continue;

                    float h = heuristic_cost(enodes[selection]);
                    float priority = cand_lb + 0.001f * std::max(0.0f, h);
                    candidates.push_back({{DecisionPhase::EXTRACT, current.value, selection}, priority, cand_lb});
                }
                if (!candidates.empty())
                    break;
            }
            last_extraction_prunes = std::move(extraction_prunes);
        }
        else if (state.ordered.size() < state.selection_map.size())
        {
            uint32_t position = static_cast<uint32_t>(state.ordered.size());
            for (uint32_t index = 0; index < state.current_ready.size(); ++index)
            {
                EClassId node = state.current_ready[index];
                DispatchContext ctx{egraph, state.selection_map, enodeInfos, state.ordered, state.current_ready,
                                     position, reducedCaps, &best_cost};
                if (state.dispatch_rules.is_pruned(node, index, ctx))
                    continue;

                float rule_lb = state.dispatch_rules.compute_lower_bound(node, index, ctx);
                float cand_lb = std::max(state.selected_operation_lower_bound, rule_lb);

                if (cand_lb >= best_cost)
                    continue;

                ENodeId node_id = egraph.getEClass(node).enodes[state.selection_map.at(node)];
                float priority = cand_lb + 0.001f * heuristic_cost(node_id);
                candidates.push_back({{DecisionPhase::DISPATCH, position, node.value}, priority, cand_lb});
            }
        }
        else if (state.k_buf < state.ordered.size())
        {
            uint32_t position = state.k_buf;
            const auto &choices = state.valid_inplace_choices[position];
            EClassId eclass = state.ordered[position];
            for (size_t choice_index = 0; choice_index < choices.size(); ++choice_index)
            {
                int choice = choices[choice_index];
                BufferizeContext ctx{state.ordered, egraph, state.selection_map, enodeInfos, state.birth_times,
                                     state.death_times, state.inplace_alias, choices, position, reducedCaps,
                                     &best_cost};
                if (state.bufferize_rules.is_pruned(choice, choice_index, ctx))
                    continue;

                float rule_lb = state.bufferize_rules.compute_lower_bound(choice, choice_index, ctx);
                float cand_lb = std::max(state.selected_operation_lower_bound, rule_lb);

                if (cand_lb >= best_cost)
                    continue;

                float priority = cand_lb + (choice == -1 ? 0.01f : 0.0f);
                candidates.push_back({{DecisionPhase::BUFFERIZE, position, choice}, priority, cand_lb});
            }
        }

        std::sort(candidates.begin(), candidates.end(), [](const CandidateMove &a, const CandidateMove &b) {
            if (std::abs(a.priority - b.priority) > 1e-5f)
                return a.priority < b.priority;
            if (std::abs(a.lower_bound - b.lower_bound) > 1e-5f)
                return a.lower_bound < b.lower_bound;
            if (a.decision.phase != b.decision.phase)
                return a.decision.phase < b.decision.phase;
            return a.decision.choice < b.decision.choice;
        });

        return candidates;
    };

    auto remember_plan = [&](const std::unordered_map<EClassId, uint32_t> &selection_map,
                             const std::vector<EClassId> &order, const std::vector<ParallelBuffer> &buffers,
                             const std::unordered_map<EClassId, BufferId> &eclass_to_buf) {
        float cost = get_cost(order, egraph, selection_map, enodeInfos);
        if (cost < best_cost)
        {
            best_cost = cost;
            best_selection_map = selection_map;
            best_order = order;
            best_buffers = buffers;
            best_eclass_to_buf = eclass_to_buf;
            float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
            LOG(INFO) << "[UnifiedSearch] Dive #" << total_dives << " found new best cost " << best_cost
                      << " ms (elapsed: " << elapsed << " s)";
        }
    };

    auto evaluate_leaf = [&]() {
        std::unordered_set<BufferId> preallocated_buf_ids;
        std::unordered_map<BufferId, ParallelBuffer> preallocated_overrides;
        for (EClassId eclass : state.ordered)
        {
            auto logical_it = eclassToLogical.find(eclass);
            auto selection_it = state.selection_map.find(eclass);
            if (logical_it == eclassToLogical.end() || selection_it == state.selection_map.end())
                continue;
            const ENode &node = egraph.getENode(egraph.getEClass(eclass).enodes[selection_it->second]);
            if (node.getOpType() != OpType::INPUT && node.getOpType() != OpType::CACHE)
                continue;
            auto preallocated_it = preallocatedBuffers.find(egraph.getEClass(eclass).base_eclass_id);
            if (preallocated_it == preallocatedBuffers.end())
                continue;
            BufferId buffer_id = state.eclass_to_buf.at(eclass);
            preallocated_buf_ids.insert(buffer_id);
            preallocated_overrides[buffer_id] = preallocated_it->second;
        }

        std::unordered_map<MemSpace, std::vector<ParallelBuffer>> by_memory_space;
        std::vector<ParallelBuffer> current_buffers;
        for (auto buffer : state.unallocated_buffers)
        {
            if (buffer.mem_space.type == HandleType::STORAGE)
            {
                buffer.offset = 0;
                current_buffers.push_back(buffer);
            }
            else if (preallocated_buf_ids.count(buffer.id))
            {
                buffer.offset = preallocated_overrides.at(buffer.id).offset;
                current_buffers.push_back(buffer);
            }
            else
            {
                by_memory_space[buffer.mem_space].push_back(buffer);
            }
        }

        for (auto &[memory_space, buffers] : by_memory_space)
        {
            uint64_t cap = reducedCaps.count(memory_space) ? reducedCaps.at(memory_space)
                                                            : std::numeric_limits<uint64_t>::max();
            std::vector<ParallelBuffer> allocated;
            BufferId overflow;
            if (!malloc_by_time_components(cap, buffers, allocated, overflow, delegate, &settings, &best_cost,
                                           &timeout_checker))
            {
                return false;
            }
            uint64_t reserved = reservedPerMemorySpace.count(memory_space) ? reservedPerMemorySpace.at(memory_space)
                                                                             : 0;
            for (auto &buffer : allocated)
                buffer.offset += static_cast<int64_t>(reserved);
            current_buffers.insert(current_buffers.end(), allocated.begin(), allocated.end());
        }
        remember_plan(state.selection_map, state.ordered, current_buffers, state.eclass_to_buf);
        return true;
    };

    // =========================================================================
    // PHASE 1: INITIAL DIVE / DFS (Find initial incumbent plan)
    // =========================================================================
    bool diving = true;
    total_dives = 1;
    size_t last_logged_dive = 0;
    auto last_log_time = std::chrono::steady_clock::now();

    while (diving && !is_time_expired())
    {
        state.prepareNextPhase();
        if (state.isLeaf())
        {
            bool leaf_valid = evaluate_leaf();
            if (leaf_valid && stop_after_first)
                break;

            if (leaf_valid)
            {
                if (onlyDive)
                {
                    if (depth_first_agenda.empty())
                        break;
                    AgendaNode next = std::move(depth_first_agenda.back());
                    depth_first_agenda.pop_back();
                    state.transition_to(next.path);
                    state.push_decision(next.next_decision);
                    total_dives++;
                    continue;
                }
                else
                {
                    // Established initial incumbent plan! Switch to Priority Diving (Phase 2).
                    diving = false;
                    float elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
                    LOG(INFO) << "[UnifiedSearch] Phase 1 Dive complete (Dive #" << total_dives
                              << "). Established initial incumbent best_cost=" << best_cost << " ms (elapsed: "
                              << elapsed << " s)";
                    for (auto &node : depth_first_agenda)
                    {
                        if (node.next_decision.phase != DecisionPhase::BUFFERIZE && node.lower_bound < best_cost)
                            agenda.push(std::move(node));
                    }
                    depth_first_agenda.clear();
                    break;
                }
            }
            else
            {
                // Leaf failed validation (e.g. malloc overflow).
                // Backtrack to continue searching:
                if (total_dives <= 5 || total_dives % 50 == 0)
                    LOG(INFO) << "[UnifiedSearch] Dive #" << total_dives << " leaf failed evaluate_leaf()! Backtrack stack: " << depth_first_agenda.size();
                if (depth_first_agenda.empty())
                {
                    break;
                }
                AgendaNode next = std::move(depth_first_agenda.back());
                depth_first_agenda.pop_back();
                state.transition_to(next.path);
                state.push_decision(next.next_decision);
                total_dives++;
                continue;
            }
        }

        auto candidates = get_sorted_candidates();
        if (candidates.empty())
        {
            if (!state.to_process.empty())
            {
                ++extraction_dead_ends;
                last_dead_end = "extract class=" + std::to_string(last_extraction_class) +
                                " frontier=" + std::to_string(state.to_process.size()) +
                                " selected=" + std::to_string(state.selection_map.size());
            }
            else if (state.ordered.empty() && !state.selection_map.empty() &&
                     state.extraction_leaf_rejected)
            {
                ++extraction_dead_ends;
                last_dead_end = "extract leaf rejected selected=" +
                                std::to_string(state.selection_map.size());
            }
            else if (state.ordered.size() < state.selection_map.size())
            {
                ++dispatch_dead_ends;
                last_dead_end = "dispatch ready=" + std::to_string(state.current_ready.size()) +
                                " ordered=" + std::to_string(state.ordered.size()) + "/" +
                                std::to_string(state.selection_map.size());
            }
            else
            {
                ++bufferize_dead_ends;
                last_dead_end = "bufferize position=" + std::to_string(state.k_buf) + "/" +
                                std::to_string(state.ordered.size());
            }
            if (total_dives <= 5 || total_dives % 50 == 0)
            {
                std::string prunes_str;
                for (const auto &p : last_extraction_prunes)
                    prunes_str += p.first + "=" + std::to_string(p.second) + " ";
                LOG(INFO) << "[UnifiedSearch] Dive #" << total_dives << " dead end: " << last_dead_end << " | Prunes: " << prunes_str << "| Backtrack stack: " << depth_first_agenda.size();
            }
            // Dead end in dive: backtrack via depth_first_agenda to continue dive
            if (depth_first_agenda.empty())
                break;

            AgendaNode next = std::move(depth_first_agenda.back());
            depth_first_agenda.pop_back();
            state.transition_to(next.path);
            state.push_decision(next.next_decision);
            total_dives++;
            continue;
        }
        else
        {
            for (size_t i = candidates.size(); i > 1; --i)
            {
                const CandidateMove &candidate = candidates[i - 1];
                if (candidate.lower_bound >= best_cost)
                    continue;
                AgendaNode node{state.active_path_ref, candidate.decision, candidate.priority, candidate.lower_bound,
                                ++sequence_id};
                depth_first_agenda.push_back(std::move(node));
            }
            if (candidates.front().lower_bound >= best_cost)
            {
                if (depth_first_agenda.empty())
                    break;
                AgendaNode next = std::move(depth_first_agenda.back());
                depth_first_agenda.pop_back();
                state.transition_to(next.path);
                state.push_decision(next.next_decision);
                total_dives++;
                continue;
            }
            else
            {
                state.push_decision(candidates.front().decision);
            }
        }

        auto now = std::chrono::steady_clock::now();
        float elapsed_since_last_log = std::chrono::duration<float>(now - last_log_time).count();
        if (elapsed_since_last_log >= 5.0f || (total_dives != last_logged_dive && (total_dives == 10 || total_dives == 50 || total_dives == 100 ||
            total_dives == 500 || (total_dives % 500 == 0))))
        {
            last_log_time = now;
            last_logged_dive = total_dives;
            float elapsed_total = std::chrono::duration<float>(now - start_time).count();
            LOG(INFO) << "[UnifiedSearch] Phase 1 Dives: " << total_dives
                      << " | Backtrack stack: " << depth_first_agenda.size()
                      << " | Elapsed: " << elapsed_total << " s";
        }
    }

    // =========================================================================
    // PHASE 2: PRIORITY DIVING SEARCH MODE
    // =========================================================================
    if (!onlyDive && !stop_after_first && best_cost < TGConstants::INF)
    {
        while (!agenda.empty() && !is_time_expired())
        {
            AgendaNode top_node = agenda.top();
            agenda.pop();
            if (top_node.lower_bound >= best_cost)
                continue;

            state.transition_to(top_node.path);
            state.push_decision(top_node.next_decision);
            total_dives++;

            bool dive_active = true;
            while (dive_active && !is_time_expired())
            {
                state.prepareNextPhase();
                if (state.isLeaf())
                {
                    evaluate_leaf();
                    dive_active = false;
                    break;
                }

                auto candidates = get_sorted_candidates();
                if (candidates.empty() || candidates.front().lower_bound >= best_cost)
                {
                    dive_active = false;
                    break;
                }

                if (candidates.front().decision.phase != DecisionPhase::BUFFERIZE)
                {
                    for (size_t i = 1; i < candidates.size(); ++i)
                    {
                        if (candidates[i].lower_bound < best_cost && agenda.size() < 200000)
                        {
                            agenda.push(AgendaNode{state.active_path_ref, candidates[i].decision,
                                                   candidates[i].priority, candidates[i].lower_bound,
                                                   ++sequence_id});
                        }
                    }
                }

                state.push_decision(candidates.front().decision);
            }

            auto now = std::chrono::steady_clock::now();
            float elapsed_since_last_log = std::chrono::duration<float>(now - last_log_time).count();
            if (elapsed_since_last_log >= 5.0f || (total_dives != last_logged_dive && (total_dives == 10 || total_dives == 50 || total_dives == 100 ||
                total_dives == 500 || total_dives == 1000 || (total_dives % 500 == 0))))
            {
                last_log_time = now;
                last_logged_dive = total_dives;
                float elapsed_total = std::chrono::duration<float>(now - start_time).count();
                LOG(INFO) << "[UnifiedSearch] Dives completed: " << total_dives
                          << " | Agenda size: " << agenda.size()
                          << " | Best cost: " << best_cost << " ms"
                          << " (elapsed: " << elapsed_total << " s)";
            }
        }
    }

    float final_elapsed = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
    LOG(INFO) << "[UnifiedSearch] Search finished. Total dives: " << total_dives
              << " | Final best cost: " << best_cost << " ms | Total elapsed: " << final_elapsed << " s";

    if (best_cost == TGConstants::INF)
    {
        LOG(ERROR) << "[UnifiedSearchPlanner] exhausted agenda without a valid leaf; dead ends: extract="
                   << extraction_dead_ends << ", dispatch=" << dispatch_dead_ends
                   << ", bufferize=" << bufferize_dead_ends << "; last=" << last_dead_end;
        if (!last_extraction_prunes.empty())
        {
            std::ostringstream prune_summary;
            bool first = true;
            for (const auto &[rule, count] : last_extraction_prunes)
            {
                if (!first)
                    prune_summary << ", ";
                first = false;
                prune_summary << rule << "=" << count;
            }
            LOG(ERROR) << "[UnifiedSearchPlanner] last extraction prune reasons: " << prune_summary.str();
        }
        Error::throw_err("[UnifiedSearchPlanner] no valid extraction found under given constraints. try running bench");
    }

    std::unordered_map<EClassId, float> best_eclass_to_cost;
    for (const auto &entry : best_selection_map)
    {
        ENodeId node_id = egraph.getEClass(entry.first).enodes[entry.second];
        best_eclass_to_cost[entry.first] = enodeInfos[node_id.value].cost;
    }
    return {best_selection_map, best_order, best_buffers, best_eclass_to_buf, best_cost, best_eclass_to_cost};
}

// =============================================================================
// CacheContext -- view into CacheIterator state at check() time
// =============================================================================
struct CacheCandidate
{
    BaseEClassId base_eclass_id;
    uint64_t size_bytes;
    DType dtype;
    MemSpace mem_space;
    uint32_t num_users;
};

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

struct CacheContext
{
    const std::vector<CacheCandidate> &candidates;
    const std::vector<uint32_t> &num_users;
    const std::vector<std::vector<int>> &valid_choices;
    const std::unordered_set<BaseEClassId> &current_cache_selection;
    uint32_t k; // index into candidate_nodes
    int choice; // candidate choice (0 = uncached, 1 = cached)
};


// =============================================================================
// CacheIterator<Rules...>
// =============================================================================
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

            // Choice 0: Not cached
            valid_choices[i].push_back(0);

            // Choice 1: Cached in the e-class's memory space.
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
                    const EClass &cCls = ctx.egraph.getEClass(child);
                    if (cCls.mem_space == ms)
                    {
                        uint64_t in_size = (getSizeBytes(cCls.shape, cCls.dtype) + 4095) & ~4095ULL;
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
                const EClass &cCls = ctx.egraph.getEClass(canon_child);
                if (cCls.mem_space == ms)
                {
                    sum_inputs_in_ms += (getSizeBytes(cCls.shape, cCls.dtype) + 4095) & ~4095ULL;
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
        const EClass &cls = ctx.egraph.getEClass(ctx.egraph.findConst(e_class_id));
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

using AllENodeDominationRuleTypes = std::tuple<MemCapENodeDominationRule, FasterEquivalentENodeDominationRule>;

struct SaturationResult
{
    EGraph egraph;
    std::unordered_map<LogicalId, EClassId> nodeToEClass;
    std::unordered_map<EClassId, LogicalId> eclassToLogical;
    std::unordered_set<EClassId> cleanEClasses;
};

struct Planner
{
    CostModel &costModel;
    prune::PruningRuleSet<MemCapENodeDominationRule, FasterEquivalentENodeDominationRule> domination_rules;
    const Settings &settings;

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
                    bool matched;
#ifdef TG_PROFILE
                    auto match_start_time = std::chrono::steady_clock::now();
                    matched = rule->match(eNodeIdx, ctx);
                    auto match_end_time = std::chrono::steady_clock::now();
                    RewriteRuleProfiler::get().recordMatch(
                        rule->name(),
                        std::chrono::duration_cast<std::chrono::nanoseconds>(match_end_time - match_start_time).count(),
                        matched);
#else
                    matched = rule->match(eNodeIdx, ctx);
#endif
                    if (!matched)
                        continue;

#ifdef TG_PROFILE
                    auto apply_start_time = std::chrono::steady_clock::now();
                    rule->apply(eNodeIdx, ctx);
                    auto apply_end_time = std::chrono::steady_clock::now();
                    RewriteRuleProfiler::get().recordApply(
                        rule->name(),
                        std::chrono::duration_cast<std::chrono::nanoseconds>(apply_end_time - apply_start_time).count());
#else
                    rule->apply(eNodeIdx, ctx);
#endif
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
#ifdef TG_PROFILE
        printRewriteProfileSummary();
#endif
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

        if (cascadePruned == 0)
            return 0;

        for (uint32_t i = 0; i < numClasses; ++i)
        {
            EClassId e_class_id = egraph.find(EClassId{i});
            if (e_class_id != EClassId{i})
                continue;

            EClass &cls = egraph.getEClass(e_class_id);
            std::vector<ENodeId> filteredEnodes;
            filteredEnodes.reserve(cls.enodes.size());
            for (ENodeId enodeId : cls.enodes)
            {
                if (enode_valid[enodeId.value])
                {
                    filteredEnodes.push_back(enodeId);
                }
            }
            cls.enodes = std::move(filteredEnodes);
        }

        return cascadePruned;
    }

    std::vector<ENodeInfo> computeENodeInfos(const EGraph &egraph,
                                             const std::unordered_map<EClassId, LogicalId> &eclassToLogical,
                                             const std::unordered_set<BaseEClassId> &cachedNodes,
                                             bool strictCache)
    {
        std::vector<ENodeInfo> enodeInfos(egraph.getENodes().size());

        ProgressTimer timer(egraph.getENodes().size(), "calculating enode info");
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

            if (info.is_view || enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
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
                    {
                        info.cost = TGConstants::INF;
                    }
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

                const ReferenceGraphEntry *refEntry = nullptr;
                std::unique_ptr<Graph> pGraph;
                std::vector<LogicalId> pInputs;

                const auto &kernel = KernelRegistry::get().getKernel(enode.getKernelId());
                if (enode.getOpType() == OpType::FUSED)
                {
                    refEntry = ReferenceGraphRegistry::get().getFactory(kernel.opName);
                    if (refEntry)
                    {
                        pGraph = std::make_unique<Graph>();
                        for (uint64_t k = 0; k < kernel.min_num_inputs; ++k)
                        {
                            pInputs.push_back(pGraph->input(kernel.dummyShapes[k], kernel.dtypes[k]));
                        }
                        refEntry->factory(pInputs, *pGraph);
                    }
                }

                for (uint64_t j = 0; j < enode.getChildren().size(); j++)
                {
                    EClassId childEClassId = enode.getChildren()[j];
                    const EClass &childCls = egraph.getEClass(egraph.findConst(childEClassId));
                    inShapes.push_back(childCls.shape);

                    std::vector<uint64_t> strides_cast;
                    strides_cast.reserve(childCls.strides.size());
                    for (uint64_t s : childCls.strides)
                        strides_cast.push_back(s);
                    inStrides.push_back(std::move(strides_cast));

                    inDTypes.push_back(childCls.dtype);

                    EClassId canonChild = egraph.findConst(childEClassId);
                    bool needed = false;

                    if (enode.getOpType() == OpType::FUSED)
                    {
                        if (refEntry && pGraph)
                        {
                            auto traceToInputIdx = [&](LogicalId pid) -> int {
                                LogicalId curr = pid;
                                while (pGraph->hasNode(curr) && (pGraph->getNode(curr).opType == OpType::CONTIGUOUS ||
                                                                 pGraph->getNode(curr).opType == OpType::CAST ||
                                                                 pGraph->getNode(curr).opType == OpType::COPY_TO ||
                                                                 pGraph->getNode(curr).opType == OpType::RESHAPE ||
                                                                 pGraph->getNode(curr).opType == OpType::PERMUTE))
                                {
                                    if (pGraph->getNode(curr).child_ids.empty())
                                        break;
                                    curr = pGraph->getNode(curr).child_ids[0];
                                }
                                for (uint64_t k = 0; k < pInputs.size(); ++k)
                                {
                                    if (pInputs[k] == curr)
                                        return (int)k;
                                }
                                return -1;
                            };

                            for (const auto &pair : pGraph->nodes)
                            {
                                const TensorNode &n = pair.second;
                                for (uint64_t p_idx = 0; p_idx < n.child_ids.size(); ++p_idx)
                                {
                                    if (isConstant(n.opType, p_idx, n.child_ids.size()))
                                    {
                                        int inputIdx = traceToInputIdx(n.child_ids[p_idx]);
                                        if (kernel.min_num_inputs != kernel.max_num_inputs)
                                        {
                                            if (inputIdx == 0 && j == 0)
                                            {
                                                needed = true;
                                                break;
                                            }
                                            else if (inputIdx >= 1 && j >= 1)
                                            {
                                                needed = true;
                                                break;
                                            }
                                        }
                                        else if (inputIdx == (int)j)
                                        {
                                            needed = true;
                                            break;
                                        }
                                    }
                                }
                                if (needed)
                                    break;
                            }
                        }
                    }
                    else
                    {
                        needed = isConstant(enode.getOpType(), j, enode.getChildren().size());
                    }

                    if (needed && egraph.constantStaging.count(canonChild))
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

            enodeInfos[i] = std::move(info);
            timer.tick();
        }

        applyDominationRules(egraph, enodeInfos, eclassToLogical);

        // DP pass for subtree cost approximation (workload sum, critical path, & Sethi-Ullman memory)
        std::vector<float> eclass_dp_cost(egraph.getClasses().size(), TGConstants::INF);
        std::vector<float> eclass_dp_cp_cost(egraph.getClasses().size(), TGConstants::INF);
        std::vector<float> eclass_dp_mem(egraph.getClasses().size(), TGConstants::INF);
        std::vector<uint32_t> eclass_depth(egraph.getClasses().size(), UINT32_MAX);

        for (uint32_t i = 0; i < egraph.getClasses().size(); ++i)
        {
            EClassId cid = egraph.findConst(EClassId{i});
            if (cid.value == i)
            {
                for (ENodeId enodeId : egraph.getEClass(cid).enodes)
                {
                    if (egraph.getENode(enodeId).getOpType() == OpType::INPUT ||
                        egraph.getENode(enodeId).getOpType() == OpType::CACHE)
                    {
                        eclass_dp_cost[i] = 0.0f;
                        eclass_dp_cp_cost[i] = 0.0f;
                        eclass_depth[i] = 0;
                        const ENode &enode = egraph.getENode(enodeId);
                        float node_size = static_cast<float>(getSizeBytes(enode.getShape(), enode.getDType()));
                        eclass_dp_mem[i] = node_size;
                        enodeInfos[enodeId.value].dp_cost = 0.0f;
                        enodeInfos[enodeId.value].dp_cp_cost = 0.0f;
                        enodeInfos[enodeId.value].dp_mem = node_size;
                    }
                }
            }
        }

        bool changed = true;
        int iters = 0;
        ProgressTimer timer2(0, "calculating enode dp cost and memory");
        while (changed)
        {
            changed = false;
            iters++;
            for (uint32_t i = 0; i < egraph.getENodes().size(); ++i)
            {
                const ENode &enode = egraph.getENodes()[i];
                float cost = enodeInfos[i].cost;
                if (cost == TGConstants::INF)
                    continue;

                float sum_child_cost = 0.0f;
                float max_child_cp_cost = 0.0f;
                uint32_t max_child_depth = 0;
                bool all_children_ready = true;
                for (EClassId child : enode.getChildren())
                {
                    EClassId canon = egraph.findConst(child);
                    if (eclass_dp_cost[canon.value] == TGConstants::INF ||
                        eclass_dp_mem[canon.value] == TGConstants::INF ||
                        eclass_depth[canon.value] == UINT32_MAX)
                    {
                        all_children_ready = false;
                        break;
                    }
                    sum_child_cost += eclass_dp_cost[canon.value];
                    max_child_cp_cost = std::max(max_child_cp_cost, eclass_dp_cp_cost[canon.value]);
                    max_child_depth = std::max(max_child_depth, eclass_depth[canon.value]);
                }

                if (all_children_ready)
                {
                    float total_cost = cost + sum_child_cost;
                    float total_cp_cost = cost + max_child_cp_cost;
                    uint32_t total_depth = max_child_depth + 1;

                    // Sethi-Ullman Memory Calculation:
                    struct ChildMem
                    {
                        float m; // Peak subtree memory
                        float s; // Output tensor size
                    };
                    std::vector<ChildMem> child_mems;
                    float sum_child_sizes = 0.0f;
                    for (EClassId child : enode.getChildren())
                    {
                        EClassId canon = egraph.findConst(child);
                        const EClass &cCls = egraph.getEClass(canon);
                        float c_size = static_cast<float>(getSizeBytes(cCls.shape, cCls.dtype));
                        float c_mem = eclass_dp_mem[canon.value];
                        child_mems.push_back({c_mem, c_size});
                        sum_child_sizes += c_size;
                    }

                    // Sort children descending by (M_j - S_j) per weighted Sethi-Ullman ordering
                    std::sort(child_mems.begin(), child_mems.end(),
                              [](const ChildMem &a, const ChildMem &b) { return (a.m - a.s) > (b.m - b.s); });

                    float peak_child_eval = 0.0f;
                    float accumulated_s = 0.0f;
                    for (const auto &cm : child_mems)
                    {
                        peak_child_eval = std::max(peak_child_eval, accumulated_s + cm.m);
                        accumulated_s += cm.s;
                    }

                    float out_size = static_cast<float>(getSizeBytes(enode.getShape(), enode.getDType()));
                    bool can_be_inplace = enodeInfos[i].is_view;
                    if (!can_be_inplace && enode.getKernelId().value != 0 &&
                        KernelRegistry::get().hasKernel(enode.getKernelId()))
                    {
                        const auto &k_entry = KernelRegistry::get().getKernel(enode.getKernelId());
                        for (uint32_t inplace_idx : k_entry.safe_inplace_idxs)
                        {
                            if (inplace_idx < enode.getChildren().size())
                            {
                                EClassId child = egraph.findConst(enode.getChildren()[inplace_idx]);
                                const EClass &cCls = egraph.getEClass(child);
                                if (cCls.mem_space == enode.getMemSpace())
                                {
                                    float in_sz = static_cast<float>(getSizeBytes(cCls.shape, cCls.dtype));
                                    if (out_size <= in_sz)
                                    {
                                        can_be_inplace = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }

                    float op_exec_mem = sum_child_sizes + (can_be_inplace ? 0.0f : out_size);
                    float total_mem = std::max(peak_child_eval, op_exec_mem);

                    EClassId e_class_id = egraph.getENodeEClass(ENodeId{i});
                    EClassId canon = egraph.findConst(e_class_id);

                    if (total_depth < eclass_depth[canon.value])
                    {
                        eclass_depth[canon.value] = total_depth;
                        changed = true;
                    }

                    if (total_cost < enodeInfos[i].dp_cost || total_cp_cost < enodeInfos[i].dp_cp_cost ||
                        total_mem < enodeInfos[i].dp_mem)
                    {
                        if (total_cost < enodeInfos[i].dp_cost)
                            enodeInfos[i].dp_cost = total_cost;
                        if (total_cp_cost < enodeInfos[i].dp_cp_cost)
                            enodeInfos[i].dp_cp_cost = total_cp_cost;
                        if (total_mem < enodeInfos[i].dp_mem)
                            enodeInfos[i].dp_mem = total_mem;
                        changed = true;

                        if (total_cost < eclass_dp_cost[canon.value])
                        {
                            eclass_dp_cost[canon.value] = total_cost;
                        }
                        if (total_cp_cost < eclass_dp_cp_cost[canon.value])
                        {
                            eclass_dp_cp_cost[canon.value] = total_cp_cost;
                        }
                        if (total_mem < eclass_dp_mem[canon.value])
                        {
                            eclass_dp_mem[canon.value] = total_mem;
                        }
                    }
                }
            }
            timer2.tick();
        }

        // Backward DP pass for rev_cp_cost (Distance to Output)
        // Impose strict DAG condition: only propagate reverse critical path across edges
        // that strictly advance in topological depth, preventing cycles in saturated e-graphs.
        std::vector<float> eclass_rev_cp_cost(egraph.getClasses().size(), 0.0f);
        std::vector<std::vector<ENodeId>> consumers(egraph.getClasses().size());
        for (uint32_t i = 0; i < egraph.getENodes().size(); ++i)
        {
            if (enodeInfos[i].cost == TGConstants::INF)
                continue;

            const ENode &enode = egraph.getENodes()[i];
            EClassId parent_canon = egraph.findConst(egraph.getENodeEClass(ENodeId{i}));
            if (eclass_depth[parent_canon.value] == UINT32_MAX)
                continue;

            std::vector<EClassId> unique_children;
            for (EClassId child : enode.getChildren())
            {
                EClassId child_canon = egraph.findConst(child);
                if (eclass_depth[child_canon.value] == UINT32_MAX)
                    continue;

                // Enforce strict DAG condition
                if (eclass_depth[parent_canon.value] > eclass_depth[child_canon.value])
                {
                    if (std::find(unique_children.begin(), unique_children.end(), child_canon) == unique_children.end())
                    {
                        unique_children.push_back(child_canon);
                        consumers[child_canon.value].push_back(ENodeId{i});
                    }
                }
            }
        }

        for (auto &info : enodeInfos)
        {
            info.rev_cp_cost = 0.0f;
        }

        bool rev_changed = true;
        ProgressTimer timer3(0, "calculating enode reverse dp cost");
        while (rev_changed)
        {
            rev_changed = false;

            for (uint32_t i = 0; i < egraph.getENodes().size(); ++i)
            {
                float cost = enodeInfos[i].cost;
                if (cost == TGConstants::INF)
                    continue;

                EClassId e_class_id = egraph.getENodeEClass(ENodeId{i});
                EClassId canon = egraph.findConst(e_class_id);

                float current_rev_cost = cost + eclass_rev_cp_cost[canon.value];
                if (current_rev_cost > enodeInfos[i].rev_cp_cost)
                {
                    enodeInfos[i].rev_cp_cost = current_rev_cost;
                    rev_changed = true;
                }
            }

            for (uint32_t i = 0; i < egraph.getClasses().size(); ++i)
            {
                float max_consumer_rev = 0.0f;
                for (ENodeId consumer_id : consumers[i])
                {
                    max_consumer_rev = std::max(max_consumer_rev, enodeInfos[consumer_id.value].rev_cp_cost);
                }
                if (max_consumer_rev > eclass_rev_cp_cost[i])
                {
                    eclass_rev_cp_cost[i] = max_consumer_rev;
                    rev_changed = true;
                }
            }
            timer3.tick();
        }

        return enodeInfos;
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
                if (enodeInfos[enodeId.value].cost != TGConstants::INF)
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
            LOG(DEBUG) << "[Planner.pruneEGraph] Pruned " << totalPruned << " dominated enodes from the search space."
                       << std::endl;
        }
    }

    ExtractionResult extractBest(const LogicalId rootId, const Graph &graph, const EGraph &egraph,
                                 const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                                 const std::unordered_set<BaseEClassId> &cachedNodes,
                                 const std::unordered_map<EClassId, LogicalId> &eclassToLogical,
                                 bool stopOnFirstValid = true, bool strictCache = false, float minCompileSeconds = 0.0f,
                                 std::shared_ptr<SearchDelegate> delegate = nullptr,
                                 const std::vector<ENodeInfo> &enodeInfos = {},
                                 const std::unordered_set<EClassId> *cachedEClasses = nullptr,
                                 const std::unordered_set<EClassId> *cleanEClasses = nullptr)
    {
        auto rootIt = nodeToEClass.find(rootId);
        if (rootIt == nodeToEClass.end())
        {
            Error::throw_err("[Planner.extractBest] Root node missing from nodeToEClass.");
        }
        EClassId rootEClassId = egraph.findConst(rootIt->second);
        if (egraph.getEClass(rootEClassId).enodes.empty())
        {
            Error::throw_err("[Planner.extractBest] Root EClass has no valid ENodes remaining after pruning. Try benchmarking kernels.");
        }

        const uint64_t numClasses = egraph.getClasses().size();
        LOG(DEBUG) << "numClasses=" << numClasses;

        if (delegate)
        {
            std::vector<float> node_features;
            std::vector<uint32_t> edge_src;
            std::vector<uint32_t> edge_dst;

            uint32_t num_classes = egraph.getClasses().size();
            uint32_t num_enodes = egraph.getENodes().size();

            for (uint32_t i = 0; i < num_classes; ++i)
            {
                const EClass &cls = egraph.getClasses()[i];
                node_features.push_back(1.0f); // is_eclass
                node_features.push_back(0.0f); // is_enode
                node_features.push_back((float)countElements(cls.shape) * getDTypeSize(cls.dtype));
                node_features.push_back((float)cls.dtype);
                node_features.push_back(0.0f); // dp_cost pad

                for (ENodeId enode_id : cls.enodes)
                {
                    edge_src.push_back(i);
                    edge_dst.push_back(num_classes + enode_id.value);
                }
            }
            for (uint32_t i = 0; i < num_enodes; ++i)
            {
                const ENode &enode = egraph.getENodes()[i];
                node_features.push_back(0.0f); // is_eclass
                node_features.push_back(1.0f); // is_enode
                node_features.push_back(enodeInfos[i].cost);
                node_features.push_back((float)enode.getOpType());
                node_features.push_back(enodeInfos[i].dp_cost);

                for (EClassId child : enode.getChildren())
                {
                    edge_src.push_back(num_classes + i);
                    edge_dst.push_back(egraph.findConst(child).value);
                }
            }
            delegate->init_egraph(node_features, edge_src, edge_dst);
        }

        std::unordered_map<BaseEClassId, ParallelBuffer> preallocatedBuffers;
        preallocate(graph, egraph, nodeToEClass, cachedNodes, preallocatedBuffers);

        const std::unordered_map<MemSpace, uint64_t> reduced_caps =
            precomputeReducedMemCaps(settings.mem_caps, preallocatedBuffers);

        std::unordered_map<MemSpace, uint64_t> reserved_per_ms;
        for (const auto &kv : preallocatedBuffers)
        {
            const ParallelBuffer &buffer = kv.second;
            auto &reserved = reserved_per_ms[buffer.mem_space];
            reserved = std::max(reserved, buffer.offset + buffer.size);
        }

        UnifiedSearchPlanner unified_planner(
            egraph, rootEClassId, enodeInfos, nodeToEClass, cachedNodes, eclassToLogical, settings, delegate,
            cachedEClasses, cleanEClasses, preallocatedBuffers, reduced_caps, reserved_per_ms);
        return unified_planner.solve(minCompileSeconds, settings.only_dive, stopOnFirstValid);

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

            // Analysis also stages dense reference snapshots of computed nodes and
            // views. Only INPUT nodes own constant storage: copying a dense snapshot
            // into a broadcast view would overwrite its scalar and adjacent buffers.
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

            uint64_t final_offset_bytes = inst.outBuffer.offset;
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

        // Register all graph inputs/logical nodes and map into nodeViews
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

    struct BaseEGraphState
    {
        EGraph egraph;
        std::unordered_map<LogicalId, EClassId> nodeToEClass;
        std::unordered_map<EClassId, LogicalId> eclassToLogical;
    };

    BaseEGraphState baseState;
    bool baseStateInitialized = false;

    void initBaseEGraph(LogicalId rootId, Graph &graph, const std::vector<LogicalId> &topo, TGStore *repo = nullptr,
                        bool doSaturate = true)
    {
        if (KernelRegistry::get().nKernels() == 0)
        {
            Error::throw_err("KernelRegistry has 0 registered kernels! "
                             "Did you forget to `#include \"generated/kernels_all.gen.hpp\"` "
                             "in your entry point (e.g. bindings.cpp or main.cpp)?");
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
                {
                    contentHash = toString(nodeId);
                }

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

            if (refs.size() == 0)
            {
                Error::throw_err("[Planner.initBaseEGraph] couldn't find any kernels "
                                 "to init EClass " +
                                 toString(e_class_id) + " " + toString(baseState.egraph.getEClass(e_class_id)) +
                                 "\nNode " + toString(node, graph));
            }

            bool any_success = false;
            for (KernelId uid : refs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);

                bool path_exists = true;
                std::vector<EClassId> children;

                for (uint64_t i = 0; i < node.child_ids.size(); ++i)
                {
                    LogicalId pid = node.child_ids[i];
                    EClassId p_eclass = baseState.egraph.findConst(baseState.nodeToEClass[pid]);
                    MemSpace src_ms = input_mem_spaces[i];

                    uint64_t ruleIdx = i;
                    if (kernel.min_num_inputs != kernel.max_num_inputs)
                    {
                        ruleIdx = std::min(
                            i, static_cast<uint64_t>(kernel.min_num_inputs > 0 ? kernel.min_num_inputs - 1 : 0));
                    }
                    MemSpace dst_ms = ram;
                    if (!kernel.input_mem_spaces.empty() && ruleIdx < kernel.input_mem_spaces.size())
                    {
                        dst_ms = kernel.input_mem_spaces[ruleIdx];
                    }

                    bool requires_contig = false;
                    if (ruleIdx < kernel.requiresContiguous.size())
                    {
                        requires_contig = kernel.requiresContiguous[ruleIdx];
                    }

                    if (src_ms == dst_ms)
                    {
                        EClassId curr_eclass = p_eclass;
                        EClass curr_cls = baseState.egraph.getEClass(curr_eclass);
                        if (requires_contig && !isContiguous(curr_cls))
                        {
                            curr_eclass = addOpToEGraph(baseState.egraph, OpType::CONTIGUOUS, {curr_eclass},
                                                        curr_cls.shape, calcContiguousStrides(curr_cls.shape),
                                                        curr_cls.dtype, curr_cls.mem_space);
                        }
                        children.push_back(curr_eclass);
                    }
                    else
                    {
                        std::vector<std::vector<MemSpace>> paths = findMemSpacePaths(src_ms, dst_ms, inputs[i], {cpu});
                        if (paths.empty())
                        {
                            path_exists = false;
                            break;
                        }
                        const auto &path = paths[0];

                        EClassId curr_eclass = p_eclass;
                        EClass curr_cls = baseState.egraph.getEClass(curr_eclass);

                        if (!isContiguous(curr_cls))
                        {
                            curr_eclass = addOpToEGraph(baseState.egraph, OpType::CONTIGUOUS, {curr_eclass},
                                                        curr_cls.shape, calcContiguousStrides(curr_cls.shape),
                                                        curr_cls.dtype, curr_cls.mem_space);
                            curr_cls = baseState.egraph.getEClass(baseState.egraph.findConst(curr_eclass));
                        }

                        for (uint64_t p_idx = 1; p_idx < path.size(); ++p_idx)
                        {
                            MemSpace next_ms = path[p_idx];
                            curr_eclass = addOpToEGraph(baseState.egraph, OpType::COPY_TO, {curr_eclass},
                                                        curr_cls.shape, curr_cls.strides, curr_cls.dtype, next_ms);
                        }
                        children.push_back(curr_eclass);
                    }
                }

                if (!path_exists)
                    continue;
                any_success = true;

                std::vector<uint64_t> strides;
                if (kernel.is_view)
                {
                    strides = node.strides;
                }
                else
                {
                    strides = calcContiguousStrides(node.getShape());
                }
                ENode enode = ENode(uid, node.opType, node.opName, children, node.getShape(), strides, node.dtype, ram,
                                    {cpu}, "", 0, node.debugOrigin);
                baseState.egraph.addENode(e_class_id, enode);
            }

            if (!any_success)
            {
                Error::throw_err("[Planner.initBaseEGraph] found kernels, but could not route "
                                 "memory spaces to satisfy input constraints for node " +
                                 toString(nodeId) + "\n" + toString(node, graph));
            }
        }

        for (const auto &kv : baseState.nodeToEClass)
        {
            baseState.eclassToLogical[baseState.egraph.findConst(kv.second)] = kv.first;
        }

        if (doSaturate && settings.do_saturate)
        {
            saturate(baseState.egraph, {}, baseState.eclassToLogical, false, false, repo);
            baseState.egraph.populateBaseEClassIds();

            for (auto &kv : baseState.nodeToEClass)
            {
                kv.second = baseState.egraph.findConst(kv.second);
            }
            std::unordered_map<EClassId, LogicalId> updatedEClassToLogical;
            for (const auto &kv : baseState.eclassToLogical)
            {
                updatedEClassToLogical[baseState.egraph.findConst(kv.first)] = kv.second;
            }
            baseState.eclassToLogical = std::move(updatedEClassToLogical);
        }

        baseStateInitialized = true;
    }

    bool injectPartialPath(EGraph &egraph, const Graph &graph, LogicalId logicalId, const std::vector<Region> &regions,
                           const std::unordered_set<BaseEClassId> &cachedNodes,
                           const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                           std::unordered_map<EClassId, LogicalId> &eclassToLogical, bool strictCache = false)
    {
        bool injected = false;
        EClassId E_L = egraph.find(nodeToEClass.at(logicalId));
        const TensorNode &sourceNode = graph.getNode(logicalId);

        bool isFullRegion = false;
        if (regions.size() == 1)
        {
            const Region &reg = regions[0];
            const auto &shape = sourceNode.getShape();
            if (reg.region.size() == shape.size())
            {
                isFullRegion = true;
                for (uint64_t d = 0; d < shape.size(); ++d)
                {
                    if (reg.region[d].start != 0 || reg.region[d].stop != shape[d])
                    {
                        isFullRegion = false;
                        break;
                    }
                }
            }
        }

        if (isFullRegion)
        {
            return false;
        }

        MemSpace ram = MemSpace{1, HandleType::CPP};
        Engine cpu = Engine{0, EngineType::CPU};

        const BaseEClassId baseEClassId = egraph.getEClass(E_L).base_eclass_id;
        if (strictCache &&
            (baseEClassId == BaseEClassId{} || cachedNodes.count(baseEClassId) == 0))
        {
            return false;
        }
        MemSpace target_mem_space = cachedNodes.count(baseEClassId) ? egraph.getEClass(E_L).mem_space : ram;

        const EClass lClass = egraph.getEClass(E_L);

        EClassId E_Cache = egraph.addEClass(lClass.shape, lClass.strides, lClass.dtype, target_mem_space);
        ENode cacheNode(KernelId{0}, OpType::CACHE, "", {}, lClass.shape, lClass.strides, lClass.dtype,
                        target_mem_space, {cpu}, toString(logicalId));
        egraph.addENode(E_Cache, cacheNode);

        eclassToLogical[E_Cache] = logicalId;
        EClassId current_E = E_Cache;

        auto addConst = [&](const std::vector<int32_t> &vals) {
            return egraph.getOrAddConstantData<int32_t>({(uint32_t)vals.size()}, DType::INT32, vals);
        };

        for (uint64_t r = 0; r < regions.size(); ++r)
        {
            const Region &recomputeRegion = regions[r];

            std::vector<uint32_t> partialShape;
            for (const Dim &d : recomputeRegion.region)
                partialShape.push_back(d.stop - d.start);

            ShapePropagator prop;
            std::vector<std::vector<Region>> dirtyInputRegions = prop.backward(sourceNode, graph, {recomputeRegion});

            std::vector<int32_t> starts, ends, steps;
            for (const Dim &d : recomputeRegion.region)
            {
                starts.push_back(d.start);
                ends.push_back(d.stop);
                steps.push_back(1);
            }

            EClassId startsId = addConst(starts);
            EClassId endsId = addConst(ends);
            EClassId stepsId = addConst(steps);

            EClassId slicedEClass;

            if (sourceNode.opType == OpType::INPUT)
            {
                std::vector<uint64_t> sliceStrides = lClass.strides;

                for (uint64_t d = 0; d < starts.size(); ++d)
                {
                    int32_t start = starts[d];
                    if (start < 0)
                        start += lClass.shape[d];
                    sliceStrides[d] *= steps[d];
                }

                slicedEClass = egraph.addEClass(partialShape, sliceStrides, lClass.dtype, lClass.mem_space);

                TensorNode dOut;
                dOut.setShape(partialShape);
                dOut.dtype = lClass.dtype;
                std::vector<TensorNode> dIns(4);
                dIns[0].setShape(lClass.shape);
                dIns[0].dtype = lClass.dtype;
                dIns[1].setShape({(uint32_t)starts.size()});
                dIns[1].dtype = DType::INT32;
                dIns[2].setShape({(uint32_t)ends.size()});
                dIns[2].dtype = DType::INT32;
                dIns[3].setShape({(uint32_t)steps.size()});
                dIns[3].dtype = DType::INT32;

                std::vector<MemSpace> input_mem_spaces = {lClass.mem_space, ram, ram, ram};

                auto sliceRefs = KernelRegistry::get().findMatchingKernels(OpType::SLICE, "", dIns, dOut, true,
                                                                           lClass.mem_space, input_mem_spaces, {cpu});
                for (KernelId kid : sliceRefs)
                {
                    ENode sliceNode(kid, OpType::SLICE, "", {E_L, startsId, endsId, stepsId}, partialShape,
                                    sliceStrides, lClass.dtype, lClass.mem_space, {cpu});
                    egraph.addENode(slicedEClass, sliceNode);
                }
            }
            else
            {
                std::vector<EClassId> slicedInputs_contig;
                std::vector<EClassId> slicedInputs_non_contig;
                std::vector<TensorNode> dummyInputNodes; // Will store NON-contiguous
                std::vector<MemSpace> dummyInputMemSpaces;

                for (uint64_t p_idx = 0; p_idx < sourceNode.child_ids.size(); ++p_idx)
                {
                    LogicalId parentLogicalId = sourceNode.child_ids[p_idx];
                    EClassId E_parent = egraph.find(nodeToEClass.at(parentLogicalId));
                    const EClass pClass = egraph.getEClass(E_parent);

                    std::vector<Region> inputSliceRegions = dirtyInputRegions[p_idx];
                    if (inputSliceRegions.size() != 1)
                    {
                        Error::throw_err("[Planner.injectPartialPath] expected exactly 1 "
                                         "input slice region for parent " +
                                         std::to_string(p_idx) + " but got " +
                                         std::to_string(inputSliceRegions.size()));
                    }
                    Region inputSliceRegion = inputSliceRegions[0];

                    std::vector<uint32_t> pPartialShape;
                    for (const Dim &d : inputSliceRegion.region)
                        pPartialShape.push_back(d.stop - d.start);

                    std::vector<int32_t> pStarts, pEnds, pSteps;
                    for (const Dim &d : inputSliceRegion.region)
                    {
                        pStarts.push_back(d.start);
                        pEnds.push_back(d.stop);
                        pSteps.push_back(1);
                    }

                    EClassId pStartsId = addConst(pStarts);
                    EClassId pEndsId = addConst(pEnds);
                    EClassId pStepsId = addConst(pSteps);

                    std::vector<uint64_t> pSliceStrides = pClass.strides;
                    for (uint64_t d = 0; d < pStarts.size(); ++d)
                    {
                        int32_t start = pStarts[d];
                        if (start < 0)
                            start += pClass.shape[d];
                        pSliceStrides[d] *= pSteps[d];
                    }

                    EClassId pSliceEClass =
                        egraph.addEClass(pPartialShape, pSliceStrides, pClass.dtype, pClass.mem_space);

                    TensorNode pOut;
                    pOut.setShape(pPartialShape);
                    pOut.dtype = pClass.dtype;

                    std::vector<TensorNode> pIns(4);
                    pIns[0].setShape(pClass.shape);
                    pIns[0].dtype = pClass.dtype;
                    pIns[1].setShape({(uint32_t)pStarts.size()});
                    pIns[1].dtype = DType::INT32;
                    pIns[2].setShape({(uint32_t)pEnds.size()});
                    pIns[2].dtype = DType::INT32;
                    pIns[3].setShape({(uint32_t)pSteps.size()});
                    pIns[3].dtype = DType::INT32;

                    std::vector<MemSpace> pSliceInputMemSpaces = {pClass.mem_space, ram, ram, ram};
                    auto pSliceRefs = KernelRegistry::get().findMatchingKernels(
                        OpType::SLICE, "", pIns, pOut, true, pClass.mem_space, pSliceInputMemSpaces, {cpu});

                    for (KernelId uid : pSliceRefs)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(uid);
                        std::vector<uint64_t> strides =
                            kernel.is_view ? pSliceStrides : calcContiguousStrides(pPartialShape);
                        ENode sn(uid, OpType::SLICE, "", {E_parent, pStartsId, pEndsId, pStepsId}, pPartialShape,
                                 strides, pClass.dtype, pClass.mem_space, {cpu});
                        egraph.addENode(pSliceEClass, sn);
                    }

                    slicedInputs_non_contig.push_back(pSliceEClass);

                    EClassId pContigEClass = egraph.addEClass(pPartialShape, calcContiguousStrides(pPartialShape),
                                                              pClass.dtype, pClass.mem_space);

                    TensorNode cOut;
                    cOut.setShape(pPartialShape);
                    cOut.dtype = pClass.dtype;
                    cOut.strides = calcContiguousStrides(pPartialShape);

                    TensorNode cIn;
                    cIn.setShape(pPartialShape);
                    cIn.dtype = pClass.dtype;
                    cIn.strides = pSliceStrides;

                    auto contigRefs = KernelRegistry::get().findMatchingKernels(
                        OpType::CONTIGUOUS, "", {cIn}, cOut, true, pClass.mem_space, {pClass.mem_space}, {cpu});
                    for (KernelId uid : contigRefs)
                    {
                        const auto &kernel = KernelRegistry::get().getKernel(uid);
                        std::vector<uint64_t> strides =
                            kernel.is_view ? pSliceStrides : calcContiguousStrides(pPartialShape);
                        ENode cn(uid, OpType::CONTIGUOUS, "", {pSliceEClass}, pPartialShape, strides, pClass.dtype,
                                 pClass.mem_space, {cpu});
                        egraph.addENode(pContigEClass, cn);
                    }

                    slicedInputs_contig.push_back(pContigEClass);

                    TensorNode dummyIn;
                    dummyIn.opType = OpType::INPUT;
                    dummyIn.setShape(pPartialShape);
                    dummyIn.dtype = pClass.dtype;
                    dummyIn.strides = pSliceStrides; // NON-CONTIG
                    dummyInputNodes.push_back(dummyIn);
                    dummyInputMemSpaces.push_back(pClass.mem_space);
                }

                TensorNode dummyOut;
                dummyOut.opType = sourceNode.opType;
                dummyOut.opName = sourceNode.opName;
                dummyOut.setShape(partialShape);
                dummyOut.dtype = sourceNode.dtype;
                dummyOut.strides = calcContiguousStrides(partialShape);

                auto opRefs = KernelRegistry::get().findMatchingKernels(
                    sourceNode.opType, sourceNode.opName, dummyInputNodes, dummyOut, true, target_mem_space,
                    dummyInputMemSpaces, {cpu}, false, false, false, true); // ignore_input_contig=true
                if (opRefs.size() == 0)
                {
                    Error::throw_err("[Planner.injectPartialPath] couldn't find any "
                                     "kernels for op " +
                                     toString(sourceNode.opType));
                }

                slicedEClass = egraph.addEClass(partialShape, calcContiguousStrides(partialShape), sourceNode.dtype,
                                                target_mem_space);
                for (KernelId uid : opRefs)
                {
                    const auto &kernel = KernelRegistry::get().getKernel(uid);
                    std::vector<EClassId> actual_inputs;
                    for (uint64_t p_idx = 0; p_idx < sourceNode.child_ids.size(); ++p_idx)
                    {
                        bool reqContig = false;
                        if (p_idx < kernel.requiresContiguous.size())
                            reqContig = kernel.requiresContiguous[p_idx];

                        if (reqContig && !isContiguous(dummyInputNodes[p_idx]))
                            actual_inputs.push_back(slicedInputs_contig[p_idx]);
                        else
                            actual_inputs.push_back(slicedInputs_non_contig[p_idx]);
                    }
                    ENode sn(uid, sourceNode.opType, sourceNode.opName, actual_inputs, partialShape,
                             calcContiguousStrides(partialShape), sourceNode.dtype, target_mem_space, {cpu});
                    egraph.addENode(slicedEClass, sn);
                }
            }

            EClassId contigEClass =
                egraph.addEClass(partialShape, calcContiguousStrides(partialShape), sourceNode.dtype, target_mem_space);

            TensorNode cOut;
            cOut.setShape(partialShape);
            cOut.dtype = sourceNode.dtype;
            cOut.strides = calcContiguousStrides(partialShape);

            TensorNode cIn;
            cIn.setShape(partialShape);
            cIn.dtype = sourceNode.dtype;
            cIn.strides = calcContiguousStrides(partialShape);

            auto contigRefs = KernelRegistry::get().findMatchingKernels(OpType::CONTIGUOUS, "", {cIn}, cOut, true,
                                                                        target_mem_space, {target_mem_space}, {cpu});
            for (KernelId uid : contigRefs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                std::vector<uint64_t> strides = kernel.is_view ? cIn.strides : calcContiguousStrides(partialShape);
                ENode cn(uid, OpType::CONTIGUOUS, "", {slicedEClass}, partialShape, strides, sourceNode.dtype,
                         target_mem_space, {cpu});
                egraph.addENode(contigEClass, cn);
            }

            EClassId scatterEClass = egraph.addEClass(lClass.shape, lClass.strides, lClass.dtype, target_mem_space);

            TensorNode sOut;
            sOut.setShape(lClass.shape);
            sOut.dtype = lClass.dtype;

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

            std::vector<MemSpace> scatterInputSpaces = {target_mem_space, ram, ram, ram, ram};

            auto scatterRefs = KernelRegistry::get().findMatchingKernels(OpType::SCATTER, "", sIns, sOut, true,
                                                                         target_mem_space, scatterInputSpaces, {cpu});
            for (KernelId uid : scatterRefs)
            {
                const auto &kernel = KernelRegistry::get().getKernel(uid);
                std::vector<uint64_t> strides = (kernel.is_view) ? lClass.strides : calcContiguousStrides(lClass.shape);
                ENode sn(uid, OpType::SCATTER, "", {contigEClass, startsId, endsId, stepsId, shapeId}, lClass.shape,
                         strides, lClass.dtype, target_mem_space, {cpu});
                egraph.addENode(scatterEClass, sn);
            }

            current_E = scatterEClass;
        }

        egraph.merge(E_L, current_E);
        eclassToLogical[egraph.find(E_L)] = logicalId;
        injected = true;
        return injected;
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
            if (!graph.hasNode(nodeId))
                continue;
            if (nodeToEClass.find(nodeId) == nodeToEClass.end())
                continue;

            const TensorNode &node = graph.getNode(nodeId);
            if (node.opType == OpType::INPUT && graph.constantStaging.count(nodeId) == 0)
            {
                if (!kv.second.empty())
                {
                    injected = injected || injectPartialPath(egraph, graph, nodeId, kv.second, cachedNodes,
                                                             nodeToEClass, eclassToLogical);
                }
            }
        }
        if (injected)
        {
            egraph.rebuild();
        }
        return injected;
    }

    bool injectOutputPartialPaths(EGraph &egraph, const Graph &graph, LogicalId rootId,
                                  const std::vector<Region> &outputNeeded,
                                  const std::unordered_set<BaseEClassId> &cachedNodes,
                                  const std::unordered_map<LogicalId, EClassId> &nodeToEClass,
                                  std::unordered_map<EClassId, LogicalId> &eclassToLogical)
    {
        bool injected = false;
        if (!outputNeeded.empty() && nodeToEClass.find(rootId) != nodeToEClass.end())
        {
            injected =
                injectPartialPath(egraph, graph, rootId, outputNeeded, cachedNodes, nodeToEClass, eclassToLogical);
        }
        if (injected)
        {
            egraph.rebuild();
        }
        return injected;
    }

    Planner(CostModel &costModel, const Settings &settings = Settings::get_default())
        : costModel(costModel), settings(settings),
          domination_rules(prune::instantiate_rules<AllENodeDominationRuleTypes>("enode", settings))
    {
    }

    SaturationResult saturateBucket(const LogicalId rootId, const Graph &graph, const Bucket &bucket,
                                    const std::unordered_set<BaseEClassId> &cachedNodes = {},
                                    bool doSaturate = true, TGStore *repo = nullptr,
                                    const SaturationResult *startingState = nullptr)
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

            const EClass &cls = result.egraph.getEClass(eclassId);
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

        // Compute bucket freshness as a monotone DP over the saturated graph.
        // An original input is clean when this bucket does not dirty it; a
        // derived class is clean when one of its alternatives has only clean
        // children. Dedicated CACHE classes are clean by construction.
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
            // Shape/index constants are immutable inputs even when they were
            // created directly in the e-graph during partial-path injection.
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
                    // Input freshness is seeded from this bucket above. An
                    // empty child list must not make a dirty input clean.
                    if (enode.getOpType() == OpType::INPUT)
                        continue;
                    bool allChildrenClean = true;
                    for (EClassId child : enode.getChildren())
                    {
                        EClassId canonChild = result.egraph.findConst(child);
                        allChildrenClean = allChildrenClean && canonChild.value < clean.size() && clean[canonChild.value];
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

    CompiledGraph plan(LogicalId rootId, const Graph &graph, const Bucket &bucket,
                       const std::unordered_set<BaseEClassId> &cachedNodes, bool doSaturate = true,
                       bool strictCache = false, TGStore *repo = nullptr,
                       float minCompileSeconds = 0.0f, std::shared_ptr<SearchDelegate> delegate = nullptr)
    {
        std::vector<LogicalId> topo = topologicalSort({rootId}, graph);
        Graph tempGraph = graph;
        initBaseEGraph(rootId, tempGraph, topo, repo, false);

        const SaturationResult full_state = saturateBucket(rootId, graph, Bucket{}, {}, doSaturate, repo);
        SaturationResult bucket_state =
            saturateBucket(rootId, graph, bucket, cachedNodes, doSaturate, repo, &full_state);
        EGraph egraph = std::move(bucket_state.egraph);
        auto eclassToLogical = std::move(bucket_state.eclassToLogical);

        const std::vector<ENodeInfo> enodeInfos = computeENodeInfos(egraph, eclassToLogical, cachedNodes, strictCache);
        pruneEGraph(egraph, enodeInfos);

        auto extraction = extractBest(rootId, graph, egraph, bucket_state.nodeToEClass, cachedNodes, eclassToLogical,
                                      minCompileSeconds == 0.0f, strictCache, minCompileSeconds, delegate, enodeInfos);
        return buildCompiledGraph(rootId, graph, egraph, bucket_state.nodeToEClass, extraction, eclassToLogical,
                                  enodeInfos);
    }
};
