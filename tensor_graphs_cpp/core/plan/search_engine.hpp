// tensor_graphs_cpp/core/plan/search_engine.hpp
#pragma once

#include <chrono>
#include <cmath>
#include <cstdint>
#include <array>
#include <iomanip>
#include <memory>
#include <sstream>
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
    std::vector<StartSelectionGuard> propagator_start_selection_guards;
    std::array<std::vector<size_t>, 4> propagators_by_var_type;

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
    std::unordered_map<std::string, uint64_t> propagation_conflicts_by_name;
    std::unordered_map<std::string, uint32_t> propagation_conflict_details_by_name;
    uint64_t restore_conflict_log_count = 0;

    SearchEngine(SearchState state, std::shared_ptr<Selector> selector = nullptr,
                 std::shared_ptr<Brancher> brancher = nullptr)
        : state(std::move(state)), selector(selector ? selector : std::make_shared<PriorityQueueSelector>()),
          brancher(brancher ? brancher : std::make_shared<HeuristicBrancher>())
    {
    }

    void addPropagator(std::unique_ptr<Propagator> prop)
    {
        const uint8_t type_mask = prop->interestedVarTypes();
        const size_t prop_idx = propagators.size();
        propagator_start_selection_guards.push_back(prop->startSelectionGuard());
        propagators.push_back(std::move(prop));
        for (uint8_t type = 0; type < propagators_by_var_type.size(); ++type)
        {
            if ((type_mask & (1u << type)) != 0)
                propagators_by_var_type[type].push_back(prop_idx);
        }
#ifdef TG_PROFILE
        propagator_timings.emplace_back();
#endif
    }

    bool restoreNode(const std::shared_ptr<SearchNode> &target_node)
    {
        if (current_node_id == target_node->id)
            return true;

        brancher->onRestore();

#ifdef TG_PROFILE
        search_timing.restore_node_calls++;
        auto lca_start = std::chrono::steady_clock::now();
#endif
        // Find Lowest Common Ancestor (LCA) using node depths in O(delta depth)
        uint32_t u = current_node_id;
        uint32_t v = target_node->id;
        uint32_t lca = UINT32_MAX;
        if (u != UINT32_MAX && v != UINT32_MAX)
        {
            while (u != UINT32_MAX && v != UINT32_MAX && all_nodes[u]->depth > all_nodes[v]->depth)
                u = all_nodes[u]->parent_id;
            while (u != UINT32_MAX && v != UINT32_MAX && all_nodes[v]->depth > all_nodes[u]->depth)
                v = all_nodes[v]->parent_id;
            while (u != v && u != UINT32_MAX && v != UINT32_MAX)
            {
                u = all_nodes[u]->parent_id;
                v = all_nodes[v]->parent_id;
            }
            if (u == v)
                lca = u;
        }

        // Backtrack to LCA's trail marker
        size_t lca_marker = (lca != UINT32_MAX) ? all_nodes[lca]->trail_marker : 0;
        state.backtrackTo(lca_marker);
#ifdef TG_PROFILE
        search_timing.restore_node_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - lca_start).count());
#endif

        // Path from child of LCA down to target_node
        restore_target_path.clear();
        uint32_t curr = target_node->id;
        while (curr != lca && curr != UINT32_MAX)
        {
            restore_target_path.push_back(curr);
            curr = all_nodes[curr]->parent_id;
        }

        // Play decisions and propagate from child of LCA down to target_node
        for (int k = static_cast<int>(restore_target_path.size()) - 1; k >= 0; --k)
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
            std::string conflict_reason;
            if (!runPropagators(all_nodes[nid]->delta.first, &conflict_reason))
            {
#ifdef TG_PROFILE
                auto bt_start = std::chrono::steady_clock::now();
#endif
                ++restore_conflict_log_count;
                if (restore_conflict_log_count <= 20 || restore_conflict_log_count % 1000 == 0)
                {
                    const VarId branch_var = all_nodes[nid]->delta.first;
                    LOG(INFO) << "[SearchEngine] Restore conflict " << restore_conflict_log_count
                              << ": node=" << nid << ", parent=" << all_nodes[nid]->parent_id
                              << ", branch=" << (branch_var < state.var_infos.size()
                                                       ? state.var_infos[branch_var].name
                                                       : std::string("<root>"))
                              << " -> " << all_nodes[nid]->delta.second.toString()
                              << "; " << conflict_reason;
                }
                current_node_id = (k < static_cast<int>(restore_target_path.size()) - 1) ? restore_target_path[k + 1] : lca;
                state.backtrackTo(current_node_id == UINT32_MAX ? 0 : all_nodes[current_node_id]->trail_marker);
#ifdef TG_PROFILE
                search_timing.restore_node_ns += static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - bt_start).count());
#endif
                return false;
            }
            all_nodes[nid]->lower_bound = state.lower_bound;
            all_nodes[nid]->trail_marker = state.getTrailMarker();
        }

        current_node_id = target_node->id;
        return true;
    }

    bool runPropagators(VarId changed, std::string *out_conflict_reason = nullptr)
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

        auto describe_var = [&](VarId var_id) {
            if (var_id == kInvalidVarId || var_id >= state.var_infos.size())
                return std::string("<initial propagation>");
            return state.var_infos[var_id].name + "=" + state.domains[var_id].toString();
        };
        auto describe_last_change = [&](uint64_t previous_revision) {
            if (state.getDomainRevision() == previous_revision)
                return std::string("no domain change in this propagator call");
            const VarId changed_var = state.getLastDomainChangeVar();
            if (changed_var == kInvalidVarId || changed_var >= state.var_infos.size())
                return std::string("domain changed, variable unavailable");
            return "last domain change: " + state.var_infos[changed_var].name + " " +
                   state.getLastDomainChangeBefore().toString() + " -> " +
                   state.getLastDomainChangeAfter().toString();
        };
        auto record_conflict = [&](const std::string &prop_name, const std::string &reason) {
            const uint64_t count = ++propagation_conflicts_by_name[prop_name];
            uint32_t &detail_count = propagation_conflict_details_by_name[prop_name];
            if (detail_count < 3)
            {
                ++detail_count;
                LOG(INFO) << "[SearchEngine conflict] propagator=" << prop_name
                          << " occurrence=" << count << ": " << reason;
            }
        };

        enqueue(changed);
        state.consumeDirtyDomains(enqueue);
        size_t worklist_scan = prop_worklist.size();

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

            const bool initial_propagation = next_changed == kInvalidVarId;
            const std::vector<size_t> *typed_propagators = initial_propagation
                                                               ? nullptr
                                                               : &propagators_by_var_type[static_cast<uint8_t>(
                                                                     state.var_infos[next_changed].type)];
            const size_t propagator_count = initial_propagation ? propagators.size() : typed_propagators->size();
            const bool changed_is_start =
                !initial_propagation && state.var_infos[next_changed].type == VarType::START;
            bool start_selection_optional = false;
            bool start_selection_fixed_positive = false;
            bool changed_start_fixed = false;
            bool changed_offset_fixed = false;
            bool changed_start_has_consumers = false;
            if (changed_is_start)
            {
                const VarInfo &start_info = state.var_infos[next_changed];
                if (start_info.selection_var < state.domains.size())
                {
                    const Domain &selection = state.domains[start_info.selection_var];
                    start_selection_optional = selection.contains(0);
                    start_selection_fixed_positive = selection.isFixed() && selection.fixedValue() > 0;
                }
                changed_start_fixed = state.domains[next_changed].isFixed();

                auto offset_it = state.offset_vars[start_info.bucket_idx].find(start_info.eclass_id);
                changed_offset_fixed = offset_it != state.offset_vars[start_info.bucket_idx].end() &&
                                       state.domains[offset_it->second].isFixed();

                if (start_info.bucket_idx < state.propagation.start_precedence.size())
                {
                    const auto &consumers_by_cid =
                        state.propagation.start_precedence[start_info.bucket_idx].consumers_by_cid;
                    changed_start_has_consumers = start_info.eclass_id.value < consumers_by_cid.size() &&
                                                  !consumers_by_cid[start_info.eclass_id.value].empty();
                }
            }
            for (size_t prop_iter = 0; prop_iter < propagator_count; ++prop_iter)
            {
                const size_t prop_idx = initial_propagation ? prop_iter : (*typed_propagators)[prop_iter];
                if (changed_is_start)
                {
                    const StartSelectionGuard guard = propagator_start_selection_guards[prop_idx];
                    if (guard != StartSelectionGuard::NONE)
                    {
                        const bool non_optional_guard =
                            guard == StartSelectionGuard::NON_OPTIONAL ||
                            guard == StartSelectionGuard::FIXED_START_NON_OPTIONAL ||
                            guard == StartSelectionGuard::NON_OPTIONAL_WITH_CONSUMERS ||
                            guard == StartSelectionGuard::FIXED_NON_OPTIONAL_WITH_CONSUMERS ||
                            guard == StartSelectionGuard::FIXED_START_NON_OPTIONAL_WITH_CONSUMERS;
                        const bool fixed_positive_guard =
                            guard == StartSelectionGuard::FIXED_POSITIVE ||
                            guard == StartSelectionGuard::FIXED_START_POSITIVE ||
                            guard == StartSelectionGuard::FIXED_NON_OPTIONAL_WITH_CONSUMERS;
                        if ((non_optional_guard && start_selection_optional) ||
                            (fixed_positive_guard && !start_selection_fixed_positive))
                            continue;

                        const bool fixed_start_guard =
                            guard == StartSelectionGuard::FIXED_START_NON_OPTIONAL ||
                            guard == StartSelectionGuard::FIXED_START_POSITIVE ||
                            guard == StartSelectionGuard::FIXED_START_NON_OPTIONAL_WITH_CONSUMERS;
                        if (fixed_start_guard && !changed_start_fixed)
                            continue;

                        if (guard == StartSelectionGuard::FIXED_OFFSET_FOR_START && !changed_offset_fixed)
                            continue;

                        const bool consumer_guard =
                            guard == StartSelectionGuard::NON_OPTIONAL_WITH_CONSUMERS ||
                            guard == StartSelectionGuard::FIXED_NON_OPTIONAL_WITH_CONSUMERS ||
                            guard == StartSelectionGuard::FIXED_START_NON_OPTIONAL_WITH_CONSUMERS;
                        if (consumer_guard && !changed_start_has_consumers)
                            continue;
                    }
                }
                auto &prop = propagators[prop_idx];
                const uint64_t domain_revision_before = state.getDomainRevision();
#ifdef TG_PROFILE
                auto &timing = propagator_timings[prop_idx];
                const bool sample_propagator_timing = ((timing.propagate_calls + 1) % 16) == 0;
                const auto propagate_start = sample_propagator_timing
                                                 ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point{};
#endif
                bool propagated = prop->propagate(state, next_changed, prop_worklist);
                // Propagators append directly to the shared worklist. Coalesce
                // duplicate pending variables before running the next item;
                // every queued variable already exposes its latest domain.
                size_t worklist_write = worklist_scan;
                for (size_t worklist_read = worklist_scan; worklist_read < prop_worklist.size(); ++worklist_read)
                {
                    const VarId queued_var = prop_worklist[worklist_read];
                    if (queued_var == kInvalidVarId || queued_var >= prop_queued_epoch.size() ||
                        prop_queued_epoch[queued_var] == current_prop_epoch)
                        continue;
                    prop_queued_epoch[queued_var] = current_prop_epoch;
                    prop_worklist[worklist_write++] = queued_var;
                }
                prop_worklist.resize(worklist_write);
                worklist_scan = worklist_write;
#ifdef TG_PROFILE
                timing.propagate_calls++;
                if (sample_propagator_timing)
                {
                    const uint64_t propagate_ns = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - propagate_start)
                            .count());
                    const uint64_t estimated_propagate_ns = propagate_ns * 16;
                    timing.propagate_ns += estimated_propagate_ns;
                    timing.max_propagate_ns = std::max(timing.max_propagate_ns, propagate_ns);
                    prop_and_lb_ns_in_call += estimated_propagate_ns;
                }
                if (!propagated)
                    timing.contradictions++;
#endif
                if (!propagated)
                {
                    preservePendingWork();
                    const std::string reason = "explicit contradiction while processing " +
                                               describe_var(next_changed) + "; " +
                                               describe_last_change(domain_revision_before) +
                                               "; pending work=" + std::to_string(prop_worklist.size());
                    record_conflict(prop->name(), reason);
                    if (out_conflict_reason)
                        *out_conflict_reason = prop->name() + ": " + reason;
#ifdef TG_PROFILE
                    record_prop_overhead();
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
                            *out_conflict_reason += state.var_infos[empty_var].name + "=" +
                                                    state.domains[empty_var].toString();
                        else
                            *out_conflict_reason += "a domain";
                        *out_conflict_reason += " while processing " + describe_var(next_changed) + "; " +
                                               describe_last_change(domain_revision_before) +
                                               "; pending work=" + std::to_string(prop_worklist.size());
                    }
                    VarId empty_var = state.getEmptyDomainVar();
                    const std::string empty_detail =
                        empty_var != kInvalidVarId && empty_var < state.var_infos.size()
                            ? state.var_infos[empty_var].name + "=" + state.domains[empty_var].toString()
                            : std::string("unknown empty domain");
                    record_conflict(prop->name(), "emptied " + empty_detail + " while processing " +
                                                       describe_var(next_changed) + "; " +
                                                       describe_last_change(domain_revision_before) +
                                                       "; pending work=" + std::to_string(prop_worklist.size()));
#ifdef TG_PROFILE
                    record_prop_overhead();
#endif
                    return false;
                }
                state.consumeDirtyDomains(enqueue);
                worklist_scan = prop_worklist.size();
            }
        }

#ifdef TG_PROFILE
        record_prop_overhead();
#endif
        // An incumbent can improve without changing a domain (for example
        // after restoring an already-propagated node).
        if (state.best_cost < TGConstants::INF && state.lower_bound >= state.best_cost)
        {
            if (out_conflict_reason)
                *out_conflict_reason = "CostLowerBoundPropagator: lower bound=" +
                                       std::to_string(state.lower_bound) + ", best=" +
                                       std::to_string(state.best_cost);
            record_conflict("CostLowerBoundPropagator", "lower bound=" + std::to_string(state.lower_bound) +
                                                            " >= best=" + std::to_string(state.best_cost));
            return false;
        }
        return true;
    }

  public:
    float evaluateMakespan(const SearchState &st, uint32_t b) const
    {
        // Simulate the selected dispatch order, waiting for both data dependencies
        // and the engines required by each operation.
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
                VarId st_v = st.start_vars[b].at(cid);
                int32_t st_val = st.domains[st_v].fixedValue();
                sorted_ops.push_back({st_val, cid});
            }
        }
        std::sort(sorted_ops.begin(), sorted_ops.end());

        MakespanSimulator sim;
        for (const auto &item : sorted_ops)
        {
            EClassId cid = item.second;
            uint32_t en_idx = static_cast<uint32_t>(st.domains[st.selected_vars[b].at(cid)].fixedValue() - 1);
            const EClass &cls = st.bucket_egraphs[b].getEClass(cid);
            ENodeId en_id = cls.enodes[en_idx];
            const ENode &enode = st.bucket_egraphs[b].getENode(en_id);
            float cost = (en_id.value < st.bucket_enode_infos[b].size()) ? st.bucket_enode_infos[b][en_id.value].cost : 0.0f;

            if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
            {
                sim.eclass_finish[cid] = 0.0f;
                continue;
            }

            const bool is_view = en_id.value < st.bucket_enode_infos[b].size() &&
                                 st.bucket_enode_infos[b][en_id.value].is_view;

            std::vector<EClassId> canon_children;
            canon_children.reserve(enode.getChildren().size());
            for (EClassId child : enode.getChildren())
            {
                canon_children.push_back(st.bucket_egraphs[b].findConst(child));
            }

            sim.addOperation(cid, canon_children, {}, UINT32_MAX, enode.getEngines(), cost, is_view);
        }

        return sim.getMakespan();
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
                    VarId st_v = st.start_vars[b].at(cid);
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
        const auto search_start_time = std::chrono::steady_clock::now();

        LOG(DEBUG) << "[SearchEngine] Starting solve: num_vars=" << state.numVars()
                   << ", candidates=" << state.candidates.size()
                   << ", buckets=" << state.buckets.size()
                   << ", timeout=" << timeout_seconds << "s";

        constexpr std::array<const char *, 4> var_type_names = {"cached", "selected", "start", "offset"};
        std::vector<std::array<uint32_t, 4>> vars_by_bucket(state.buckets.size());
        for (const VarInfo &info : state.var_infos)
        {
            const size_t type_idx = static_cast<size_t>(info.type);
            if (info.bucket_idx < vars_by_bucket.size() && type_idx < var_type_names.size())
                vars_by_bucket[info.bucket_idx][type_idx]++;
        }
        for (uint32_t b = 0; b < vars_by_bucket.size(); ++b)
        {
            LOG(INFO) << "[SearchEngine] Bucket " << b << " search variables: cached=" << vars_by_bucket[b][0]
                      << ", selected=" << vars_by_bucket[b][1] << ", start=" << vars_by_bucket[b][2]
                      << ", offset=" << vars_by_bucket[b][3];
        }

        // 1. Root node
        auto root_node = std::make_shared<SearchNode>(0, UINT32_MAX, std::make_pair(kInvalidVarId, Domain{}), 0.0f,
                                                      0.0f, 0);
        all_nodes.push_back(root_node);
        current_node_id = 0;

        std::string initial_conflict;
        if (!runPropagators(kInvalidVarId, &initial_conflict))
        {
            LOG(WARNING) << "[SearchEngine] Root state contradicted during initial propagation: "
                         << initial_conflict;
            return false;
        }

        if (incumbent_best_cost < TGConstants::INF)
        {
            selector->setIncumbent(incumbent_best_cost);
        }

        root_node->lower_bound = state.lower_bound;
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
        uint64_t restore_conflicts = 0;
        uint64_t left_branch_conflicts = 0;
        uint64_t lower_bound_prunes = 0;
        std::array<uint64_t, 4> branch_counts = {};
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
            if (!node)
                continue;
            if (node->lower_bound >= incumbent_best_cost)
            {
                ++lower_bound_prunes;
                if (lower_bound_prunes <= 10 || lower_bound_prunes % 1000 == 0)
                    LOG(INFO) << "[SearchEngine] Lower-bound prune " << lower_bound_prunes
                              << ": node=" << node->id << ", node_lb=" << node->lower_bound
                              << ", incumbent=" << incumbent_best_cost;
                continue;
            }

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
                restore_conflicts++;
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

            const size_t branch_type = static_cast<size_t>(state.var_infos[decision.left_delta.first].type);
            if (branch_type < branch_counts.size())
                branch_counts[branch_type]++;
            if (iterations % 1000 == 0)
            {
                const double elapsed_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - search_start_time).count();
                std::ostringstream conflict_summary;
                bool first_conflict = true;
                for (const auto &entry : propagation_conflicts_by_name)
                {
                    if (!first_conflict)
                        conflict_summary << ", ";
                    first_conflict = false;
                    conflict_summary << entry.first << "=" << entry.second;
                }
                LOG(INFO) << "[SearchEngine progress] iterations=" << iterations
                          << " elapsed=" << std::fixed << std::setprecision(1) << elapsed_seconds << "s"
                          << " rate=" << (elapsed_seconds > 0.0 ? iterations / elapsed_seconds : 0.0) << "/s"
                          << " queue=" << selector->size() << " depth=" << node->depth
                          << " nodes=" << all_nodes.size()
                          << " restore-conflicts=" << restore_conflicts
                          << " left-conflicts=" << left_branch_conflicts
                          << " lower-bound-prunes=" << lower_bound_prunes
                          << " branches(cached/selected/start/offset)=" << branch_counts[0] << "/"
                          << branch_counts[1] << "/" << branch_counts[2] << "/" << branch_counts[3]
                          << " conflicts{" << (first_conflict ? "none" : conflict_summary.str()) << "}"
                          << " best=" << (incumbent_best_cost < TGConstants::INF
                                               ? std::to_string(incumbent_best_cost)
                                               : "inf");
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
            std::string left_conflict;
            bool left_ok = runPropagators(decision.left_delta.first, &left_conflict);
            const float left_lb = state.lower_bound;
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
                    left_branch_conflicts++;
                    if (left_branch_conflicts <= 20 || left_branch_conflicts % 1000 == 0)
                    {
                        LOG(INFO) << "[SearchEngine] Left-branch conflict " << left_branch_conflicts
                                  << " at iter " << iterations << ", parent=" << node->id
                                  << ", branch=" << state.var_infos[decision.left_delta.first].name
                                  << " -> " << decision.left_delta.second.toString()
                                  << ": " << left_conflict;
                    }
                }
                else
                {
                    ++lower_bound_prunes;
                    if (lower_bound_prunes <= 10 || lower_bound_prunes % 1000 == 0)
                        LOG(INFO) << "[SearchEngine] Left-branch lower-bound prune " << lower_bound_prunes
                                  << ": parent=" << node->id
                                  << ", branch=" << state.var_infos[decision.left_delta.first].name
                                  << " -> " << decision.left_delta.second.toString()
                                  << ", child_lb=" << left_lb << ", incumbent=" << incumbent_best_cost;
                }
                // Left branch failed, backtrack in place to parent node
                state.backtrackTo(node->trail_marker);
                current_node_id = node->id;
            }
        }

        const double search_elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - search_start_time).count();
        LOG(INFO) << "[SearchEngine] Search finished after " << iterations << " iterations in "
                  << std::fixed << std::setprecision(2)
                  << search_elapsed_seconds << "s ("
                  << all_nodes.size() << " total nodes generated). Best cost: "
                  << (incumbent_best_cost < TGConstants::INF ? std::to_string(incumbent_best_cost) : "none");

#ifdef TG_PROFILE
        LOG(INFO) << "[SearchEngine profile] branch=" << search_timing.choose_branch_ns / 1.0e9
                  << "s restore=" << search_timing.restore_node_ns / 1.0e9
                  << "s propagation-overhead=" << search_timing.run_prop_overhead_ns / 1.0e9
                  << "s leaf-evaluation=" << search_timing.leaf_eval_ns / 1.0e9
                  << "s queue-push=" << search_timing.queue_push_ns / 1.0e9
                  << "s queue-pop=" << search_timing.queue_pop_ns / 1.0e9 << "s";
        if (auto *hb = dynamic_cast<HeuristicBrancher *>(brancher.get()))
        {
            const auto &bt = hb->getTiming();
            LOG(INFO) << "[SearchEngine profile] brancher parts: selection=" << bt.selection_ns / 1.0e9
                      << "s cache=" << bt.cache_ns / 1.0e9
                      << "s sched-topo=" << bt.sched_topo_ns / 1.0e9
                      << "s sched-start=" << bt.sched_start_ns / 1.0e9
                      << "s sched-offset=" << bt.sched_offset_ns / 1.0e9
                      << "s (preferred-offset=" << bt.preferred_offset_ns / 1.0e9 << "s, "
                      << bt.preferred_offset_calls << " calls)";
        }
        for (size_t prop_idx = 0; prop_idx < propagators.size(); ++prop_idx)
        {
            const auto &timing = propagator_timings[prop_idx];
            LOG(INFO) << "[SearchEngine profile] propagator=" << propagators[prop_idx]->name()
                      << " calls=" << timing.propagate_calls
                      << " total=" << timing.propagate_ns / 1.0e9
                      << "s max=" << timing.max_propagate_ns / 1.0e6 << "ms"
                      << " contradictions=" << timing.contradictions;
            if (auto *memory_prop = dynamic_cast<MemoryNoOverlapPropagator *>(propagators[prop_idx].get()))
            {
                LOG(INFO) << "[SearchEngine profile] memory-no-overlap parts: fixed-offset-start="
                          << memory_prop->fixedOffsetProfileNs() / 1.0e9
                          << "s write-after-read="
                          << memory_prop->writeAfterReadProfileNs() / 1.0e9 << "s";
                LOG(INFO) << "[SearchEngine profile] write-after-read paths: fixed-offset="
                          << memory_prop->writeAfterReadFixedOffsetProfileNs() / 1.0e9
                          << "s ranged-offset="
                          << memory_prop->writeAfterReadRangedOffsetProfileNs() / 1.0e9
                          << "s start="
                          << memory_prop->writeAfterReadStartProfileNs() / 1.0e9
                          << "s initial="
                          << memory_prop->writeAfterReadInitialProfileNs() / 1.0e9 << "s";
            }
        }
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
#endif
};

} // namespace plan
