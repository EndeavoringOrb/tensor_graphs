// tensor_graphs_cpp/core/plan/search_engine.hpp
#pragma once

#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <array>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
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
    // Bounded storage: preserve recurring concrete witnesses without growing
    // memory with the search tree. Counts beyond the retained set stay visible.
    std::unordered_map<std::string, uint64_t> conflict_witness_counts;
    uint64_t untracked_conflict_witnesses = 0;
    bool measure_search_volume = false;
    double total_domain_bits = 0.0;
    uint64_t incomplete_volume_branches = 0;
    // Primal-dual gap tracking (docs/core/search_gap.md). The certified bound
    // is the smallest stored lower bound over open nodes below the incumbent
    // (the incumbent itself once the frontier is exhausted).
    bool measure_search_gap = false;
    double gap_sample_interval_seconds = 0.25;
    float max_certified_bound = -TGConstants::INF;
    float root_lower_bound = 0.0f;

    SearchEngine(SearchState state, std::shared_ptr<Selector> selector = nullptr,
                 std::shared_ptr<Brancher> brancher = nullptr)
        : state(std::move(state)), selector(selector ? selector : std::make_shared<PriorityQueueSelector>()),
          brancher(brancher ? brancher : std::make_shared<HeuristicBrancher>())
    {
        const char *volume_setting = std::getenv("TG_SEARCH_VOLUME");
        measure_search_volume = volume_setting && std::string(volume_setting) == "1";
        const char *gap_setting = std::getenv("TG_SEARCH_GAP");
        measure_search_gap = gap_setting && std::string(gap_setting) == "1";
        // Diagnostic override of the periodic sampling interval (seconds).
        if (const char *interval = std::getenv("TG_SEARCH_GAP_INTERVAL"))
        {
            gap_sample_interval_seconds = std::stod(interval);
            if (!std::isfinite(gap_sample_interval_seconds) || gap_sample_interval_seconds < 0.0)
                Error::throw_err("TG_SEARCH_GAP_INTERVAL must be a finite non-negative number of seconds");
        }
    }

    // Valid lower bound on the optimum over everything not yet excluded.
    // Without an incumbent and with an empty frontier the problem is
    // infeasible (INF).
    float certifiedBound() const
    {
        const float open = selector->minOpenLowerBound(incumbent_best_cost);
        return std::min(open, incumbent_best_cost);
    }

    // At an iteration boundary, active_node is null. At a propagated branch
    // point, pass the popped node so its region is counted exactly once.
    SearchVolumeEvaluation evaluateSearchVolume(const std::shared_ptr<SearchNode> &active_node = nullptr) const
    {
        if (!measure_search_volume)
            Error::throw_err("Search volume is disabled; set TG_SEARCH_VOLUME=1 before creating the engine");
        std::vector<double> box_bits;
        box_bits.reserve(selector->size() + (active_node ? 1 : 0));
        selector->appendRemainingBoxBits(incumbent_best_cost, box_bits);
        if (active_node && active_node->lower_bound < incumbent_best_cost)
            box_bits.push_back(active_node->log2_volume);
        return plan::evaluateSearchVolume(total_domain_bits, box_bits);
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
            if (measure_search_volume)
                all_nodes[nid]->log2_volume = state.domainLog2Volume();
        }

        current_node_id = target_node->id;
        return true;
    }

    bool runPropagators(VarId changed, std::string *out_conflict_reason = nullptr)
    {
        struct BoundBatchScope
        {
            SearchState &state;
            bool previous_pending;
            explicit BoundBatchScope(SearchState &search_state)
                : state(search_state), previous_pending(search_state.lower_bounds_pending)
            {
                state.lower_bounds_pending = true;
            }
            ~BoundBatchScope() { state.lower_bounds_pending = previous_pending; }
        } bound_batch_scope(state);
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
            if (detail_count < 3 || count % 1000 == 0)
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
        struct DeferredCall
        {
            VarId changed = kInvalidVarId;
            bool pending = false;
        };
        const size_t batch_slots = state.buckets.size() + 1;
        std::vector<DeferredCall> deferred_calls(propagators.size() * batch_slots);
        size_t deferred_count = 0;
        VarId active_changed = kInvalidVarId;
        auto preservePendingWork = [&]() {
            if (active_changed != kInvalidVarId)
                state.schedulePropagation(active_changed);
            for (size_t i = worklist_head; i < prop_worklist.size(); ++i)
                state.schedulePropagation(prop_worklist[i]);
            for (const auto &call : deferred_calls)
                if (call.pending && call.changed != kInvalidVarId)
                    state.schedulePropagation(call.changed);
        };
        while (worklist_head < prop_worklist.size() || deferred_count != 0)
        {
            const bool flushing_deferred = worklist_head == prop_worklist.size();
            size_t deferred_prop_idx = 0;
            VarId next_changed;
            if (flushing_deferred)
            {
                // Memory batches can shrink domains. Finish them (and their
                // consequences) before computing bucket bounds.
                size_t chosen_slot = deferred_calls.size();
                for (size_t slot = 0; slot < deferred_calls.size(); ++slot)
                {
                    if (!deferred_calls[slot].pending)
                        continue;
                    if (chosen_slot == deferred_calls.size())
                        chosen_slot = slot;
                    if (propagators[slot / batch_slots]->changesDomains())
                    {
                        chosen_slot = slot;
                        break;
                    }
                }
                deferred_prop_idx = chosen_slot / batch_slots;
                next_changed = deferred_calls[chosen_slot].changed;
                deferred_calls[chosen_slot].pending = false;
                --deferred_count;
            }
            else
            {
                next_changed = prop_worklist[worklist_head++];
                prop_queued_epoch[next_changed] = 0;
            }
            active_changed = next_changed;
            // LOG(DEBUG) << "propagator worklist pos: " << worklist_head << "/" << prop_worklist.size();

            const bool initial_propagation = next_changed == kInvalidVarId;
            const std::vector<size_t> *typed_propagators = initial_propagation
                                                               ? nullptr
                                                               : &propagators_by_var_type[static_cast<uint8_t>(
                                                                     state.var_infos[next_changed].type)];
            const size_t propagator_count = flushing_deferred ? 1 : initial_propagation ? propagators.size() : typed_propagators->size();
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
                const size_t prop_idx = flushing_deferred ? deferred_prop_idx : initial_propagation ? prop_iter : (*typed_propagators)[prop_iter];
#ifdef TG_PROFILE
                auto &timing = propagator_timings[prop_idx];
                if (timing.by_bucket.empty())
                    timing.by_bucket.resize(state.buckets.size() + 1);
                const size_t bucket_slot = initial_propagation || state.var_infos[next_changed].type == VarType::CACHED
                                               ? state.buckets.size() : state.var_infos[next_changed].bucket_idx;
                if (bucket_slot >= timing.by_bucket.size())
                    Error::throw_err("Invalid profiling bucket index");
                auto &bucket_timing = timing.by_bucket[bucket_slot];
                if (!flushing_deferred)
                {
                    ++timing.matched_calls;
                    ++bucket_timing.matched_calls;
                }
#endif
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
                if (!flushing_deferred)
                {
                    const PropagationBatch batch = prop->propagationBatch(state, next_changed);
                    if (batch == PropagationBatch::SKIP)
                        continue;
                    if (batch != PropagationBatch::NONE)
                    {
                        const size_t bucket_slot = batch == PropagationBatch::GLOBAL
                                                       ? state.buckets.size() : state.var_infos.at(next_changed).bucket_idx;
                        if (bucket_slot >= batch_slots)
                            Error::throw_err("Invalid deferred propagation bucket");
                        auto &call = deferred_calls[prop_idx * batch_slots + bucket_slot];
                        if (!call.pending)
                        {
                            call = {next_changed, true};
                            ++deferred_count;
                        }
                        continue;
                    }
                }
                const uint64_t domain_revision_before = state.getDomainRevision();
#ifdef TG_PROFILE
                const uint64_t initialization_ns_before = prop->initializationProfileNs();
                const auto propagate_start = std::chrono::steady_clock::now();
                const auto lower_before = state.start_lower_tightenings;
                const auto upper_before = state.start_upper_tightenings;
                const auto selection_before = state.selection_values_removed;
#endif
                bool propagated = prop->propagate(state, next_changed, prop_worklist);
#ifdef TG_PROFILE
                // Stop before queue coalescing, which belongs to dispatcher overhead.
                const uint64_t propagate_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - propagate_start).count());
#endif
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
                bucket_timing.propagate_calls++;
                const uint64_t initialization_ns = prop->initializationProfileNs() - initialization_ns_before;
                bucket_timing.initialization_ns += initialization_ns;
                timing.propagate_ns += propagate_ns;
                bucket_timing.propagate_ns += propagate_ns;
                bucket_timing.max_propagate_ns = std::max(bucket_timing.max_propagate_ns, propagate_ns);
                timing.max_propagate_ns = std::max(timing.max_propagate_ns, propagate_ns);
                prop_and_lb_ns_in_call += propagate_ns;
                timing.domain_changes += state.getDomainRevision() - domain_revision_before;
                timing.no_change_calls += state.getDomainRevision() == domain_revision_before;
                timing.start_lower_tightenings += state.start_lower_tightenings - lower_before;
                timing.start_upper_tightenings += state.start_upper_tightenings - upper_before;
                timing.selection_values_removed += state.selection_values_removed - selection_before;
                if (!propagated)
                    timing.contradictions++;
#endif
                if (!propagated)
                {
                    preservePendingWork();
                    std::string reason = "explicit contradiction while processing " +
                                               describe_var(next_changed) + "; " +
                                               describe_last_change(domain_revision_before) +
                                               "; pending work=" + std::to_string(prop_worklist.size() - worklist_head) +
                                               "; deferred batches=" + std::to_string(deferred_count);
                    const std::string detail = prop->conflictReason();
                    if (!detail.empty())
                        reason += "; " + detail;
                    record_conflict(prop->name(), reason);
                    if (!detail.empty())
                    {
                        const std::string signature = prop->name() + ": " + detail;
                        auto it = conflict_witness_counts.find(signature);
                        if (it != conflict_witness_counts.end()) ++it->second;
                        else if (conflict_witness_counts.size() < 256) conflict_witness_counts.emplace(signature, 1);
                        else ++untracked_conflict_witnesses;
                    }
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
                                               "; pending work=" + std::to_string(prop_worklist.size() - worklist_head) +
                                               "; deferred batches=" + std::to_string(deferred_count);
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
                    pb.size = getStorageSpanBytes(cls.shape, cls.strides, cls.dtype);
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
                pb.size = getStorageSpanBytes(cls.shape, cls.strides, cls.dtype);
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
        const char *deadline_text = std::getenv("TG_PROGRAM_DEADLINE");
        const double program_deadline = deadline_text ? std::stod(deadline_text) : 0.0;
        if (deadline_text && (!std::isfinite(program_deadline) || program_deadline <= 0 || !measure_search_volume))
            throw std::runtime_error("Invalid program evaluation deadline or disabled volume tracking");

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
        if (measure_search_volume)
        {
            state.initializeDomainVolume();
            total_domain_bits = state.domainLog2Volume();
            if (!std::isfinite(total_domain_bits))
                Error::throw_err("Search volume requires nonempty initial domains");
            incomplete_volume_branches = 0;
        }
        auto root_node = std::make_shared<SearchNode>(0, UINT32_MAX, std::make_pair(kInvalidVarId, Domain{}), 0.0f,
                                                      0.0f, 0);
        all_nodes.push_back(root_node);
        current_node_id = 0;

        std::string initial_conflict;
        state.recordStartDomainShapeAtEntry();
        if (!runPropagators(kInvalidVarId, &initial_conflict))
        {
            LOG(WARNING) << "[SearchEngine] Root state contradicted during initial propagation: "
                         << initial_conflict;
            sampleSearchGap(0, "final", true);
            if (measure_search_volume)
                reportSearchVolume(0, search_start_time);
#ifdef TG_PROFILE
            reportProfile();
#endif
            return false;
        }

        if (incumbent_best_cost < TGConstants::INF)
        {
            selector->setIncumbent(incumbent_best_cost);
        }

        reportScheduleDiagnostics();
        root_node->lower_bound = state.lower_bound;
        root_node->priority = root_node->lower_bound;
        root_node->trail_marker = state.getTrailMarker();
        if (measure_search_volume)
            root_node->log2_volume = state.domainLog2Volume();
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
        root_lower_bound = root_node->lower_bound;
        sampleSearchGap(0, "root", true);

        uint32_t iterations = 0;
#ifdef TG_PROFILE
        uint32_t last_profile_iteration = 0;
#endif
        uint64_t restore_conflicts = 0;
        uint64_t left_branch_conflicts = 0;
        uint64_t lower_bound_prunes = 0;
        std::array<uint64_t, 4> branch_counts = {};
        while (!selector->empty())
        {
            if (deadline_text)
            {
                const double now = std::chrono::duration<double>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                if (now >= program_deadline)
                {
                    sampleSearchGap(iterations, "deadline", true);
                    reportSearchVolume(iterations, search_start_time);
                    LOG(INFO) << "[Program deadline] overshoot=" << (now - program_deadline);
                    // Evaluation worker only: no incumbent is required and no plan is returned.
                    std::exit(0);
                }
            }
            sampleSearchGap(iterations, "sample", false);
#ifdef TG_PROFILE
            if (iterations > 0 && iterations % 10000 == 0 && iterations != last_profile_iteration)
            {
                LOG(INFO) << "[SearchEngine profile] iterations=" << iterations << " cumulative=true";
                reportProfile();
                last_profile_iteration = iterations;
            }
#endif
            if (timer.is_expired() && incumbent_best_cost < TGConstants::INF)
            {
                LOG(INFO) << "[SearchEngine] Timeout reached with feasible incumbent best cost: "
                          << incumbent_best_cost << " at iteration " << iterations;
                break;
            }

            // Zero requests the first feasible plan. Negative values have no
            // deadline and keep searching until the queue is exhausted.
            if (timeout_seconds == 0.0f && incumbent_best_cost < TGConstants::INF)
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

            if (measure_search_volume)
                node->log2_volume = state.domainLog2Volume();

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
                              << " at iteration " << iterations
                              << " event-epoch=" << std::setprecision(17)
                              << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
                              << " program-deadline=" << program_deadline;
                    state.best_cost = incumbent_best_cost;
                    sampleSearchGap(iterations, "incumbent", true);
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
            if (branch_type >= branch_counts.size()) Error::throw_err("Invalid branch variable type");
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
                if (iterations % 10000 == 0) reportScheduleDiagnostics();
            }

            // 1. Create Right Child (lazy alternative, pushed to queue without upfront propagation)
            double right_box_bits = std::numeric_limits<double>::quiet_NaN();
            if (measure_search_volume)
            {
                const VarId branch_var = decision.left_delta.first;
                if (branch_var >= state.numVars() || decision.right_delta.first != branch_var)
                    Error::throw_err("Search volume requires both branch children to restrict the same valid variable");
                const Domain &parent_domain = state.domains[branch_var];
                const Domain &left_domain = decision.left_delta.second;
                const Domain &right_domain = decision.right_delta.second;
                const uint64_t parent_count = domainCardinality(parent_domain);
                const uint64_t left_count = domainCardinality(left_domain);
                const uint64_t right_count = domainCardinality(right_domain);
                if (left_count == 0 || right_count == 0 ||
                    domainIntersectionCardinality(parent_domain, left_domain) != left_count ||
                    domainIntersectionCardinality(parent_domain, right_domain) != right_count)
                    Error::throw_err("Search volume branch children must be nonempty subsets of the parent domain");
                if (domainIntersectionCardinality(left_domain, right_domain) != 0)
                    Error::throw_err("Overlapping branch children would double-count the search volume union");
                if (left_count + right_count != parent_count)
                    ++incomplete_volume_branches;
                right_box_bits = node->log2_volume + std::log2(static_cast<double>(right_count)) -
                                 std::log2(static_cast<double>(parent_count));
            }
            uint32_t right_id = static_cast<uint32_t>(all_nodes.size());
            float right_lb = node->lower_bound;
            float right_prio = right_lb + 1.0f;
            auto right_node = std::make_shared<SearchNode>(
                right_id, node->id, decision.right_delta, right_lb,
                right_prio, node->depth + 1);
            right_node->log2_volume = right_box_bits;
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
                if (measure_search_volume)
                    left_node->log2_volume = state.domainLog2Volume();
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
        sampleSearchGap(iterations, "final", true);
        if (measure_search_volume)
            reportSearchVolume(iterations, search_start_time);
        LOG(INFO) << "[SearchEngine] Search finished after " << iterations << " iterations in "
                  << std::fixed << std::setprecision(2)
                  << search_elapsed_seconds << "s ("
                  << all_nodes.size() << " total nodes generated). Best cost: "
                  << (incumbent_best_cost < TGConstants::INF ? std::to_string(incumbent_best_cost) : "none");

#ifdef TG_PROFILE
        reportProfile();
#endif

        return incumbent_best_cost < TGConstants::INF;
    }

  private:
    double last_gap_sample_ = -1.0;

    // Logs (epoch, incumbent, certified bound). The supervisor integrates the
    // gap as a step function holding each sample until the next one; bounds
    // only rise and incumbents only fall, so this over-estimates the integral.
    void sampleSearchGap(uint32_t iterations, const char *event, bool force)
    {
        if (!measure_search_gap)
            return;
        const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        if (!force && last_gap_sample_ >= 0.0 && now - last_gap_sample_ < gap_sample_interval_seconds)
            return;
        last_gap_sample_ = now;
        const float bound = certifiedBound();
        if (bound < TGConstants::INF)
            max_certified_bound = std::max(max_certified_bound, bound);
        std::ostringstream line;
        line << std::setprecision(17) << "[SearchEngine gap] epoch=" << now << " event=" << event
             << " iterations=" << iterations << " incumbent="
             << (incumbent_best_cost < TGConstants::INF ? std::to_string(incumbent_best_cost) : std::string("inf"))
             << " bound=" << (bound < TGConstants::INF ? std::to_string(bound) : std::string("inf"));
        LOG(INFO) << line.str();
    }

    void reportSearchVolume(uint32_t iterations, std::chrono::steady_clock::time_point start_time,
                            const std::shared_ptr<SearchNode> &active_node = nullptr) const
    {
        const auto volume = evaluateSearchVolume(active_node);
        const double elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start_time).count();
        std::ostringstream summary;
        summary << std::setprecision(17)
                << "[SearchEngine volume] iterations=" << iterations
                << " elapsed=" << elapsed_seconds
                << " total-domain-bits=" << volume.total_domain_bits
                << " remaining-domain-bits=" << volume.remaining_domain_bits
                << " log2-remaining-fraction=" << volume.log2_remaining_fraction
                << " eliminated-bits=" << volume.eliminated_bits
                << " remaining-nodes=" << volume.remaining_nodes
                << " exhausted=" << (volume.exhausted() ? "true" : "false")
                << " incomplete-branches=" << incomplete_volume_branches;
        LOG(INFO) << summary.str();
    }

    void reportScheduleDiagnostics()
    {
        // Sample only at a propagated branch point, never from a contradictory
        // state with unfinished work. This is an independent closure audit.
        for (uint32_t bucket = 0; bucket < state.buckets.size(); ++bucket)
        {
            size_t mandatory = 0, fixed = 0, forward_gaps = 0, backward_gaps = 0;
            int32_t earliest = INT32_MAX, latest = INT32_MIN;
            std::string first_gap;
            for (const auto &pair : state.selected_vars[bucket])
            {
                const Domain &selection = state.domains[pair.second];
                if (selection.contains(0)) continue;
                ++mandatory;
                const VarId start_var = state.start_vars[bucket].at(pair.first);
                const Domain &domain = state.domains[start_var];
                fixed += domain.isFixed();
                earliest = std::min(earliest, domain.getMin());
                latest = std::max(latest, domain.getMax());
                const EClass &cls = state.bucket_egraphs[bucket].getEClass(pair.first);
                int64_t required_min = INT64_MAX;
                size_t alternatives = 0;
                std::unordered_map<EClassId, size_t> child_counts;
                for (uint32_t index = 0; index < cls.enodes.size(); ++index)
                {
                    if (!selection.contains(static_cast<int32_t>(index + 1))) continue;
                    ++alternatives;
                    int64_t alternative_min = 0;
                    std::unordered_set<EClassId> seen;
                    for (EClassId child : state.bucket_egraphs[bucket].getENode(cls.enodes[index]).getChildren())
                    {
                        child = state.bucket_egraphs[bucket].findConst(child);
                        if (!seen.insert(child).second) continue;
                        ++child_counts[child];
                        const auto child_start = state.start_vars[bucket].find(child);
                        const auto child_selection = state.selected_vars[bucket].find(child);
                        if (child_start == state.start_vars[bucket].end() || child_selection == state.selected_vars[bucket].end()) continue;
                        const Domain &selected = state.domains[child_selection->second];
                        if (selected.isFixed() && selected.fixedValue() == 0) continue;
                        alternative_min = std::max(alternative_min, static_cast<int64_t>(state.domains[child_start->second].getMin()) + 1);
                    }
                    required_min = std::min(required_min, alternative_min);
                }
                if (alternatives && required_min > domain.getMin())
                {
                    ++forward_gaps;
                    if (first_gap.empty()) first_gap = state.var_infos[start_var].name + domain.toString() + " required-min=" + std::to_string(required_min);
                }
                for (const auto &child : child_counts)
                {
                    if (child.second != alternatives) continue;
                    const auto it = state.start_vars[bucket].find(child.first);
                    if (it == state.start_vars[bucket].end()) continue;
                    if (state.domains[it->second].getMax() >= domain.getMax())
                    {
                        ++backward_gaps;
                        if (first_gap.empty()) first_gap = state.var_infos[it->second].name + state.domains[it->second].toString() + " consumer=" + state.var_infos[start_var].name + domain.toString();
                    }
                }
            }
            LOG(INFO) << "[SearchEngine schedule] bucket=" << bucket << " mandatory=" << mandatory
                      << " fixed=" << fixed << " unfixed=" << mandatory - fixed
                      << " horizon=[" << earliest << ".." << latest << "]"
                      << " capacity=" << (mandatory ? static_cast<int64_t>(latest) - earliest + 1 : 0)
                      << " closure-gaps(forward/backward)=" << forward_gaps << "/" << backward_gaps
                      << " first-gap=" << first_gap;
        }
        std::vector<std::pair<uint64_t, std::string>> witnesses;
        for (const auto &entry : conflict_witness_counts) witnesses.emplace_back(entry.second, entry.first);
        std::sort(witnesses.rbegin(), witnesses.rend());
        for (size_t i = 0; i < std::min<size_t>(5, witnesses.size()); ++i)
            LOG(INFO) << "[SearchEngine recurring conflict] count=" << witnesses[i].first << " " << witnesses[i].second;
        LOG(INFO) << "[SearchEngine conflict witnesses] retained=" << witnesses.size()
                  << " capacity=256 untracked-occurrences=" << untracked_conflict_witnesses;
    }

#ifdef TG_PROFILE
    void reportProfile()
    {
        LOG(INFO) << "[SearchEngine profile] timing=measured initialization=included-in-total branch=" << search_timing.choose_branch_ns / 1.0e9
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
                      << " matched=" << timing.matched_calls
                      << " calls=" << timing.propagate_calls
                      << " initialization=" << propagators[prop_idx]->initializationProfileNs() / 1.0e9 << "s"
                      << " total=" << timing.propagate_ns / 1.0e9
                      << "s max=" << timing.max_propagate_ns / 1.0e6 << "ms"
                      << " contradictions=" << timing.contradictions
                      << " domain-changes=" << timing.domain_changes << " no-change-calls=" << timing.no_change_calls
                      << " start-lower=" << timing.start_lower_tightenings << " start-upper=" << timing.start_upper_tightenings
                      << " selection-values-removed=" << timing.selection_values_removed
                      << " " << propagators[prop_idx]->diagnosticSummary();
            for (size_t bucket_slot = 0; bucket_slot < timing.by_bucket.size(); ++bucket_slot)
            {
                const auto &bucket_timing = timing.by_bucket[bucket_slot];
                LOG(INFO) << "[SearchEngine profile] propagator=" << propagators[prop_idx]->name()
                          << " bucket=" << (bucket_slot < state.buckets.size() ? std::to_string(bucket_slot) : "global/initial")
                          << " matched=" << bucket_timing.matched_calls
                          << " calls=" << bucket_timing.propagate_calls
                          << " initialization=" << bucket_timing.initialization_ns / 1.0e9 << "s"
                          << " total=" << bucket_timing.propagate_ns / 1.0e9 << "s"
                          << " max=" << bucket_timing.max_propagate_ns / 1.0e6 << "ms";
            }
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
    }

    struct BucketPropagatorTiming
    {
        uint64_t initialization_ns = 0;
        uint64_t matched_calls = 0;
        uint64_t propagate_calls = 0;
        uint64_t propagate_ns = 0;
        uint64_t max_propagate_ns = 0;
    };

    struct PropagatorTiming
    {
        uint64_t matched_calls = 0;
        std::vector<BucketPropagatorTiming> by_bucket;
        uint64_t propagate_calls = 0;
        uint64_t propagate_ns = 0;
        uint64_t max_propagate_ns = 0;
        uint64_t contradictions = 0;
        uint64_t domain_changes = 0;
        uint64_t no_change_calls = 0;
        uint64_t start_lower_tightenings = 0;
        uint64_t start_upper_tightenings = 0;
        uint64_t selection_values_removed = 0;
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
