// tensor_graphs_cpp/core/plan/search_engine.hpp
#pragma once

#include <chrono>
#include <cstdlib>
#include <exception>
#include <thread>
#include <cmath>
#include <functional>
#include <map>
#include <tuple>
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
#include "core/plan/primal_heuristic.hpp"
#include "core/plan/propagator.hpp"
#include "core/plan/propagators/relaxation_bound.hpp"
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
    struct DeferredCall
    {
        VarId changed = kInvalidVarId;
        bool pending = false;
    };
    std::vector<DeferredCall> deferred_calls_;
    std::vector<size_t> pending_deferred_slots_;
    std::vector<uint32_t> restore_current_path;
    std::vector<uint32_t> restore_target_path;

    float incumbent_best_cost = TGConstants::INF;
    std::vector<ExtractionResult> incumbent_extractions;
    std::unordered_set<BaseEClassId> incumbent_cached_nodes;
    // Adds the propagators that validate a plan in a restricted state
    // (addValidationPropagators). When set, solve validates its first root
    // heuristic plans in restricted states before root propagation.
    std::function<void(SearchEngine &)> add_validation_propagators;
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
                                                       ? state.var_infos[branch_var].name()
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
            // A node's box lies inside its parent's: keep the larger bound.
            all_nodes[nid]->lower_bound = std::max(all_nodes[nid]->lower_bound, state.lower_bound);
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
            return state.var_infos[var_id].name() + "=" + state.domains[var_id].toString();
        };
        auto describe_last_change = [&](uint64_t previous_revision) {
            if (state.getDomainRevision() == previous_revision)
                return std::string("no domain change in this propagator call");
            const VarId changed_var = state.getLastDomainChangeVar();
            if (changed_var == kInvalidVarId || changed_var >= state.var_infos.size())
                return std::string("domain changed, variable unavailable");
            return "last domain change: " + state.var_infos[changed_var].name() + " " +
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
        const size_t batch_slots = state.buckets.size() + 1;
        const size_t needed_slots = propagators.size() * batch_slots;
        if (deferred_calls_.size() < needed_slots)
            deferred_calls_.resize(needed_slots);
        for (size_t slot : pending_deferred_slots_)
            deferred_calls_[slot].pending = false;
        pending_deferred_slots_.clear();
        size_t deferred_count = 0;
        VarId active_changed = kInvalidVarId;
        auto preservePendingWork = [&]() {
            if (active_changed != kInvalidVarId)
                state.schedulePropagation(active_changed);
            for (size_t i = worklist_head; i < prop_worklist.size(); ++i)
                state.schedulePropagation(prop_worklist[i]);
            for (size_t slot : pending_deferred_slots_)
            {
                const auto &call = deferred_calls_[slot];
                if (call.pending && call.changed != kInvalidVarId)
                    state.schedulePropagation(call.changed);
            }
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
                size_t chosen_slot = deferred_calls_.size();
                for (size_t slot : pending_deferred_slots_)
                {
                    if (!deferred_calls_[slot].pending)
                        continue;
                    if (chosen_slot == deferred_calls_.size())
                        chosen_slot = slot;
                    if (propagators[slot / batch_slots]->changesDomains())
                    {
                        chosen_slot = slot;
                        break;
                    }
                }
                deferred_prop_idx = chosen_slot / batch_slots;
                next_changed = deferred_calls_[chosen_slot].changed;
                deferred_calls_[chosen_slot].pending = false;
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
                        const size_t slot = prop_idx * batch_slots + bucket_slot;
                        auto &call = deferred_calls_[slot];
                        if (!call.pending)
                        {
                            call = {next_changed, true};
                            pending_deferred_slots_.push_back(slot);
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
                            *out_conflict_reason += state.var_infos[empty_var].name() + "=" +
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
                            ? state.var_infos[empty_var].name() + "=" + state.domains[empty_var].toString()
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
        // Unlike TG_PROGRAM_DEADLINE (which ends an evaluation worker), the
        // search deadline stops the search and keeps the incumbent so the
        // caller can finish compiling.
        const char *search_deadline_text = std::getenv("TG_SEARCH_DEADLINE");
        const double search_deadline = search_deadline_text ? std::stod(search_deadline_text) : 0.0;
        if (search_deadline_text && (!std::isfinite(search_deadline) || search_deadline <= 0))
            throw std::runtime_error("Invalid TG_SEARCH_DEADLINE");

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
        for (auto &propagator : propagators)
            if (auto *relaxation = dynamic_cast<RelaxationBoundPropagator *>(propagator.get()))
                relaxation->deferRootBound();
        // First incumbent before root propagation (see runEarlyPrimal). The
        // root node is open at bound 0 meanwhile, so gap samples stay sound;
        // root propagation and everything after it run only if they fit
        // before the deadline (about kRootSecondsPerVar per variable).
        const double phase_deadline = program_deadline > 0.0 ? program_deadline : search_deadline;
        bool root_pushed = false;
        if (add_validation_propagators)
        {
            root_node->trail_marker = state.getTrailMarker();
            if (measure_search_volume)
                root_node->log2_volume = state.domainLog2Volume();
            selector->push(root_node);
            root_pushed = true;
            sampleSearchGap(0, "root", true);
            const bool adopted = runEarlyPrimal(phase_deadline);
            const double needed = kRootSecondsPerVar * static_cast<double>(state.numVars());
            const double now_epoch =
                std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            if (adopted && phase_deadline > 0.0 && phase_deadline - now_epoch < needed)
            {
                LOG(INFO) << "[SearchEngine] keeping the early incumbent: root propagation of " << state.numVars()
                          << " variables (~" << needed << "s) does not fit before the deadline ("
                          << (phase_deadline - now_epoch) << "s left)";
                sampleSearchGap(0, "final", true);
                if (measure_search_volume)
                    reportSearchVolume(0, search_start_time);
                return false;
            }
        }
        const auto root_propagation_start = std::chrono::steady_clock::now();
        prepareConcurrently();
        LOG(INFO) << "[SearchEngine] structural preparation took "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - root_propagation_start).count()
                  << "s";
        const bool root_consistent = runPropagators(kInvalidVarId, &initial_conflict);
        LOG(INFO) << "[SearchEngine] root propagation took "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - root_propagation_start).count()
                  << "s";
#ifdef TG_PROFILE
        LOG(INFO) << "[SearchEngine profile] after root propagation";
        reportProfile();
#endif
        if (!root_consistent)
        {
            if (incumbent_best_cost < TGConstants::INF)
                Error::throw_err("[SearchEngine] root propagation contradicted a validated incumbent: " +
                                 initial_conflict);
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
        root_bucket_lower_bounds_ = state.bucket_lower_bounds;
        root_node->trail_marker = state.getTrailMarker();
        if (measure_search_volume)
            root_node->log2_volume = state.domainLog2Volume();
        // An early incumbent's root node is already the only open node; its
        // bound only rises here.
        if (!root_pushed)
        {
#ifdef TG_PROFILE
            auto push_start = std::chrono::steady_clock::now();
            selector->push(root_node);
            search_timing.queue_push_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - push_start).count());
            search_timing.queue_push_calls++;
#else
            selector->push(root_node);
#endif
        }
        root_lower_bound = root_node->lower_bound;
        sampleSearchGap(0, "root", true);
        // First incumbent, then the interval bounds, then the heuristic's
        // improvement moves, then deeper bound refinement.
        bool deferred_root_consistent = true;
        runRootPrimalHeuristic(
            phase_deadline, [&]() {
                deferred_root_consistent = computeDeferredRootBound(*root_node, phase_deadline);
                if (deferred_root_consistent)
                    refineRootBound(*root_node, phase_deadline, -2, -2);
            },
            [&]() { refineRootBound(*root_node, phase_deadline, -1, -1); });
        if (!deferred_root_consistent)
        {
            if (incumbent_best_cost < TGConstants::INF)
                Error::throw_err("Deferred root relaxation contradicted a validated incumbent");
            sampleSearchGap(0, "final", true);
            if (measure_search_volume)
                reportSearchVolume(0, search_start_time);
            return false;
        }
        refineRootBound(*root_node, phase_deadline, 0, kMaxRefineDepth);
        // With an incumbent from the root heuristic, the remaining search is
        // proof work: refresh and split the largest open regions first.
        if (incumbent_best_cost < TGConstants::INF)
            selector->setLargestBoxFirst();

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
            if (search_deadline_text && incumbent_best_cost >= TGConstants::INF &&
                std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() >=
                    search_deadline)
                Error::throw_err("[SearchEngine] search deadline reached without a feasible plan after " +
                                 std::to_string(iterations) + " iterations");
            if (search_deadline_text && incumbent_best_cost < TGConstants::INF &&
                std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() >=
                    search_deadline - 0.5)
            {
                LOG(INFO) << "[SearchEngine] Search deadline reached with incumbent best cost: "
                          << incumbent_best_cost << " at iteration " << iterations;
                break;
            }
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
                    recordIncumbent(total_cost, iterations, program_deadline, "search");
                continue;
            }

            if (iterations <= 25 || iterations % 50 == 0)
            {
                VarId branch_var = decision.left_delta.first;
                LOG(DEBUG) << "[SearchEngine] Iter " << iterations << ": branching on var " << branch_var
                            << " (" << state.var_infos[branch_var].name() << ") [dom: " << state.domains[branch_var].toString()
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
            const float left_lb = std::max(state.lower_bound, node->lower_bound);
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
                                  << ", branch=" << state.var_infos[decision.left_delta.first].name()
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
                                  << ", branch=" << state.var_infos[decision.left_delta.first].name()
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

    // Leaf evaluation shared by the search and the primal heuristic: every
    // variable is fixed and propagation succeeded.
    float evaluateLeafCost() const
    {
        float total_cost = 0.0f;
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            float w = (b < state.bucket_weights.size()) ? state.bucket_weights[b] : 1.0f;
            total_cost += w * evaluateMakespan(state, b);
        }
        return total_cost;
    }

    void recordIncumbent(float total_cost, uint32_t iterations, double program_deadline, const char *source)
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
                  << " at iteration " << iterations << " source=" << source
                  << " event-epoch=" << std::setprecision(17)
                  << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
                  << " program-deadline=" << program_deadline;
        state.best_cost = incumbent_best_cost;
        sampleSearchGap(iterations, "incumbent", true);
    }

    static bool fixDomain(SearchState &st, VarId var, int32_t value)
    {
        const Domain &domain = st.domains[var];
        if (!domain.contains(value))
            return false;
        if (domain.isFixed())
            return true;
        st.setDomain(var, domain.is_mask ? Domain::makeFixed(value, true, domain.min_val)
                                         : Domain::makeRange(value, value));
        return true;
    }

    // Propose a disjoint allocation for each selected storage owner. Views
    // retain their owner's offset. This is only a primal proposal: the full
    // propagator pipeline still validates it before it can become a leaf.
    bool proposeDisjointOffsets()
    {
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            std::unordered_map<EClassId, uint32_t> selections;
            for (const auto &[cid, var] : state.selected_vars[b])
            {
                const Domain &domain = state.domains[var];
                if (!domain.isFixed())
                    continue;
                if (domain.fixedValue() > 0)
                    selections.emplace(cid, static_cast<uint32_t>(domain.fixedValue() - 1));
            }
            std::unordered_map<MemSpace, uint64_t> ends;
            for (const auto &[base, buffer] : state.preallocated_buffers)
                if (buffer.offset >= 0)
                    ends[buffer.mem_space] = std::max<uint64_t>(
                        ends[buffer.mem_space], state.bytesToPages(buffer.offset + buffer.size, buffer.mem_space));
            std::vector<std::pair<VarId, EClassId>> owners;
            std::vector<std::pair<VarId, VarId>> views;
            for (const auto &[cid, var] : state.offset_vars[b])
            {
                if (!selections.count(cid))
                    continue;
                const EClassId owner = resolve_view_alias(cid, state.bucket_egraphs[b], selections,
                                                         state.bucket_enode_infos[b]);
                if (owner != cid)
                {
                    auto it = state.offset_vars[b].find(owner);
                    if (it == state.offset_vars[b].end())
                        return false;
                    views.emplace_back(var, it->second);
                    continue;
                }
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                const Domain &domain = state.domains[var];
                if (domain.isFixed())
                    ends[cls.mem_space] = std::max<uint64_t>(
                        ends[cls.mem_space], static_cast<uint64_t>(domain.fixedValue()) +
                            state.bytesToPages(getStorageSpanBytes(cls.shape, cls.strides, cls.dtype), cls.mem_space));
                else
                    owners.emplace_back(var, cid);
            }
            std::sort(owners.begin(), owners.end());
            for (const auto &[var, cid] : owners)
            {
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                uint64_t &end = ends[cls.mem_space];
                end = std::max<uint64_t>(end, state.domains[var].getMin());
                if (end > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
                    !fixDomain(state, var, static_cast<int32_t>(end)))
                    return false;
                end += state.bytesToPages(getStorageSpanBytes(cls.shape, cls.strides, cls.dtype), cls.mem_space);
            }
            for (const auto &[var, owner_var] : views)
            {
                if (!state.domains[owner_var].isFixed() ||
                    !fixDomain(state, var, state.domains[owner_var].fixedValue()))
                    return false;
            }
        }
        return true;
    }

    // Applies a complete plan below the current node, propagates it with every
    // propagator and evaluates it as a normal leaf. The state is restored to
    // the current node afterwards. Returns the leaf cost, or INF with a reason.
    float tryPlan(const PrimalPlan &plan, const std::vector<PrimalBucketContext> &contexts, uint32_t iterations,
                  double program_deadline, std::string &failure)
    {
        const size_t marker = state.getTrailMarker();
        float cost = TGConstants::INF;
        const auto try_start = std::chrono::steady_clock::now();
        double fixed_seconds = 0.0, propagated_seconds = 0.0;
        uint64_t completed = 0, fallbacks = 0;
        state.completing_plan = true;
        auto finish = [&]() {
            state.completing_plan = false;
            const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - try_start).count();
            LOG(INFO) << "[PrimalHeuristic tryPlan] fixed=" << fixed_seconds << "s propagated=" << propagated_seconds
                      << "s total=" << total << "s completed=" << completed << " fallbacks=" << fallbacks;
            state.backtrackTo(marker);
            brancher->onRestore();
            return cost;
        };
        for (const auto &[base, var] : state.cached_vars)
            if (!fixDomain(state, var, plan.cached.count(base) ? 1 : 0))
            {
                failure = "cache decision outside domain: " + state.var_infos[var].name();
                return finish();
            }
        for (uint32_t b = 0; b < contexts.size(); ++b)
        {
            const RelaxationBucket &rb = contexts[b].rb;
            const PrimalBucketPlan &bp = plan.buckets[b];
            for (uint32_t u = 0; u < rb.classes.size(); ++u)
            {
                if (rb.classes[u].selected_var == kInvalidVarId)
                    continue;
                if (!fixDomain(state, rb.classes[u].selected_var, bp.value[u]))
                {
                    failure = "selection outside domain: " + state.var_infos[rb.classes[u].selected_var].name() +
                              " value " + std::to_string(bp.value[u]);
                    return finish();
                }
            }
            // Ranks: keep fixed starts, give the others increasing ranks in
            // dispatch order inside their windows.
            std::unordered_set<int32_t> pinned;
            for (uint32_t u : bp.order)
            {
                const Domain &d = state.domains[state.start_vars[b].at(rb.classes[u].cid)];
                if (d.isFixed())
                    pinned.insert(d.fixedValue());
            }
            int32_t previous = -1;
            for (uint32_t u : bp.order)
            {
                const VarId sv = state.start_vars[b].at(rb.classes[u].cid);
                const Domain d = state.domains[sv];
                if (d.isFixed())
                    continue;
                int32_t rank = std::max(previous + 1, d.getMin());
                while (rank <= d.getMax() && (pinned.count(rank) || !d.contains(rank)))
                    ++rank;
                if (!fixDomain(state, sv, rank))
                {
                    failure = "no rank for " + state.var_infos[sv].name() + " in " + d.toString() + " after " +
                              std::to_string(previous);
                    return finish();
                }
                previous = rank;
            }
        }
        fixed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - try_start).count();
        auto propagated_at = try_start;
        cost = completeFixedPlan(failure, propagated_at, completed, fallbacks);
        propagated_seconds = std::chrono::duration<double>(propagated_at - try_start).count();
        if (cost < incumbent_best_cost)
            recordIncumbent(cost, iterations, program_deadline, "root-heuristic");
        return finish();
    }

    // With the cache decisions, selections and ranks fixed (completing_plan
    // set), propagates, proposes disjoint offsets, then completes what
    // propagation left open, propagating after each choice. Returns the
    // leaf cost, or infinity with `failure` set. `propagated_at` receives the
    // time the first propagation finished.
    float completeFixedPlan(std::string &failure, std::chrono::steady_clock::time_point &propagated_at,
                            uint64_t &completed, uint64_t &fallbacks)
    {
        std::string conflict;
        const bool consistent = runPropagators(kInvalidVarId, &conflict);
        propagated_at = std::chrono::steady_clock::now();
        if (!consistent)
        {
            failure = "propagation conflict: " + conflict;
            return TGConstants::INF;
        }
        // Shared cache offsets and inactive classes are now fixed. Reserve
        // them before proposing disjoint offsets for the remaining owners.
        const size_t allocation_marker = state.getTrailMarker();
        const bool disjoint_proposed = proposeDisjointOffsets();
        const bool disjoint_valid = disjoint_proposed && runPropagators(kInvalidVarId, &conflict);
        LOG(INFO) << "[PrimalHeuristic allocation] disjoint-proposed=" << disjoint_proposed
                  << " validated=" << disjoint_valid;
        if (!disjoint_valid)
        {
            state.backtrackTo(allocation_marker);
            brancher->onRestore();
        }
        // Complete what propagation left open (typically memory offsets when no
        // static layout applies), propagating after each choice. Offsets try
        // the brancher's preferred placement, then the space past every other
        // buffer of the memory space, then the end and the start of the
        // domain; other variables take the smallest value.
        HeuristicBrancher placement;
        for (VarId v = 0; v < state.numVars(); ++v)
        {
            if (state.domains[v].isFixed())
                continue;
            const Domain d = state.domains[v];
            const VarInfo &info = state.var_infos[v];
            if (d.isEmpty())
            {
                failure = "could not complete " + info.name() + "=" + d.toString();
                return TGConstants::INF;
            }
            std::vector<int32_t> values;
            if (info.type == VarType::OFFSET)
            {
                const int32_t preferred = placement.preferredOffset(state, info.bucket_idx, info.eclass_id, d);
                if (d.contains(preferred))
                    values.push_back(preferred);
                else
                {
                    const int32_t past = placement.offsetPastAllocations(state, info.bucket_idx, info.eclass_id);
                    for (int32_t candidate : {past, d.getMax(), d.getMin()})
                        if (d.contains(candidate) && std::find(values.begin(), values.end(), candidate) == values.end())
                            values.push_back(candidate);
                }
            }
            else
                values.push_back(d.getMin());
            bool placed = false;
            ++completed;
            for (size_t val_idx = 0; val_idx < values.size(); ++val_idx)
            {
                const int32_t value = values[val_idx];
                const size_t choice_marker = state.getTrailMarker();
                if (fixDomain(state, v, value) && runPropagators(v, &conflict))
                {
                    placed = true;
                    break;
                }
                ++fallbacks;
                state.backtrackTo(choice_marker);
                brancher->onRestore();
                if (val_idx == 0 && values.size() == 1 && info.type == VarType::OFFSET)
                {
                    const int32_t past = placement.offsetPastAllocations(state, info.bucket_idx, info.eclass_id);
                    for (int32_t candidate : {past, d.getMax(), d.getMin()})
                        if (d.contains(candidate) && std::find(values.begin(), values.end(), candidate) == values.end())
                            values.push_back(candidate);
                }
            }
            if (!placed)
            {
                failure = "propagation conflict completing " + info.name() + ": " + conflict;
                return TGConstants::INF;
            }
        }
        return evaluateLeafCost();
    }

    // Validates and completes a root heuristic plan in a state restricted to
    // the plan's active classes plus every class of a cached base (so cache
    // coverage is still checked), with the validation propagators. A selected
    // alternative whose child is not active is rejected here: the restricted
    // state has no variable for that child. When valid and better, the
    // restricted state's extraction becomes the incumbent. Returns its cost
    // (infinity with `failure` set).
    float validateRestricted(const PrimalPlan &plan, const std::vector<PrimalBucketContext> &contexts,
                             double program_deadline, std::string &failure)
    {
        if (!add_validation_propagators)
            Error::throw_err("[SearchEngine] validateRestricted needs add_validation_propagators");
        const auto start = std::chrono::steady_clock::now();
        const uint32_t nb = static_cast<uint32_t>(state.buckets.size());
        if (contexts.size() != nb || plan.buckets.size() != nb)
            Error::throw_err("[SearchEngine] validateRestricted: plan and contexts must cover every bucket");
        std::vector<std::unordered_map<EClassId, int32_t>> selection(nb);
        std::vector<std::vector<EClassId>> classes(nb), order(nb);
        size_t total_classes = 0;
        for (uint32_t b = 0; b < nb; ++b)
        {
            const RelaxationBucket &rb = contexts[b].rb;
            const PrimalBucketPlan &bp = plan.buckets[b];
            const uint32_t orig_n = rb.orig_class_count ? rb.orig_class_count : static_cast<uint32_t>(rb.classes.size());
            for (uint32_t u = 0; u < orig_n; ++u)
                if (bp.value[u] > 0)
                    selection[b][rb.classes[u].cid] = bp.value[u];
            const EGraph &egraph = state.bucket_egraphs[b];
            for (const auto &[cid, value] : selection[b])
            {
                const EClass &cls = egraph.getEClass(cid);
                if (value > static_cast<int32_t>(cls.enodes.size()))
                    Error::throw_err("[SearchEngine] validateRestricted: selection outside its class");
                for (EClassId child : egraph.getENode(cls.enodes[static_cast<size_t>(value - 1)]).getChildren())
                {
                    const EClassId canonical = egraph.findConst(child);
                    if (!selection[b].count(canonical) && state.selected_vars[b].count(canonical))
                    {
                        failure = "bucket " + std::to_string(b) + " selects class " + std::to_string(cid.value) +
                                  " without its child " + std::to_string(canonical.value);
                        return TGConstants::INF;
                    }
                }
            }
            for (EClassId cid : state.reachable_cids[b])
                if (selection[b].count(cid) || plan.cached.count(egraph.getEClass(cid).base_eclass_id))
                    classes[b].push_back(cid);
            std::unordered_set<EClassId> ordered;
            for (uint32_t u : bp.order)
            {
                const EClassId cid = rb.classes[u].cid;
                if (selection[b].count(cid) && ordered.insert(cid).second)
                    order[b].push_back(cid);
            }
            total_classes += classes[b].size();
        }

        // Freeing the restricted state takes a few tenths of a second at large
        // bucket counts and nothing reads it afterwards: free it off this
        // thread.
        struct BackgroundEngineFree
        {
            std::unique_ptr<SearchEngine> engine;
            ~BackgroundEngineFree() { std::thread([doomed = std::move(engine)]() mutable {}).detach(); }
        } holder{std::make_unique<SearchEngine>(state.restrictedTo(classes), nullptr,
                                                std::make_shared<HeuristicBrancher>())};
        SearchEngine &validation = *holder.engine;
        validation.measure_search_volume = false;
        validation.measure_search_gap = false;
        add_validation_propagators(validation);
        SearchState &vs = validation.state;
        const double build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        vs.completing_plan = true;
        for (const auto &[base, var] : vs.cached_vars)
            if (!fixDomain(vs, var, plan.cached.count(base) ? 1 : 0))
            {
                failure = "cache decision outside domain: " + vs.var_infos[var].name();
                return TGConstants::INF;
            }
        for (uint32_t b = 0; b < nb; ++b)
        {
            for (EClassId cid : classes[b])
            {
                const auto it = selection[b].find(cid);
                const VarId var = vs.selected_vars[b].at(cid);
                if (!fixDomain(vs, var, it == selection[b].end() ? 0 : it->second))
                {
                    failure = "selection outside domain: " + vs.var_infos[var].name();
                    return TGConstants::INF;
                }
            }
            // Dispatch order gives the ranks.
            int32_t rank = 0;
            for (EClassId cid : order[b])
            {
                const VarId var = vs.start_vars[b].at(cid);
                if (!fixDomain(vs, var, rank++))
                {
                    failure = "no rank " + std::to_string(rank - 1) + " for " + vs.var_infos[var].name();
                    return TGConstants::INF;
                }
            }
        }
        uint64_t completed = 0, fallbacks = 0;
        auto propagated_at = start;
        const float cost = validation.completeFixedPlan(failure, propagated_at, completed, fallbacks);
        const auto done = std::chrono::steady_clock::now();
        LOG(INFO) << "[PrimalHeuristic restricted] classes=" << total_classes << " vars=" << vs.numVars()
                  << " build=" << build_seconds
                  << "s propagated=" << std::chrono::duration<double>(propagated_at - start).count()
                  << "s total=" << std::chrono::duration<double>(done - start).count() << "s completed=" << completed
                  << " fallbacks=" << fallbacks;
        if (!std::isfinite(cost))
            return TGConstants::INF;
        if (cost < incumbent_best_cost)
        {
            incumbent_best_cost = cost;
            selector->setIncumbent(incumbent_best_cost);
            incumbent_extractions = validation.extractSolution(vs);
            incumbent_cached_nodes.clear();
            for (const auto &[base, var] : vs.cached_vars)
                if (vs.domains[var].isFixed() && vs.domains[var].fixedValue() == 1)
                    incumbent_cached_nodes.insert(base);
            LOG(INFO) << "[SearchEngine] New best cost: " << incumbent_best_cost << " at iteration 0"
                      << " source=root-heuristic-restricted event-epoch=" << std::setprecision(17)
                      << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()
                      << " program-deadline=" << program_deadline;
            state.best_cost = incumbent_best_cost;
            sampleSearchGap(0, "incumbent", true);
        }
        return cost;
    }

    // Before root propagation: the root heuristic's plans built on the
    // unpropagated domains, each validated in a restricted state until one
    // holds. Returns whether an incumbent was adopted.
    bool runEarlyPrimal(double program_deadline)
    {
        const auto start = std::chrono::steady_clock::now();
        std::vector<PrimalBucketContext> contexts;
        struct BackgroundFree
        {
            std::vector<PrimalBucketContext> &contexts;
            ~BackgroundFree() { std::thread([doomed = std::move(contexts)]() mutable {}).detach(); }
        } background_free{contexts};
        bool adopted = false;
        buildRootPrimalPlans(state, contexts, [&](const PrimalPlan &plan) {
            if (!std::isfinite(plan.cost))
                return false;
            std::string failure;
            const float cost = validateRestricted(plan, contexts, program_deadline, failure);
            LOG(INFO) << "[PrimalHeuristic] early restricted validate predicted=" << plan.cost << " evaluated="
                      << (std::isfinite(cost) ? std::to_string(cost) : std::string("rejected"))
                      << (failure.empty() ? "" : " failure=" + failure);
            adopted = adopted || std::isfinite(cost);
            return std::isfinite(cost);
        });
        LOG(INFO) << "[PrimalHeuristic] early primal "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << "s adopted="
                  << adopted;
        return adopted;
    }

    // Structural indexes (the shared propagation state and each propagator's
    // prepare) are independent of each other: build them concurrently rather
    // than one after another as root propagation first needs them.
    void prepareConcurrently()
    {
        std::vector<std::function<void()>> tasks;
        tasks.push_back([this]() { state.ensurePropagationState(); });
        for (auto &propagator : propagators)
            tasks.push_back([this, p = propagator.get()]() { p->prepare(state); });
        std::vector<std::exception_ptr> errors(tasks.size());
        std::vector<std::thread> threads;
        for (size_t i = 0; i < tasks.size(); ++i)
            threads.emplace_back([&, i]() {
                try
                {
                    tasks[i]();
                }
                catch (...)
                {
                    errors[i] = std::current_exception();
                }
            });
        for (auto &thread : threads)
            thread.join();
        for (auto &error : errors)
            if (error)
                std::rethrow_exception(error);
    }

    bool computeDeferredRootBound(SearchNode &root, double deadline_epoch)
    {
        if (state.completing_plan || state.getTrailMarker() != root.trail_marker)
            Error::throw_err("Deferred root bound requires restored root domains");
        auto deadline = std::chrono::steady_clock::time_point::max();
        if (deadline_epoch > 0.0)
        {
            const double now = std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            deadline = std::chrono::steady_clock::now() +
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(std::max(0.0, 0.5 * (deadline_epoch - now))));
        }
        for (auto &propagator : propagators)
            if (auto *relaxation = dynamic_cast<RelaxationBoundPropagator *>(propagator.get()))
                if (!relaxation->computeRootBound(state, deadline))
                {
                    LOG(INFO) << "[SearchEngine] Deferred root relaxation conflict: " << relaxation->conflictReason();
                    return false;
                }
        // Trial validation leaves per-domain bound caches at the trial.
        // Only the saved root bounds and the root relaxation are valid here.
        float bound = 0.0f;
        for (size_t b = 0; b < root_bucket_lower_bounds_.size(); ++b)
        {
            const float weight = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0f;
            if (weight > 0.0f)
                bound += weight * std::max(root_bucket_lower_bounds_[b], state.bucket_relaxation_lower_bounds[b]);
        }
        root.lower_bound = std::max(root.lower_bound, bound);
        root.priority = root.lower_bound;
        root_lower_bound = root.lower_bound;
        sampleSearchGap(0, "sample", true);
        return true;
    }

    // Computes the relaxation's interval bound, then nests its threshold
    // splits one level deeper at a time (docs/core/relaxation_bound.md,
    // "Case splits"), raising the root node's bound after each level. It spends at most half the time
    // left before `deadline_epoch` (5 s without a deadline): the search
    // below the root cannot raise these bounds, deeper splits can.
    void refineRootBound(SearchNode &root, double deadline_epoch, int first_depth, int last_depth)
    {
        RelaxationBoundPropagator *relaxation = nullptr;
        for (auto &propagator : propagators)
            if (auto *candidate = dynamic_cast<RelaxationBoundPropagator *>(propagator.get()))
                relaxation = candidate;
        if (!relaxation)
            return;
        const double now_epoch =
            std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        const double budget = deadline_epoch > 0.0 ? 0.5 * (deadline_epoch - now_epoch) : 5.0;
        if (!(budget > 0.0))
            return;
        const auto start = std::chrono::steady_clock::now();
        const auto deadline = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                          std::chrono::duration<double>(budget));
        // Level -2 is the interval bound over flexible windows, level -1 adds
        // long windows (kLongWindowLinks) ending at its partition's
        // boundaries, level 0 the stride windows (all deferred from root
        // propagation so the heuristic starts sooner), then nested splits.
        for (int depth = first_depth; depth <= last_depth && std::chrono::steady_clock::now() < deadline; ++depth)
        {
            const bool raised = depth == -2   ? relaxation->computeFlexibleIntervals(state, deadline)
                                : depth == -1 ? relaxation->computeFlexibleIntervals(state, deadline, kLongWindowLinks)
                                : depth == 0  ? relaxation->computeIntervals(state, deadline)
                                              : relaxation->refine(state, depth, deadline);
            if (!raised)
                continue;
            // The root's bucket bounds from root propagation, raised by the
            // refined relaxation bounds (the state's own bucket bounds can be
            // stale after the heuristic restored the root).
            float bound = 0.0f;
            for (size_t b = 0; b < root_bucket_lower_bounds_.size(); ++b)
            {
                const float weight = b < state.bucket_weights.size() ? state.bucket_weights[b] : 1.0f;
                if (weight > 0.0f)
                    bound += weight * std::max(root_bucket_lower_bounds_[b], state.bucket_relaxation_lower_bounds[b]);
            }
            if (bound > root.lower_bound)
            {
                root.lower_bound = bound;
                root_lower_bound = bound;
                sampleSearchGap(0, "sample", true);
            }
            LOG(INFO) << "[RelaxationBound refine] depth=" << depth << " bound=" << root.lower_bound << " elapsed="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << "s";
        }
    }
    static constexpr int kMaxRefineDepth = 6;
    // Root propagation, the root heuristic on propagated domains and one
    // validation take about this long per search variable (about 1.2 us/var
    // for root propagation alone, measured at N=21 on the Linux box).
    static constexpr double kRootSecondsPerVar = 2.5e-6;
    static constexpr size_t kLongWindowLinks = 24;
    std::vector<float> root_bucket_lower_bounds_;

    // `after_first` runs once the first plan is validated (or none was),
    // before the improvement moves; `after_windows` after the first window
    // moves, which then run again on the partition it leaves (the bound's
    // long windows).
    void runRootPrimalHeuristic(double program_deadline, const std::function<void()> &after_first,
                                const std::function<void()> &after_windows)
    {
        const auto start = std::chrono::steady_clock::now();
        std::vector<PrimalBucketContext> contexts;
        // The relaxations take most of a second to free at large bucket
        // counts and nothing reads them after the heuristic, so they are
        // freed off the critical path (extraction and plan writing follow).
        struct BackgroundFree
        {
            std::vector<PrimalBucketContext> &contexts;
            ~BackgroundFree() { std::thread([doomed = std::move(contexts)]() mutable {}).detach(); }
        } background_free{contexts};
        // Plans already given to tryPlan. Validating one again gives the same
        // result at best (the incumbent only falls), so it is skipped.
        std::vector<PrimalPlan> tried_storage;
        auto samePlan = [](const PrimalPlan &a, const PrimalPlan &b) {
            if (a.buckets.size() != b.buckets.size() || a.cached != b.cached)
                return false;
            for (size_t i = 0; i < a.buckets.size(); ++i)
                if (a.buckets[i].value != b.buckets[i].value || a.buckets[i].order != b.buckets[i].order)
                    return false;
            return true;
        };
        auto wasTried = [&](const PrimalPlan &plan) {
            for (const PrimalPlan &tried : tried_storage)
                if (samePlan(tried, plan))
                    return true;
            return false;
        };
        // Construct and validate starts lazily. Once one is accepted, spend
        // the remaining budget improving it and proving bounds rather than
        // constructing every possible engine weighting first.
        std::vector<PrimalPlan> plans = buildRootPrimalPlans(state, contexts, [&](const PrimalPlan &plan) {
            // A start the heuristic itself prices at infinity (it uses a
            // disallowed cache) would be rejected; skip its validation, and
            // that of a start not predicted to beat an early incumbent.
            if (!std::isfinite(plan.cost) || !(plan.cost < incumbent_best_cost - 1e-6))
                return false;
            tried_storage.push_back(plan);
            std::string failure;
            const float cost = tryPlan(plan, contexts, 0, program_deadline, failure);
            LOG(INFO) << "[PrimalHeuristic] early validate predicted=" << plan.cost << " evaluated="
                      << (std::isfinite(cost) ? std::to_string(cost) : std::string("rejected"))
                      << (failure.empty() ? "" : " failure=" + failure);
            return std::isfinite(cost);
        });
        const double build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (plans.empty())
        {
            LOG(INFO) << "[PrimalHeuristic] no consistent plan (build " << build_seconds << "s)";
            after_first();
            after_windows();
            return;
        }
        auto describe = [](const PrimalPlan &plan) {
            std::ostringstream os;
            for (size_t e = 0; e < plan.lambda.size(); ++e)
                os << (e ? "," : "") << plan.lambda[e];
            return os.str();
        };
        std::string failure;
        LOG(INFO) << "[PrimalHeuristic] " << plans.size() << " initial plan(s), best predicted=" << plans.front().cost
                  << " lambda=[" << describe(plans.front()) << "] cached=" << plans.front().cached.size()
                  << " build=" << build_seconds << "s";
        auto validate = [&](const std::vector<const PrimalPlan *> &order) {
            for (const PrimalPlan *plan : order)
            {
                // Past the program deadline with an incumbent, stop: the
                // remaining time belongs to extraction and plan writing.
                if (incumbent_best_cost < TGConstants::INF && program_deadline > 0.0 &&
                    std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() >=
                        program_deadline)
                    return;
                if (!(plan->cost < incumbent_best_cost - 1e-6) || wasTried(*plan))
                    continue;
                tried_storage.push_back(*plan);
                failure.clear();
                const float cost = tryPlan(*plan, contexts, 0, program_deadline, failure);
                LOG(INFO) << "[PrimalHeuristic] validate predicted=" << plan->cost << " evaluated="
                          << (std::isfinite(cost) ? std::to_string(cost) : std::string("rejected"))
                          << " lambda=[" << describe(*plan) << "]" << (failure.empty() ? "" : " failure=" + failure);
                if (std::isfinite(cost))
                    return;
            }
        };
        // Starts were already validated during construction; improve them
        // without repeating a rejected or already accepted validation.
        const std::vector<PrimalPlan> initial_plans = plans;
        // Local search from every initial plan until no move improves (cheap),
        // bounded by the program deadline; validate improvements afterwards.
        const auto ls_start = std::chrono::steady_clock::now();
        auto ls_deadline = std::chrono::steady_clock::time_point::max();
        if (program_deadline > 0.0)
        {
            const double now = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            // Leave time to validate the result afterwards (tryPlan takes
            // about 3.5x the plan build in practice, at most ~15 s so far;
            // the build includes the estimates' relaxation evaluations).
            const double reserve = std::min(4.0 * build_seconds + 10.0, 20.0);
            ls_deadline = ls_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                         std::chrono::duration<double>(std::max(0.0, program_deadline - reserve - now)));
        }
        uint64_t evaluations = 0;
        {
            // One round of tasks: the best starts (plans are sorted by
            // predicted cost), as many as threads per weighted bucket.
            std::vector<double> initial;
            std::vector<PrimalPlan *> starts;
            size_t weighted = 0;
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
                weighted += !(b < state.bucket_weights.size()) || state.bucket_weights[b] > 0.0f;
            const size_t threads = std::max(1u, std::min(std::thread::hardware_concurrency(), 16u));
            const size_t limit = std::max<size_t>(1, threads / std::max<size_t>(1, weighted));
            for (PrimalPlan &plan : plans)
            {
                initial.push_back(plan.cost);
                if (starts.size() < limit)
                    starts.push_back(&plan);
            }
            improvePrimalPlans(state, contexts, starts, ls_deadline, evaluations);
            const double ls_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - ls_start).count();
            for (size_t i = 0; i < plans.size(); ++i)
                LOG(INFO) << "[PrimalHeuristic] local search lambda=[" << describe(plans[i]) << "] " << initial[i]
                          << " -> " << plans[i].cost;
            LOG(INFO) << "[PrimalHeuristic] local search " << starts.size() << " of " << plans.size() << " start(s): " << evaluations
                      << " schedules, " << ls_seconds << "s";
        }
        // Validate in cost order (improved plans first, then the initial
        // ones) until one is accepted, so the first incumbent comes early.
        {
            std::vector<const PrimalPlan *> order;
            for (const PrimalPlan &plan : plans)
                order.push_back(&plan);
            std::stable_sort(order.begin(), order.end(),
                             [](const PrimalPlan *a, const PrimalPlan *b) { return a->cost < b->cost; });
            for (const PrimalPlan &plan : initial_plans)
                order.push_back(&plan);
            validate(order);
        }
        after_first();
        // Window moves on the best local optimum, then local search again.
        {
            size_t best = 0;
            for (size_t i = 1; i < plans.size(); ++i)
                if (plans[i].cost < plans[best].cost)
                    best = i;
            const double initial = plans[best].cost;
            const auto wm_start = std::chrono::steady_clock::now();
            // The bound's flexible partition (same relaxation construction).
            RelaxationBoundPropagator *relaxation = nullptr;
            for (auto &propagator : propagators)
                if (auto *candidate = dynamic_cast<RelaxationBoundPropagator *>(propagator.get()))
                    relaxation = candidate;
            auto partitioned = [&](uint32_t b) -> RelaxationBoundComputer * {
                if (!relaxation)
                    return nullptr;
                RelaxationBoundComputer *computer = relaxation->computerFor(b);
                if (!computer)
                    return nullptr;
                const RelaxationBucket &theirs = relaxation->relaxationOf(b);
                const RelaxationBucket &ours = contexts[b].rb;
                bool same = theirs.classes.size() == ours.classes.size();
                for (size_t u = 0; same && u < ours.classes.size(); ++u)
                    same = theirs.classes[u].cid == ours.classes[u].cid &&
                           theirs.classes[u].alternatives.size() == ours.classes[u].alternatives.size();
                return same ? computer : nullptr;
            };
            if (improveByWindows(state, contexts, plans[best], ls_deadline, evaluations, partitioned))
                improvePrimalPlan(state, contexts, plans[best], ls_deadline, evaluations);
            LOG(INFO) << "[PrimalHeuristic] window moves " << initial << " -> " << plans[best].cost << " ("
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - wm_start).count() << "s)";
            if (plans[best].cost < std::min<double>(initial, incumbent_best_cost) * (1.0 - 1e-3))
                validate({&plans[best]});
            // Again on the partition with the long windows.
            after_windows();
            const double before_long = plans[best].cost;
            const auto lw_start = std::chrono::steady_clock::now();
            if (improveByWindows(state, contexts, plans[best], ls_deadline, evaluations, partitioned))
                improvePrimalPlan(state, contexts, plans[best], ls_deadline, evaluations);
            LOG(INFO) << "[PrimalHeuristic] long window moves " << before_long << " -> " << plans[best].cost << " ("
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - lw_start).count() << "s)";
            if (plans[best].cost < std::min<double>(before_long, incumbent_best_cost) * (1.0 - 1e-3))
                validate({&plans[best]});
        }
        // Then cache-set moves on the best local optimum (they rarely pay
        // much: capped at 10 s), validated (as expensive as the first
        // validation) only if they predict at least 0.1% better.
        {
            size_t best = 0;
            for (size_t i = 1; i < plans.size(); ++i)
                if (plans[i].cost < plans[best].cost)
                    best = i;
            const double initial = plans[best].cost;
            uint32_t dropped = 0;
            const auto cs_start = std::chrono::steady_clock::now();
            const auto cs_deadline = std::min(ls_deadline, cs_start + std::chrono::seconds(10));
            uint32_t added = 0;
            addCacheBases(state, contexts, plans[best], cs_deadline, evaluations, added);
            improveCacheSet(state, contexts, plans[best], cs_deadline, evaluations, dropped);
            const double cs_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - cs_start).count();
            LOG(INFO) << "[PrimalHeuristic] cache-set moves " << initial << " -> " << plans[best].cost
                      << " added=" << added << " dropped=" << dropped << " (" << cs_seconds << "s)";
            if (plans[best].cost < std::min<double>(initial, incumbent_best_cost) * (1.0 - 1e-3))
                validate({&plans[best]});
        }
        LOG(INFO) << "[PrimalHeuristic] total "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << "s";
    }

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
                    if (first_gap.empty()) first_gap = state.var_infos[start_var].name() + domain.toString() + " required-min=" + std::to_string(required_min);
                }
                for (const auto &child : child_counts)
                {
                    if (child.second != alternatives) continue;
                    const auto it = state.start_vars[bucket].find(child.first);
                    if (it == state.start_vars[bucket].end()) continue;
                    if (state.domains[it->second].getMax() >= domain.getMax())
                    {
                        ++backward_gaps;
                        if (first_gap.empty()) first_gap = state.var_infos[it->second].name() + state.domains[it->second].toString() + " consumer=" + state.var_infos[start_var].name() + domain.toString();
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
