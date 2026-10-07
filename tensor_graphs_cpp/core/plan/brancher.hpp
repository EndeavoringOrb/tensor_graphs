#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/types.hpp"
#include "core/logging.hpp"
#include "core/plan/domain.hpp"
#include "core/plan/enode_info.hpp"
#include "core/plan/search_state.hpp"

namespace plan
{

struct BranchDecision
{
    std::pair<VarId, Domain> left_delta;
    std::pair<VarId, Domain> right_delta;
};

class Brancher
{
  public:
    virtual ~Brancher() = default;

    virtual bool chooseBranch(const SearchState &state, BranchDecision &out_decision) = 0;
};

class HeuristicBrancher : public Brancher
{
  private:
    static float getENodeScore(const SearchState &state, uint32_t bucket_idx, EClassId cid,
                               uint32_t enode_idx)
    {
        const EClass &cls = state.bucket_egraphs[bucket_idx].getEClass(cid);
        if (enode_idx >= cls.enodes.size())
            return TGConstants::INF;

        ENodeId enode_id = cls.enodes[enode_idx];
        if (enode_id.value >= state.bucket_enode_infos[bucket_idx].size())
            return 1.0f;

        const ENodeInfo &info = state.bucket_enode_infos[bucket_idx][enode_id.value];
        // Feasibility-first ordering, inspired by the old delegate: prefer
        // implementations with lower estimated memory pressure, then the
        // optimistic parallel DAG makespan.  Keep dp_cost as a fallback for
        // callers that construct ENodeInfo values without the new pass.
        float dag_cost = info.optimistic_dag_cost;
        if (dag_cost == TGConstants::INF)
            dag_cost = info.dp_cost;
        const float heuristic_cost = dag_cost < TGConstants::INF ? dag_cost : info.cost;
        if (info.dp_mem < TGConstants::INF)
            return info.dp_mem * 1.0e-6f + heuristic_cost * 1.0e-3f + info.cost;
        if (dag_cost < TGConstants::INF)
            return dag_cost;
        return info.cost;
    }

    int32_t preferredENode(const SearchState &state, uint32_t bucket_idx, EClassId cid,
                           const Domain &domain) const
    {
        const EClass &cls = state.bucket_egraphs[bucket_idx].getEClass(cid);
        int32_t best_value = domain.getMin();
        float best_score = TGConstants::INF;
        bool found_compatible = false;

        for (uint32_t enode_idx = 0; enode_idx < cls.enodes.size(); ++enode_idx)
        {
            int32_t value = static_cast<int32_t>(enode_idx + 1);
            if (!domain.contains(value))
                continue;

            if (!candidateIsTopologicallyCompatible(state, bucket_idx, cid, enode_idx))
                continue;

            float score = getENodeScore(state, bucket_idx, cid, enode_idx);
            if (!found_compatible || score < best_score)
            {
                best_score = score;
                best_value = value;
                found_compatible = true;
            }
        }

        // Keep an actual branch value if all alternatives are incompatible;
        // propagation will report the contradiction and preserve completeness.
        return best_value;
    }

    static Domain makeFixedLike(const Domain &domain, int32_t value)
    {
        return Domain::makeFixed(value, domain.is_mask);
    }

    static Domain without(const Domain &domain, int32_t value)
    {
        Domain result = domain;
        result.remove(value);
        return result;
    }

    // Selected edges point from a child e-class to its parent. Following
    // fixed selections therefore tells us whether adding a candidate edge
    // would close a precedence cycle.
    bool hasSelectedPath(const SearchState &state, uint32_t bucket_idx,
                         EClassId from, EClassId target) const
    {
        if (from == target)
            return true;

        const size_t num_classes = state.bucket_egraphs[bucket_idx].classes.size();
        if (path_visited_epoch.size() < num_classes)
            path_visited_epoch.resize(num_classes, 0);

        ++current_path_visited_epoch;
        if (current_path_visited_epoch == 0)
        {
            std::fill(path_visited_epoch.begin(), path_visited_epoch.end(), 0);
            current_path_visited_epoch = 1;
        }

        path_frontier.clear();
        path_frontier.push_back(from);
        path_visited_epoch[from.value] = current_path_visited_epoch;

        for (size_t i = 0; i < path_frontier.size(); ++i)
        {
            EClassId cid = path_frontier[i];
            if (cid == target)
                return true;

            auto sel_it = state.selected_vars[bucket_idx].find(cid);
            if (sel_it == state.selected_vars[bucket_idx].end())
                continue;

            const Domain &selection = state.domains[sel_it->second];
            if (!selection.isFixed() || selection.fixedValue() <= 0)
                continue;

            uint32_t enode_idx = static_cast<uint32_t>(selection.fixedValue() - 1);
            const EClass &cls = state.bucket_egraphs[bucket_idx].getEClass(cid);
            if (enode_idx >= cls.enodes.size())
                continue;

            const ENode &enode = state.bucket_egraphs[bucket_idx].getENode(cls.enodes[enode_idx]);
            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = state.bucket_egraphs[bucket_idx].findConst(child);
                if (path_visited_epoch[canon_child.value] != current_path_visited_epoch)
                {
                    path_visited_epoch[canon_child.value] = current_path_visited_epoch;
                    path_frontier.push_back(canon_child);
                }
            }
        }
        return false;
    }

    bool candidateIsTopologicallyCompatible(const SearchState &state, uint32_t bucket_idx,
                                            EClassId cid, uint32_t enode_idx) const
    {
        const EClass &cls = state.bucket_egraphs[bucket_idx].getEClass(cid);
        if (enode_idx >= cls.enodes.size())
            return false;

        const ENode &enode = state.bucket_egraphs[bucket_idx].getENode(cls.enodes[enode_idx]);
        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = state.bucket_egraphs[bucket_idx].findConst(child);
            if (canon_child == cid || hasSelectedPath(state, bucket_idx, canon_child, cid))
                return false;

            auto child_it = state.selected_vars[bucket_idx].find(canon_child);
            if (child_it == state.selected_vars[bucket_idx].end())
                return false;

            // This candidate will force every child to be selected.
            if (state.domains[child_it->second].getMax() <= 0)
                return false;
        }
        return true;
    }

    static bool setBinaryDecision(const Domain &domain, int32_t preferred,
                                  BranchDecision &out_decision, VarId var_id)
    {
        if (domain.isFixed() || domain.isEmpty() || !domain.contains(preferred))
            return false;

        Domain right = without(domain, preferred);
        if (right.isEmpty())
            return false;

        out_decision.left_delta = {var_id, makeFixedLike(domain, preferred)};
        out_decision.right_delta = {var_id, right};
        return true;
    }

    static EClassId resolveBaseClass(const SearchState &state, uint32_t b, EClassId cid)
    {
        EClassId curr = cid;
        uint32_t steps = 0;
        while (true)
        {
            if (++steps > 32)
            {
                Error::throw_err("resolveBaseClass exceeded maximum depth of 32 steps (view cycle detected)");
            }
            auto sel_it = state.selected_vars[b].find(curr);
            if (sel_it == state.selected_vars[b].end())
                break;
            const Domain &dom = state.domains[sel_it->second];
            if (!dom.isFixed() || dom.fixedValue() <= 0)
                break;
            uint32_t en_idx = static_cast<uint32_t>(dom.fixedValue() - 1);
            const EClass &cls = state.bucket_egraphs[b].getEClass(curr);
            if (en_idx >= cls.enodes.size())
                break;
            ENodeId en_id = cls.enodes[en_idx];
            bool is_view = en_id.value < state.bucket_enode_infos[b].size() &&
                           state.bucket_enode_infos[b][en_id.value].is_view;
            if (!is_view)
                break;
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
            if (enode.getChildren().empty())
                break;
            curr = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
        }
        return curr;
    }

    static bool isViewOf(const SearchState &state, uint32_t b, EClassId base, EClassId view_cand)
    {
        return resolveBaseClass(state, b, view_cand) == state.bucket_egraphs[b].findConst(base);
    }

    struct Obstacle
    {
        uint32_t start;
        uint32_t end;
    };

    // Reusable scratch members for HeuristicBrancher
    mutable std::vector<uint32_t> path_visited_epoch;
    mutable uint32_t current_path_visited_epoch = 1;
    mutable std::vector<EClassId> path_frontier;

    mutable std::vector<Obstacle> offset_obstacles;
    mutable std::vector<int32_t> offset_eclass_start;
    mutable std::vector<int32_t> offset_eclass_end;
    mutable std::vector<EClassId> offset_base_cids;
    mutable std::vector<uint8_t> offset_is_input_or_cache;

    mutable std::vector<EClassId> sched_active;
    mutable std::vector<uint32_t> sched_is_active_epoch;
    mutable uint32_t current_sched_active_epoch = 1;
    mutable std::vector<int> sched_in_degree;
    mutable std::vector<std::vector<EClassId>> sched_parents;
    mutable std::vector<EClassId> sched_queue;
    mutable std::vector<EClassId> sched_topo_order;
    mutable std::vector<uint32_t> sched_child_seen_epoch;
    mutable uint32_t current_sched_child_seen_epoch = 1;

    struct CachedOffsetCandidate
    {
        uint32_t bucket_idx;
        EClassId eclass_id;
        VarId cached_var;
        VarId offset_var;
    };
    mutable bool cached_offset_index_initialized = false;
    mutable std::vector<CachedOffsetCandidate> cached_offset_candidates;

#ifdef TG_PROFILE
    struct BrancherTiming
    {
        uint64_t total_calls = 0;
        uint64_t total_ns = 0;
        uint64_t selection_ns = 0;
        uint64_t cache_ns = 0;
        uint64_t sched_topo_ns = 0;
        uint64_t sched_start_ns = 0;
        uint64_t sched_offset_ns = 0;
        uint64_t preferred_offset_ns = 0;
        uint64_t preferred_offset_calls = 0;
    };
    mutable BrancherTiming brancher_timing;

#endif

    int32_t preferredOffset(const SearchState &state, uint32_t b, EClassId cid,
                            const Domain &offset_domain) const
    {
#ifdef TG_PROFILE
        auto pref_start = std::chrono::steady_clock::now();
        brancher_timing.preferred_offset_calls++;
#endif
        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        uint32_t psize = state.bytesToPages(getSizeBytes(cls.shape, cls.dtype), cls.mem_space);
        uint32_t curr_size = (psize == 0) ? 1 : psize;

        EClassId base_cid = resolveBaseClass(state, b, cid);
        if (base_cid != cid)
        {
            auto base_off_it = state.offset_vars[b].find(base_cid);
            if (base_off_it != state.offset_vars[b].end())
            {
                const Domain &base_dom = state.domains[base_off_it->second];
                if (base_dom.isFixed() && offset_domain.contains(base_dom.fixedValue()))
                {
#ifdef TG_PROFILE
                    brancher_timing.preferred_offset_ns += static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - pref_start).count());
#endif
                    return base_dom.fixedValue();
                }
            }
        }

        const size_t num_classes = state.bucket_egraphs[b].classes.size();
        if (offset_base_cids.size() < num_classes)
        {
            offset_base_cids.resize(num_classes);
            offset_is_input_or_cache.resize(num_classes, 0);
            offset_eclass_start.resize(num_classes, 0);
            offset_eclass_end.resize(num_classes, 0);
        }

        offset_obstacles.clear();
        for (const auto &pair : state.preallocated_buffers)
        {
            if (pair.second.mem_space == cls.mem_space)
            {
                uint32_t align = state.getPageAlignment(cls.mem_space);
                uint32_t start_p = static_cast<uint32_t>(pair.second.offset / align);
                uint32_t end_p = static_cast<uint32_t>((pair.second.offset + pair.second.size + align - 1) / align);
                offset_obstacles.push_back({start_p, end_p});
            }
        }

        // Pass 1: compute base_cid, start, and initial end for each candidate eclass
        for (const auto &pair : state.selected_vars[b])
        {
            EClassId cand = pair.first;
            EClassId cand_base = state.bucket_egraphs[b].findConst(resolveBaseClass(state, b, cand));
            offset_base_cids[cand.value] = cand_base;

            const EClass &c_cls = state.bucket_egraphs[b].getEClass(cand);
            bool is_input_or_cache = (c_cls.base_eclass_id != BaseEClassId{} && state.preallocated_buffers.count(c_cls.base_eclass_id));
            const Domain &sel_dom = state.domains[pair.second];
            uint32_t en_idx = (sel_dom.isFixed() && sel_dom.fixedValue() > 0)
                                  ? static_cast<uint32_t>(sel_dom.fixedValue() - 1)
                                  : 0;
            if (en_idx < c_cls.enodes.size())
            {
                const ENode &enode = state.bucket_egraphs[b].getENode(c_cls.enodes[en_idx]);
                if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
                    is_input_or_cache = true;
            }
            offset_is_input_or_cache[cand.value] = is_input_or_cache ? 1 : 0;

            bool is_root = (b < state.bucket_root_ids.size() &&
                            state.bucket_egraphs[b].findConst(cand) == state.bucket_egraphs[b].findConst(state.bucket_root_ids[b]));

            int32_t st_val = 0;
            if (!is_input_or_cache && !is_root)
            {
                auto st_it = state.start_vars[b].find(cand);
                if (st_it != state.start_vars[b].end())
                {
                    const Domain &st_dom = state.domains[st_it->second];
                    if (st_dom.isFixed())
                        st_val = st_dom.fixedValue();
                }
            }
            offset_eclass_start[cand.value] = st_val;
            offset_eclass_end[cand.value] = (is_input_or_cache || is_root)
                                               ? std::numeric_limits<int32_t>::max()
                                               : (st_val + 1);
        }

        // Pass 2: propagate reader finish times to the base class of their inputs
        for (const auto &pair : state.selected_vars[b])
        {
            EClassId r_cid = pair.first;
            const Domain &r_sel_dom = state.domains[pair.second];
            if (!r_sel_dom.isFixed() || r_sel_dom.fixedValue() <= 0)
                continue;

            uint32_t r_en = static_cast<uint32_t>(r_sel_dom.fixedValue() - 1);
            const EClass &r_cls = state.bucket_egraphs[b].getEClass(r_cid);
            if (r_en >= r_cls.enodes.size())
                continue;

            int32_t r_finish = 1;
            auto r_st_it = state.start_vars[b].find(r_cid);
            if (r_st_it != state.start_vars[b].end())
            {
                const Domain &r_st_dom = state.domains[r_st_it->second];
                r_finish = r_st_dom.isFixed() ? (r_st_dom.fixedValue() + 1) : (r_st_dom.getMax() + 1);
            }

            const ENode &r_enode = state.bucket_egraphs[b].getENode(r_cls.enodes[r_en]);
            for (EClassId ch : r_enode.getChildren())
            {
                EClassId canon_ch = state.bucket_egraphs[b].findConst(ch);
                auto it = state.selected_vars[b].find(canon_ch);
                EClassId base_ch = (it != state.selected_vars[b].end()) ? offset_base_cids[canon_ch.value] : canon_ch;
                if (base_ch != r_cid && offset_eclass_end[base_ch.value] != std::numeric_limits<int32_t>::max())
                {
                    offset_eclass_end[base_ch.value] = std::max(offset_eclass_end[base_ch.value], r_finish);
                }
            }
        }

        const bool is_curr_input_or_cache = offset_is_input_or_cache[cid.value];
        const int32_t curr_start = offset_eclass_start[cid.value];
        const EClassId canon_base_cid = offset_base_cids[cid.value];
        const int32_t curr_end = offset_eclass_end[canon_base_cid.value];

        // Pass 3: collect obstacles from fixed offset variables
        for (const auto &pair : state.selected_vars[b])
        {
            EClassId other_cid = pair.first;
            if (other_cid == cid)
                continue;

            auto off_it = state.offset_vars[b].find(other_cid);
            if (off_it == state.offset_vars[b].end())
                continue;

            const Domain &off_dom = state.domains[off_it->second];
            if (!off_dom.isFixed())
                continue;

            const Domain &other_sel_dom = state.domains[pair.second];
            if (other_sel_dom.isFixed() && other_sel_dom.fixedValue() == 0)
                continue;

            const EClass &other_cls = state.bucket_egraphs[b].getEClass(other_cid);
            if (other_cls.mem_space != cls.mem_space)
                continue;

            if (canon_base_cid == offset_base_cids[other_cid.value])
                continue;

            bool is_other_input_or_cache = offset_is_input_or_cache[other_cid.value];
            int32_t other_start = offset_eclass_start[other_cid.value];
            int32_t other_end = offset_eclass_end[other_cid.value];

            bool overlap = (is_curr_input_or_cache || is_other_input_or_cache ||
                            std::max(curr_start, other_start) < std::min(curr_end, other_end));
            if (!overlap)
                continue;

            uint32_t o_psize = state.bytesToPages(getSizeBytes(other_cls.shape, other_cls.dtype), other_cls.mem_space);
            uint32_t other_size = (o_psize == 0) ? 1 : o_psize;
            uint32_t o_offset = static_cast<uint32_t>(off_dom.fixedValue());
            offset_obstacles.push_back({o_offset, o_offset + other_size});
        }

        std::sort(offset_obstacles.begin(), offset_obstacles.end(), [](const Obstacle &a, const Obstacle &b) {
            return a.start < b.start;
        });

        uint32_t p = static_cast<uint32_t>(offset_domain.getMin());
        bool pushed = true;
        while (pushed)
        {
            pushed = false;
            for (const auto &obs : offset_obstacles)
            {
                if (p < obs.end && p + curr_size > obs.start)
                {
                    p = obs.end;
                    pushed = true;
                }
            }
        }

        int32_t result = (p <= static_cast<uint32_t>(offset_domain.getMax()))
                             ? static_cast<int32_t>(p)
                             : offset_domain.getMin();
#ifdef TG_PROFILE
        brancher_timing.preferred_offset_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - pref_start).count());
#endif
        return result;
    }

    static bool setOffsetDecision(const Domain &domain, int32_t preferred,
                                  BranchDecision &out_decision, VarId var_id)
    {
        if (domain.isFixed() || domain.isEmpty() || !domain.contains(preferred))
            return false;

        out_decision.left_delta = {var_id, Domain::makeFixed(preferred, false)};
        if (preferred + 1 <= domain.getMax())
        {
            out_decision.right_delta = {var_id, Domain::makeRange(preferred + 1, domain.getMax())};
        }
        else
        {
            Domain right = without(domain, preferred);
            out_decision.right_delta = {var_id, right};
        }
        return true;
    }

    bool chooseSelection(const SearchState &state, BranchDecision &out_decision,
                         bool required_only) const
    {
        static const std::vector<EClassId> empty_cids;
        VarId best_var = kInvalidVarId;
        uint32_t best_bucket = 0;
        EClassId best_cid;
        int32_t best_size = std::numeric_limits<int32_t>::max();

        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            const auto &cids = (b < state.reachable_cids.size()) ? state.reachable_cids[b] : empty_cids;
            for (EClassId cid : cids)
            {
                auto sel_it = state.selected_vars[b].find(cid);
                if (sel_it == state.selected_vars[b].end())
                    continue;

                VarId var_id = sel_it->second;
                const Domain &domain = state.domains[var_id];
                if (domain.isFixed() || domain.isEmpty())
                    continue;

                const bool required = !domain.contains(0);
                if (required != required_only)
                    continue;

                if (domain.size() < best_size)
                {
                    best_var = var_id;
                    best_bucket = b;
                    best_cid = cid;
                    best_size = domain.size();
                }
            }
        }

        if (best_var == kInvalidVarId)
            return false;

        const Domain &domain = state.domains[best_var];
        int32_t preferred = required_only ? preferredENode(state, best_bucket, best_cid, domain) : 0;
        return setBinaryDecision(domain, preferred, out_decision, best_var);
    }

    bool chooseCache(const SearchState &state, BranchDecision &out_decision) const
    {
        for (const CacheCandidate &candidate : state.candidates)
        {
            auto it = state.cached_vars.find(candidate.base_eclass_id);
            if (it == state.cached_vars.end())
                continue;

            VarId var_id = it->second;
            const Domain &domain = state.domains[var_id];
            if (domain.isFixed() || domain.isEmpty())
                continue;

            return setBinaryDecision(domain, 0, out_decision, var_id);
        }
        return false;
    }

    bool chooseCachedOffset(const SearchState &state, BranchDecision &out_decision) const
    {
        if (!cached_offset_index_initialized)
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &offset_pair : state.offset_vars[b])
                {
                    const EClass &cls = state.bucket_egraphs[b].getEClass(offset_pair.first);
                    if (state.preallocated_buffers.count(cls.base_eclass_id))
                        continue;
                    auto cached_it = state.cached_vars.find(cls.base_eclass_id);
                    auto selected_it = state.selected_vars[b].find(offset_pair.first);
                    if (cached_it == state.cached_vars.end() || selected_it == state.selected_vars[b].end())
                        continue;
                    cached_offset_candidates.push_back(
                        {b, offset_pair.first, cached_it->second, offset_pair.second});
                }
            }
            cached_offset_index_initialized = true;
        }

        for (uint32_t pass = 0; pass < 2; ++pass)
        {
            for (const CachedOffsetCandidate &candidate : cached_offset_candidates)
            {
                const Domain &cached = state.domains[candidate.cached_var];
                const Domain &selected = state.domains[state.selected_vars[candidate.bucket_idx].at(candidate.eclass_id)];
                const Domain &offset = state.domains[candidate.offset_var];
                if (!cached.isFixed() || cached.fixedValue() != 1 || !selected.isFixed() ||
                    selected.fixedValue() <= 0 || offset.isFixed() || offset.isEmpty())
                    continue;

                const EClass &cls = state.bucket_egraphs[candidate.bucket_idx].getEClass(candidate.eclass_id);
                const ENode &selected_enode = state.bucket_egraphs[candidate.bucket_idx].getENode(
                    cls.enodes[static_cast<uint32_t>(selected.fixedValue() - 1)]);
                const bool selected_cache = selected_enode.getOpType() == OpType::CACHE;
                if (selected_cache != (pass == 0))
                    continue;

                const int32_t preferred = preferredOffset(state, candidate.bucket_idx, candidate.eclass_id, offset);
                return setOffsetDecision(offset, preferred, out_decision, candidate.offset_var);
            }
        }
        return false;
    }

    bool chooseSchedule(const SearchState &state, BranchDecision &out_decision) const
    {
        static const std::vector<EClassId> empty_cids;
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
#ifdef TG_PROFILE
            auto topo_start = std::chrono::steady_clock::now();
#endif
            const size_t num_classes = state.bucket_egraphs[b].classes.size();
            if (sched_is_active_epoch.size() < num_classes)
            {
                sched_is_active_epoch.resize(num_classes, 0);
                sched_in_degree.resize(num_classes, 0);
                sched_parents.resize(num_classes);
                sched_child_seen_epoch.resize(num_classes, 0);
            }

            sched_active.clear();
            ++current_sched_active_epoch;
            if (current_sched_active_epoch == 0)
            {
                std::fill(sched_is_active_epoch.begin(), sched_is_active_epoch.end(), 0);
                current_sched_active_epoch = 1;
            }

            const auto &cids = (b < state.reachable_cids.size()) ? state.reachable_cids[b] : empty_cids;
            for (EClassId cid : cids)
            {
                auto sel_it = state.selected_vars[b].find(cid);
                if (sel_it == state.selected_vars[b].end())
                    continue;

                const Domain &selection = state.domains[sel_it->second];
                if (!selection.isFixed() || selection.fixedValue() <= 0)
                    continue;

                sched_active.push_back(cid);
                sched_is_active_epoch[cid.value] = current_sched_active_epoch;
                sched_in_degree[cid.value] = 0;
                sched_parents[cid.value].clear();
            }

            for (EClassId cid : sched_active)
            {
                const Domain &selection = state.domains[state.selected_vars[b].at(cid)];
                uint32_t enode_idx = static_cast<uint32_t>(selection.fixedValue() - 1);
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (enode_idx >= cls.enodes.size())
                    continue;

                ++current_sched_child_seen_epoch;
                if (current_sched_child_seen_epoch == 0)
                {
                    std::fill(sched_child_seen_epoch.begin(), sched_child_seen_epoch.end(), 0);
                    current_sched_child_seen_epoch = 1;
                }

                const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[enode_idx]);
                for (EClassId child : enode.getChildren())
                {
                    EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                    if (sched_is_active_epoch[canon_child.value] == current_sched_active_epoch &&
                        sched_child_seen_epoch[canon_child.value] != current_sched_child_seen_epoch)
                    {
                        sched_child_seen_epoch[canon_child.value] = current_sched_child_seen_epoch;
                        ++sched_in_degree[cid.value];
                        sched_parents[canon_child.value].push_back(cid);
                    }
                }
            }

            sched_queue.clear();
            for (EClassId cid : sched_active)
            {
                if (sched_in_degree[cid.value] == 0)
                    sched_queue.push_back(cid);
            }

            sched_topo_order.clear();
            size_t queue_head = 0;
            while (queue_head < sched_queue.size())
            {
                EClassId child = sched_queue[queue_head++];
                sched_topo_order.push_back(child);
                for (EClassId parent : sched_parents[child.value])
                {
                    if (--sched_in_degree[parent.value] == 0)
                        sched_queue.push_back(parent);
                }
            }

#ifdef TG_PROFILE
            brancher_timing.sched_topo_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - topo_start).count());
#endif

            if (sched_topo_order.size() != sched_active.size())
                continue;

#ifdef TG_PROFILE
            auto start_vars_begin = std::chrono::steady_clock::now();
#endif
            // Assign starts in dependency order: children first, then their
            // consumers. This makes the preferred branch a valid topological
            // scheduling attempt instead of a split chosen by domain size.
            for (EClassId cid : sched_topo_order)
            {
                auto start_it = state.start_vars[b].find(cid);
                if (start_it == state.start_vars[b].end())
                    continue;

                VarId start_var = start_it->second;
                const Domain &start_domain = state.domains[start_var];
                if (!start_domain.isFixed())
                {
#ifdef TG_PROFILE
                    brancher_timing.sched_start_ns += static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start_vars_begin).count());
#endif
                    return setBinaryDecision(start_domain, start_domain.getMin(), out_decision, start_var);
                }
            }
#ifdef TG_PROFILE
            brancher_timing.sched_start_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start_vars_begin).count());
            auto offset_vars_begin = std::chrono::steady_clock::now();
#endif

            for (EClassId cid : sched_topo_order)
            {
                auto offset_it = state.offset_vars[b].find(cid);
                if (offset_it != state.offset_vars[b].end())
                {
                    VarId offset_var = offset_it->second;
                    const Domain &offset_domain = state.domains[offset_var];
                    if (!offset_domain.isFixed())
                    {
                        int32_t preferred = preferredOffset(state, b, cid, offset_domain);
#ifdef TG_PROFILE
                        brancher_timing.sched_offset_ns += static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - offset_vars_begin).count());
#endif
                        return setOffsetDecision(offset_domain, preferred, out_decision, offset_var);
                    }
                }
            }
#ifdef TG_PROFILE
            brancher_timing.sched_offset_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - offset_vars_begin).count());
#endif
        }
        return false;
    }

  public:
    bool chooseBranch(const SearchState &state, BranchDecision &out_decision) override
    {
#ifdef TG_PROFILE
        auto total_start = std::chrono::steady_clock::now();
        brancher_timing.total_calls++;
#endif
        out_decision.left_delta = {kInvalidVarId, Domain{}};
        out_decision.right_delta = {kInvalidVarId, Domain{}};

#ifdef TG_PROFILE
        auto sel_start = std::chrono::steady_clock::now();
#endif
        // First satisfy selections forced by the currently selected DAG.
        if (chooseSelection(state, out_decision, true))
        {
#ifdef TG_PROFILE
            auto now = std::chrono::steady_clock::now();
            brancher_timing.selection_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - sel_start).count());
            brancher_timing.total_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - total_start).count());
#endif
            return true;
        }
#ifdef TG_PROFILE
        brancher_timing.selection_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - sel_start).count());
        auto cache_start = std::chrono::steady_clock::now();
#endif

        if (chooseCache(state, out_decision))
        {
#ifdef TG_PROFILE
            auto now = std::chrono::steady_clock::now();
            brancher_timing.cache_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - cache_start).count());
            brancher_timing.total_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - total_start).count());
#endif
            return true;
        }
#ifdef TG_PROFILE
        brancher_timing.cache_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - cache_start).count());
#endif

        // Schedule and allocate one active operation at a time. This keeps
        // start/offset variables in the same hyperbox search instead of
        // manufacturing a complete heuristic assignment in one delta.
        // Place persistent cache buffers first so CachedOffsetPropagator can
        // immediately share their addresses with every bucket copy.
        if (chooseCachedOffset(state, out_decision))
        {
#ifdef TG_PROFILE
            brancher_timing.total_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - total_start).count());
#endif
            return true;
        }
        if (chooseSchedule(state, out_decision))
        {
#ifdef TG_PROFILE
            brancher_timing.total_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - total_start).count());
#endif
            return true;
        }

#ifdef TG_PROFILE
        auto opt_sel_start = std::chrono::steady_clock::now();
#endif
        // Only after the currently selected DAG has a cache/schedule attempt
        // do we branch optional support. The branch remains complete, but a
        // promising selected DAG can now reach a full feasible leaf without
        // first fixing every potentially reachable e-class in the e-graph.
        if (chooseSelection(state, out_decision, false))
        {
#ifdef TG_PROFILE
            auto now = std::chrono::steady_clock::now();
            brancher_timing.selection_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - opt_sel_start).count());
            brancher_timing.total_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - total_start).count());
#endif
            return true;
        }
#ifdef TG_PROFILE
        auto now = std::chrono::steady_clock::now();
        brancher_timing.selection_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - opt_sel_start).count());
        brancher_timing.total_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - total_start).count());
#endif

        LOG(DEBUG) << "[HeuristicBrancher] No unfixed relevant variable remains; leaf reached.";
        return false;
    }
};

} // namespace plan
