// tensor_graphs_cpp/core/plan/brancher.hpp
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/logging.hpp"
#include "core/plan/domain.hpp"
#include "core/plan/enode_info.hpp"
#include "core/plan/search_state.hpp"

namespace plan
{

struct BranchDecision
{
    std::vector<std::pair<VarId, Domain>> left_delta;
    std::vector<std::pair<VarId, Domain>> right_delta;
};

class Brancher
{
  public:
    virtual ~Brancher() = default;

    virtual bool chooseBranch(const SearchState &state, BranchDecision &out_decision) = 0;
};

class HeuristicBrancher : public Brancher
{
  public:
    bool chooseBranch(const SearchState &state, BranchDecision &out_decision) override
    {
        out_decision.left_delta.clear();
        out_decision.right_delta.clear();

        // 1. Branch on REQUIRED selected enodes first (cannot be 0, starting from root)
        VarId best_req_var = kInvalidVarId;
        int32_t best_req_size = std::numeric_limits<int32_t>::max();
        int32_t preferred_enode_val = -1;

        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            const auto &cids = (b < state.reachable_cids.size() && !state.reachable_cids[b].empty())
                                   ? state.reachable_cids[b]
                                   : std::vector<EClassId>{};
            for (EClassId cid : cids)
            {
                VarId v = state.selected_vars[b].at(cid);
                const Domain &dom = state.domains[v];
                if (!dom.isFixed() && !dom.isEmpty() && !dom.contains(0))
                {
                    if (dom.size() < best_req_size)
                    {
                        best_req_size = dom.size();
                        best_req_var = v;

                        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                        float min_c = std::numeric_limits<float>::infinity();
                        int32_t best_idx = -1;
                        for (uint32_t e_idx = 0; e_idx < cls.enodes.size(); ++e_idx)
                        {
                            int32_t val = static_cast<int32_t>(e_idx + 1);
                            if (dom.contains(val))
                            {
                                ENodeId en_id = cls.enodes[e_idx];
                                float c = 1.0f;
                                if (en_id.value < state.bucket_enode_infos[b].size())
                                    c = state.bucket_enode_infos[b][en_id.value].cost;
                                if (c < min_c)
                                {
                                    min_c = c;
                                    best_idx = val;
                                }
                            }
                        }
                        preferred_enode_val = (best_idx != -1) ? best_idx : dom.getMin();
                    }
                }
            }
        }

        if (best_req_var != kInvalidVarId)
        {
            out_decision.left_delta.push_back({best_req_var, Domain::makeFixed(preferred_enode_val, true)});
            Domain right = state.domains[best_req_var];
            right.remove(preferred_enode_val);
            out_decision.right_delta.push_back({best_req_var, right});
            return true;
        }

        // 2. Branch on cache candidates (try uncached 0 first for greedy feasibility, 1 on right)
        for (const auto &cand : state.candidates)
        {
            auto it = state.cached_vars.find(cand.base_eclass_id);
            if (it != state.cached_vars.end())
            {
                VarId v = it->second;
                const Domain &dom = state.domains[v];
                if (!dom.isFixed() && !dom.isEmpty())
                {
                    out_decision.left_delta.push_back({v, Domain::makeFixed(0, true)});
                    out_decision.right_delta.push_back({v, Domain::makeFixed(1, true)});
                    return true;
                }
            }
        }

        // 3. Check if any start_var or offset_var is not yet fixed for active eclasses
        bool any_unfixed_schedule = false;
        for (uint32_t b = 0; b < state.buckets.size() && !any_unfixed_schedule; ++b)
        {
            const auto &cids = (b < state.reachable_cids.size() && !state.reachable_cids[b].empty())
                                   ? state.reachable_cids[b]
                                   : std::vector<EClassId>{};
            for (EClassId cid : cids)
            {
                VarId sel_v = state.selected_vars[b].at(cid);
                if (state.domains[sel_v].isFixed() && state.domains[sel_v].fixedValue() > 0)
                {
                    uint32_t en_idx = static_cast<uint32_t>(state.domains[sel_v].fixedValue() - 1);
                    VarId st_v = state.start_vars[b].at(cid)[en_idx];
                    if (!state.domains[st_v].isFixed())
                    {
                        any_unfixed_schedule = true;
                        break;
                    }
                    auto off_it = state.offset_vars[b].find(cid);
                    if (off_it != state.offset_vars[b].end() && !state.domains[off_it->second].isFixed())
                    {
                        any_unfixed_schedule = true;
                        break;
                    }
                }
            }
        }

        if (!any_unfixed_schedule)
        {
            LOG(DEBUG) << "[HeuristicBrancher] All variables are fixed, returning false (leaf reached).";
            return false; // Leaf reached: all variables are fixed!
        }

        LOG(DEBUG) << "[HeuristicBrancher] Starting bulk scheduling & allocation...";

        // 4. Perform bulk scheduling (dispatch order) and memory offset allocation (first-fit)
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            const auto &cids = (b < state.reachable_cids.size() && !state.reachable_cids[b].empty())
                                   ? state.reachable_cids[b]
                                   : std::vector<EClassId>{};

            std::vector<EClassId> active_cids;
            std::unordered_set<EClassId> active_set;
            for (EClassId cid : cids)
            {
                VarId sel_v = state.selected_vars[b].at(cid);
                if (state.domains[sel_v].isFixed() && state.domains[sel_v].fixedValue() > 0)
                {
                    active_cids.push_back(cid);
                    active_set.insert(cid);
                }
            }
            LOG(DEBUG) << "[HeuristicBrancher] Bucket " << b << ": active_cids count=" << active_cids.size();

            // Topological sort of active eclasses (inputs before consumers)
            std::unordered_map<EClassId, int> in_degree;
            std::unordered_map<EClassId, std::vector<EClassId>> active_parents; // child -> parents

            for (EClassId cid : active_cids)
            {
                in_degree[cid] = 0;
            }

            for (EClassId cid : active_cids)
            {
                uint32_t en_idx = static_cast<uint32_t>(state.domains[state.selected_vars[b].at(cid)].fixedValue() - 1);
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (en_idx < cls.enodes.size())
                {
                    const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                    std::unordered_set<EClassId> unique_children;
                    for (EClassId ch : enode.getChildren())
                    {
                        EClassId canon_ch = state.bucket_egraphs[b].findConst(ch);
                        if (active_set.count(canon_ch) && canon_ch != cid)
                        {
                            unique_children.insert(canon_ch);
                        }
                    }
                    in_degree[cid] = static_cast<int>(unique_children.size());
                    for (EClassId canon_ch : unique_children)
                    {
                        active_parents[canon_ch].push_back(cid);
                    }
                }
            }

            std::vector<EClassId> active_topo_order;
            std::vector<EClassId> q;
            for (EClassId cid : active_cids)
            {
                if (in_degree[cid] == 0)
                {
                    q.push_back(cid);
                }
            }

            size_t head = 0;
            while (head < q.size())
            {
                EClassId curr = q[head++];
                active_topo_order.push_back(curr);

                auto it = active_parents.find(curr);
                if (it != active_parents.end())
                {
                    for (EClassId parent : it->second)
                    {
                        if (--in_degree[parent] == 0)
                        {
                            q.push_back(parent);
                        }
                    }
                }
            }

            // In case of any unvisited active node (e.g. subtle cycle)
            if (active_topo_order.size() < active_cids.size())
            {
                for (EClassId cid : active_cids)
                {
                    if (in_degree[cid] > 0)
                    {
                        active_topo_order.push_back(cid);
                    }
                }
            }

            LOG(DEBUG) << "[HeuristicBrancher] active_topo_order count=" << active_topo_order.size();

            // Schedule start times
            std::unordered_map<Engine, int32_t> engine_next_free_time;
            std::unordered_map<EClassId, int32_t> start_times;

            for (EClassId cid : active_topo_order)
            {
                uint32_t en_idx = static_cast<uint32_t>(state.domains[state.selected_vars[b].at(cid)].fixedValue() - 1);
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                ENodeId en_id = cls.enodes[en_idx];
                const ENode &enode = state.bucket_egraphs[b].getENode(en_id);

                int32_t earliest_st = 0;
                for (EClassId ch : enode.getChildren())
                {
                    EClassId canon_ch = state.bucket_egraphs[b].findConst(ch);
                    auto it = start_times.find(canon_ch);
                    if (it != start_times.end())
                    {
                        earliest_st = std::max(earliest_st, it->second + 1);
                    }
                }

                int32_t t = earliest_st;
                for (const Engine &eng : enode.getEngines())
                {
                    auto it = engine_next_free_time.find(eng);
                    if (it != engine_next_free_time.end())
                    {
                        t = std::max(t, it->second);
                    }
                }

                for (const Engine &eng : enode.getEngines())
                {
                    engine_next_free_time[eng] = t + 1;
                }

                start_times[cid] = t;
                VarId st_v = state.start_vars[b].at(cid)[en_idx];
                out_decision.left_delta.push_back({st_v, Domain::makeFixed(t)});

                for (uint32_t e = 0; e < state.start_vars[b].at(cid).size(); ++e)
                {
                    if (e != en_idx)
                    {
                        out_decision.left_delta.push_back({state.start_vars[b].at(cid)[e], Domain::makeFixed(0)});
                    }
                }
            }
            LOG(DEBUG) << "[HeuristicBrancher] Scheduled start times for " << active_topo_order.size() << " active ops.";

            // Build selection map for view resolution
            std::unordered_map<EClassId, uint32_t> selection_map;
            for (EClassId cid : active_cids)
            {
                selection_map[cid] = static_cast<uint32_t>(state.domains[state.selected_vars[b].at(cid)].fixedValue() - 1);
            }

            // Find all consumers for each base allocation
            std::unordered_map<EClassId, std::unordered_set<EClassId>> base_consumers;
            for (EClassId cid : active_topo_order)
            {
                EClassId base_cid = resolve_view_alias(cid, state.bucket_egraphs[b], selection_map, state.bucket_enode_infos[b]);
                if (base_cid == EClassId{UINT32_MAX})
                    base_cid = cid;
                for (EClassId p_cid : active_parents[cid])
                {
                    base_consumers[base_cid].insert(p_cid);
                }
            }

            int32_t max_finish_time = 0;
            for (const auto &pair : start_times)
            {
                max_finish_time = std::max(max_finish_time, pair.second + 1);
            }

            // Allocate memory offsets (First-Fit Pages)
            struct PlacedBuffer
            {
                MemSpace ms;
                int32_t start_time;
                int32_t end_time;
                uint32_t page_offset;
                uint32_t page_size;
            };
            std::vector<PlacedBuffer> placed_buffers;
            std::unordered_map<EClassId, uint32_t> placed_offsets;

            // Pass 1: non-views
            for (EClassId cid : active_topo_order)
            {
                auto off_it = state.offset_vars[b].find(cid);
                if (off_it == state.offset_vars[b].end())
                    continue;

                uint32_t en_idx = selection_map[cid];
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                ENodeId en_id = cls.enodes[en_idx];
                bool is_view = (en_id.value < state.bucket_enode_infos[b].size()) &&
                               state.bucket_enode_infos[b][en_id.value].is_view;

                if (is_view)
                    continue;

                int32_t st = start_times[cid];
                int32_t end_t = st + 1;
                auto bc_it = base_consumers.find(cid);
                if (bc_it != base_consumers.end())
                {
                    for (EClassId consumer_cid : bc_it->second)
                    {
                        auto st_it = start_times.find(consumer_cid);
                        if (st_it != start_times.end())
                        {
                            end_t = std::max(end_t, st_it->second + 1);
                        }
                    }
                }
                else
                {
                    // No consumer (e.g. output): keep alive until end of graph
                    end_t = std::max(end_t, max_finish_time);
                }

                uint32_t p_size = std::max(1u, state.bytesToPages(getSizeBytes(cls.shape, cls.dtype), cls.mem_space));

                std::vector<std::pair<uint32_t, uint32_t>> conflicts;
                for (const auto &pl : placed_buffers)
                {
                    if (pl.ms == cls.mem_space)
                    {
                        bool time_overlap = !(end_t <= pl.start_time || pl.end_time <= st);
                        if (time_overlap)
                        {
                            conflicts.push_back({pl.page_offset, pl.page_offset + pl.page_size});
                        }
                    }
                }
                std::sort(conflicts.begin(), conflicts.end());

                std::vector<std::pair<uint32_t, uint32_t>> merged;
                for (const auto &c : conflicts)
                {
                    if (merged.empty() || merged.back().second < c.first)
                    {
                        merged.push_back(c);
                    }
                    else
                    {
                        merged.back().second = std::max(merged.back().second, c.second);
                    }
                }

                uint32_t min_p = state.preallocated_pages.count(cls.mem_space)
                                     ? state.preallocated_pages.at(cls.mem_space)
                                     : 0;
                uint32_t p = min_p;
                for (const auto &c : merged)
                {
                    if (p + p_size <= c.first)
                    {
                        break;
                    }
                    if (p < c.second)
                    {
                        p = std::max(p, c.second);
                    }
                }

                placed_buffers.push_back({cls.mem_space, st, end_t, p, p_size});
                placed_offsets[cid] = p;
                out_decision.left_delta.push_back({off_it->second, Domain::makeFixed(static_cast<int32_t>(p))});
            }

            // Pass 2: views
            for (EClassId cid : active_topo_order)
            {
                auto off_it = state.offset_vars[b].find(cid);
                if (off_it == state.offset_vars[b].end())
                    continue;

                uint32_t en_idx = selection_map[cid];
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                ENodeId en_id = cls.enodes[en_idx];
                bool is_view = (en_id.value < state.bucket_enode_infos[b].size()) &&
                               state.bucket_enode_infos[b][en_id.value].is_view;

                if (!is_view)
                    continue;

                EClassId base_cid = resolve_view_alias(cid, state.bucket_egraphs[b], selection_map, state.bucket_enode_infos[b]);
                uint32_t p = 0;
                auto it = placed_offsets.find(base_cid);
                if (it != placed_offsets.end())
                {
                    p = it->second;
                }
                placed_offsets[cid] = p;
                out_decision.left_delta.push_back({off_it->second, Domain::makeFixed(static_cast<int32_t>(p))});
            }

            // Inactive eclasses in reachable_cids
            for (EClassId cid : cids)
            {
                if (!active_set.count(cid))
                {
                    auto st_it = state.start_vars[b].find(cid);
                    if (st_it != state.start_vars[b].end())
                    {
                        for (VarId st_v : st_it->second)
                        {
                            out_decision.left_delta.push_back({st_v, Domain::makeFixed(0)});
                        }
                    }
                    auto off_it = state.offset_vars[b].find(cid);
                    if (off_it != state.offset_vars[b].end())
                    {
                        out_decision.left_delta.push_back({off_it->second, Domain::makeFixed(0)});
                    }
                }
            }
            LOG(DEBUG) << "[HeuristicBrancher] Bucket " << b << " memory allocation completed.";
        }

        LOG(DEBUG) << "[HeuristicBrancher] Bulk scheduling & allocation completed, left_delta size=" << out_decision.left_delta.size();
        out_decision.right_delta.clear();
        return true;
    }
};

} // namespace plan
