// tensor_graphs_cpp/core/plan/brancher.hpp
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "core/plan/domain.hpp"
#include "core/plan/search_state.hpp"

namespace plan
{

class Brancher
{
  public:
    virtual ~Brancher() = default;

    virtual bool chooseBranch(const SearchState &state, VarId &out_var, Domain &out_left_domain,
                              Domain &out_right_domain) = 0;
};

class HeuristicBrancher : public Brancher
{
  public:
    bool chooseBranch(const SearchState &state, VarId &out_var, Domain &out_left_domain,
                      Domain &out_right_domain) override
    {
        // 1. Branch on REQUIRED selected enodes first (cannot be 0, starting from root)
        VarId best_req_var = kInvalidVarId;
        int32_t best_req_size = std::numeric_limits<int32_t>::max();
        int32_t preferred_enode_val = -1;

        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &pair : state.selected_vars[b])
            {
                VarId v = pair.second;
                const Domain &dom = state.domains[v];
                if (!dom.isFixed() && !dom.isEmpty() && !dom.contains(0))
                {
                    if (dom.size() < best_req_size)
                    {
                        best_req_size = dom.size();
                        best_req_var = v;

                        EClassId cid = pair.first;
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
            out_var = best_req_var;
            const Domain &dom = state.domains[best_req_var];
            out_left_domain = Domain::makeFixed(preferred_enode_val, true);
            Domain right = dom;
            right.remove(preferred_enode_val);
            out_right_domain = right;
            return true;
        }

        // 2. Branch on remaining unfixed selected variables (prefer 0 = unselected)
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &pair : state.selected_vars[b])
            {
                VarId v = pair.second;
                const Domain &dom = state.domains[v];
                if (!dom.isFixed() && !dom.isEmpty())
                {
                    out_var = v;
                    if (dom.contains(0))
                    {
                        out_left_domain = Domain::makeFixed(0, true);
                        Domain right = dom;
                        right.remove(0);
                        out_right_domain = right;
                    }
                    else
                    {
                        out_left_domain = Domain::makeFixed(dom.getMin(), true);
                        Domain right = dom;
                        right.remove(dom.getMin());
                        out_right_domain = right;
                    }
                    return true;
                }
            }
        }

        // 3. Branch on cache candidates (try uncached 0 first for greedy feasibility, 1 on right)
        for (const auto &cand : state.candidates)
        {
            auto it = state.cached_vars.find(cand.base_eclass_id);
            if (it != state.cached_vars.end())
            {
                VarId v = it->second;
                const Domain &dom = state.domains[v];
                if (!dom.isFixed() && !dom.isEmpty())
                {
                    out_var = v;
                    out_left_domain = Domain::makeFixed(0, true);
                    out_right_domain = Domain::makeFixed(1, true);
                    return true;
                }
            }
        }

        // 4. Branch on dispatch start variables for selected enodes
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &pair : state.start_vars[b])
            {
                EClassId cid = pair.first;
                VarId sel_v = state.selected_vars[b].at(cid);
                if (state.domains[sel_v].isFixed())
                {
                    int32_t sel_val = state.domains[sel_v].fixedValue();
                    if (sel_val > 0)
                    {
                        uint32_t en_idx = static_cast<uint32_t>(sel_val - 1);
                        if (en_idx < pair.second.size())
                        {
                            VarId st_v = pair.second[en_idx];
                            const Domain &dom = state.domains[st_v];
                            if (!dom.isFixed() && !dom.isEmpty())
                            {
                                out_var = st_v;
                                int32_t min_v = dom.getMin();
                                out_left_domain = Domain::makeFixed(min_v, true);
                                out_right_domain = Domain::makeRange(min_v + 1, dom.getMax());
                                return true;
                            }
                        }
                    }
                }
            }
        }

        // 5. Branch on memory offset variables for selected eclasses
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &pair : state.offset_vars[b])
            {
                EClassId cid = pair.first;
                VarId sel_v = state.selected_vars[b].at(cid);
                if (state.domains[sel_v].isFixed() && state.domains[sel_v].fixedValue() > 0)
                {
                    VarId off_v = pair.second;
                    const Domain &dom = state.domains[off_v];
                    if (!dom.isFixed() && !dom.isEmpty())
                    {
                        out_var = off_v;
                        int32_t min_v = dom.getMin();
                        out_left_domain = Domain::makeFixed(min_v, true);
                        out_right_domain = Domain::makeRange(min_v + 1, dom.getMax());
                        return true;
                    }
                }
            }
        }

        return false;
    }
};

} // namespace plan
