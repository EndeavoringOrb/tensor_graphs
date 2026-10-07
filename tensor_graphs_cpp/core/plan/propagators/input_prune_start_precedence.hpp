// tensor_graphs_cpp/core/plan/propagators/input_prune_start_precedence.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// InputPruneStartPrecedencePropagator: see docs/core/propagators.md.
class InputPruneStartPrecedencePropagator : public Propagator
{
    bool pruneEClass(SearchState &state, uint32_t b, EClassId cid, std::vector<VarId> &worklist)
    {
        auto sel_it = state.selected_vars[b].find(cid);
        if (sel_it == state.selected_vars[b].end())
            return true;
        VarId sel_v = sel_it->second;
        Domain sel_dom = state.domains[sel_v];
        if (sel_dom.contains(0))
            return true;

        auto st_it = state.start_vars[b].find(cid);
        if (st_it == state.start_vars[b].end())
            return true;
        VarId st_var = st_it->second;

        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        bool sel_modified = false;

        for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
        {
            int32_t cand_val = static_cast<int32_t>(en_idx + 1);
            if (!sel_dom.contains(cand_val))
                continue;
            int32_t cand_start_max = state.domains[st_var].getMax();

            ENodeId en_id = cls.enodes[en_idx];
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
            bool remove_cand = false;

            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                auto ch_sel_it = state.selected_vars[b].find(canon_child);
                auto ch_st_it = state.start_vars[b].find(canon_child);
                if (ch_sel_it == state.selected_vars[b].end() || ch_st_it == state.start_vars[b].end())
                    continue;

                const Domain &ch_sel_dom = state.domains[ch_sel_it->second];
                if (ch_sel_dom.isFixed() && ch_sel_dom.fixedValue() == 0)
                    continue;
                int32_t input_start_min = state.domains[ch_st_it->second].getMin();
                if (input_start_min != INT32_MAX && input_start_min >= cand_start_max)
                {
                    remove_cand = true;
                    break;
                }
            }

            if (remove_cand)
            {
                if (sel_dom.isFixed())
                    return false;
                sel_dom.remove(cand_val);
                sel_modified = true;
                if (sel_dom.isEmpty())
                    return false;
            }
        }

        if (sel_modified)
        {
            state.setDomain(sel_v, sel_dom);
            worklist.push_back(sel_v);
        }

        int32_t min_start_max_C = INT32_MAX;
        std::unordered_map<EClassId, uint32_t> input_occurrence_count;
        uint32_t num_remaining_cands = 0;

        for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
        {
            int32_t cand_val = static_cast<int32_t>(en_idx + 1);
            if (!sel_dom.contains(cand_val))
                continue;
            ++num_remaining_cands;
            min_start_max_C = std::min(min_start_max_C, state.domains[st_var].getMax());

            ENodeId en_id = cls.enodes[en_idx];
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
            std::unordered_set<EClassId> seen_children;
            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                if (seen_children.insert(canon_child).second)
                {
                    input_occurrence_count[canon_child]++;
                }
            }
        }

        if (num_remaining_cands > 0 && min_start_max_C != INT32_MAX)
        {
            int32_t max_allowed_input_start = min_start_max_C - 1;
            for (const auto &pair : input_occurrence_count)
            {
                if (pair.second == num_remaining_cands)
                {
                    EClassId input_cid = pair.first;
                    auto in_st_it = state.start_vars[b].find(input_cid);
                    if (in_st_it != state.start_vars[b].end())
                    {
                        VarId in_st_v = in_st_it->second;
                        Domain in_st_dom = state.domains[in_st_v];
                        if (in_st_dom.setMax(max_allowed_input_start))
                        {
                            if (in_st_dom.isEmpty())
                                return false;
                            state.setDomain(in_st_v, in_st_dom);
                            worklist.push_back(in_st_v);
                        }
                    }
                }
            }
        }

        return true;
    }

  public:
    std::string name() const override
    {
        return "InputPruneStartPrecedencePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::START)
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            return pruneEClass(state, b, cid, worklist);
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    if (!pruneEClass(state, b, pair.first, worklist))
                        return false;
                }
            }
        }
        return true;
    }
};

} // namespace plan
