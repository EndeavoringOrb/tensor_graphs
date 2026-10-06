// tensor_graphs_cpp/core/plan/propagators/start_precedence.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// StartPrecedencePropagator: see docs/core/propagators.md.
class StartPrecedencePropagator : public Propagator
{
    bool propagateEClass(SearchState &state, uint32_t b, EClassId cid, std::vector<VarId> &worklist)
    {
        auto sel_it = state.selected_vars[b].find(cid);
        if (sel_it == state.selected_vars[b].end())
            return true;
        VarId sel_v = sel_it->second;
        const Domain &sel_dom = state.domains[sel_v];
        if (sel_dom.contains(0))
            return true;

        auto st_it = state.start_vars[b].find(cid);
        if (st_it == state.start_vars[b].end())
            return true;
        const auto &st_vars = st_it->second;
        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);

        int32_t min_enode_start_min = INT32_MAX;
        for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
        {
            if (!sel_dom.contains(static_cast<int32_t>(en_idx + 1)))
                continue;

            ENodeId en_id = cls.enodes[en_idx];
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
            int32_t enode_start_min = 0;
            for (EClassId child : enode.getChildren())
            {
                EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                auto ch_sel_it = state.selected_vars[b].find(canon_child);
                auto ch_st_it = state.start_vars[b].find(canon_child);
                if (ch_sel_it == state.selected_vars[b].end() || ch_st_it == state.start_vars[b].end())
                    continue;

                const Domain &ch_sel_dom = state.domains[ch_sel_it->second];
                const auto &ch_st_vars = ch_st_it->second;
                int32_t input_start_min = INT32_MAX;
                for (uint32_t ch_en = 0; ch_en < ch_st_vars.size(); ++ch_en)
                {
                    if (ch_sel_dom.contains(static_cast<int32_t>(ch_en + 1)))
                    {
                        input_start_min = std::min(input_start_min, state.domains[ch_st_vars[ch_en]].getMin());
                    }
                }
                if (input_start_min != INT32_MAX)
                {
                    enode_start_min = std::max(enode_start_min, input_start_min + 1);
                }
            }
            min_enode_start_min = std::min(min_enode_start_min, enode_start_min);
        }

        if (min_enode_start_min != INT32_MAX && min_enode_start_min > 0)
        {
            for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
            {
                if (!sel_dom.contains(static_cast<int32_t>(en_idx + 1)))
                    continue;
                if (en_idx >= st_vars.size())
                    continue;
                VarId st_v = st_vars[en_idx];
                Domain st_dom = state.domains[st_v];
                if (st_dom.setMin(min_enode_start_min))
                {
                    if (st_dom.isEmpty())
                        return false;
                    state.setDomain(st_v, st_dom);
                    worklist.push_back(st_v);
                }
            }
        }
        return true;
    }

  public:
    explicit StartPrecedencePropagator(bool /*fixed_starts_only*/ = false)
    {
    }

    std::string name() const override
    {
        return "StartPrecedencePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            return propagateEClass(state, b, cid, worklist);
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    if (!propagateEClass(state, b, pair.first, worklist))
                        return false;
                }
            }
        }
        return true;
    }
};

} // namespace plan
