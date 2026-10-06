// tensor_graphs_cpp/core/plan/propagators/selection_reachability.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// SelectionReachabilityPropagator: see docs/core/propagators.md.
class SelectionReachabilityPropagator : public Propagator
{
  public:
    std::string name() const override
    {
        return "SelectionReachabilityPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;
            std::vector<VarId> unreachable;
            state.updateSelectionReachability(changed, unreachable);
            for (VarId sel_v : unreachable)
            {
                const Domain &sel_dom = state.domains[sel_v];
                if (!sel_dom.contains(0))
                    return false;
                if (!sel_dom.isFixed())
                    state.setDomain(sel_v, Domain::makeFixed(0, sel_dom.is_mask));
            }
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                if (b >= state.bucket_root_ids.size())
                    continue;
                EClassId root_cid = state.bucket_root_ids[b];
                auto it = state.selected_vars[b].find(root_cid);
                if (it == state.selected_vars[b].end())
                    continue;
                VarId root_var = it->second;
                std::vector<VarId> unreachable;
                state.updateSelectionReachability(root_var, unreachable);
                for (VarId sel_v : unreachable)
                {
                    const Domain &sel_dom = state.domains[sel_v];
                    if (!sel_dom.contains(0))
                        return false;
                    if (!sel_dom.isFixed())
                        state.setDomain(sel_v, Domain::makeFixed(0, sel_dom.is_mask));
                }
            }
        }
        return true;
    }
};

} // namespace plan
