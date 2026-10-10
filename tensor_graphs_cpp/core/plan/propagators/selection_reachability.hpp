// tensor_graphs_cpp/core/plan/propagators/selection_reachability.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// SelectionReachabilityPropagator: see docs/core/propagators.md.
class SelectionReachabilityPropagator : public Propagator
{
    bool is_dag_ = false;

  public:
    explicit SelectionReachabilityPropagator(bool is_dag = false) : is_dag_(is_dag) {}

    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }

    std::string name() const override
    {
        return "SelectionReachabilityPropagator";
    }

    bool isDag() const
    {
        return is_dag_;
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;
            std::vector<VarId> unreachable;
            std::vector<std::pair<VarId, int32_t>> forced_parents;
            state.updateSelectionReachability(changed, unreachable, is_dag_, &forced_parents);
            for (VarId sel_v : unreachable)
            {
                const Domain &sel_dom = state.domains[sel_v];
                if (!sel_dom.contains(0))
                    return false;
                if (!sel_dom.isFixed())
                    state.setDomain(sel_v, Domain::makeFixed(0, sel_dom.is_mask));
            }
            for (const auto &[p_var, req_val] : forced_parents)
            {
                const Domain &p_dom = state.domains[p_var];
                if (!p_dom.contains(req_val))
                    return false;
                if (!p_dom.isFixed() || p_dom.fixedValue() != req_val)
                {
                    state.setDomain(p_var, Domain::makeFixed(req_val, p_dom.is_mask));
                    worklist.push_back(p_var);
                }
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
                std::vector<std::pair<VarId, int32_t>> forced_parents;
                state.updateSelectionReachability(root_var, unreachable, is_dag_, &forced_parents);
                for (VarId sel_v : unreachable)
                {
                    const Domain &sel_dom = state.domains[sel_v];
                    if (!sel_dom.contains(0))
                        return false;
                    if (!sel_dom.isFixed())
                        state.setDomain(sel_v, Domain::makeFixed(0, sel_dom.is_mask));
                }
                for (const auto &[p_var, req_val] : forced_parents)
                {
                    const Domain &p_dom = state.domains[p_var];
                    if (!p_dom.contains(req_val))
                        return false;
                    if (!p_dom.isFixed() || p_dom.fixedValue() != req_val)
                    {
                        state.setDomain(p_var, Domain::makeFixed(req_val, p_dom.is_mask));
                        worklist.push_back(p_var);
                    }
                }
            }
        }
        return true;
    }
};

} // namespace plan
