// tensor_graphs_cpp/core/plan/propagators/start_unique.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// StartUniquePropagator: see docs/core/propagators.md.
class StartUniquePropagator : public Propagator
{
    bool removeStartVal(SearchState &state, uint32_t b, VarId source_st_v, int32_t st_val)
    {
        for (const auto &pair : state.selected_vars[b])
        {
            EClassId other_cid = pair.first;
            VarId other_sel_v = pair.second;
            const Domain &other_sel_dom = state.domains[other_sel_v];
            if (other_sel_dom.contains(0))
                continue;

            VarId other_st_v = state.start_vars[b].at(other_cid);
            if (other_st_v == source_st_v)
                continue;
            Domain other_st_dom = state.domains[other_st_v];
            if (other_st_dom.contains(st_val))
            {
                if (other_st_dom.isFixed())
                    return false;
                other_st_dom.remove(st_val);
                if (other_st_dom.isEmpty())
                    return false;
                state.setDomain(other_st_v, other_st_dom);
            }
        }
        return true;
    }

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::START); }
    StartSelectionGuard startSelectionGuard() const override { return StartSelectionGuard::FIXED_POSITIVE; }

    std::string name() const override
    {
        return "StartUniquePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::START)
                return true;
            const Domain &dom = state.domains[changed];
            if (!dom.isFixed())
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            VarId sel_v = state.selected_vars[b].at(cid);
            const Domain &sel_dom = state.domains[sel_v];
            if (!sel_dom.isFixed() || sel_dom.fixedValue() <= 0)
                return true;
            return removeStartVal(state, b, changed, dom.fixedValue());
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    EClassId cid = pair.first;
                    VarId sel_v = pair.second;
                    const Domain &sel_dom = state.domains[sel_v];
                    if (!sel_dom.isFixed() || sel_dom.fixedValue() <= 0)
                        continue;
                    VarId st_v = state.start_vars[b].at(cid);
                    const Domain &st_dom = state.domains[st_v];
                    if (st_dom.isFixed())
                    {
                        if (!removeStartVal(state, b, st_v, st_dom.fixedValue()))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};

} // namespace plan
