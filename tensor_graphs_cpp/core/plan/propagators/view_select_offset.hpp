// tensor_graphs_cpp/core/plan/propagators/view_select_offset.hpp
#pragma once

#include "core/plan/propagators/base.hpp"
#include "core/plan/propagators/view_offset_helpers.hpp"

namespace plan
{

// ViewSelectOffsetPropagator: see docs/core/propagators.md.
class ViewSelectOffsetPropagator : public Propagator
{
    bool propagateEClass(SearchState &state, uint32_t b, EClassId cid, std::vector<VarId> &worklist)
    {
        EClassId curr = cid;
        for (uint32_t step = 0; step < 32; ++step)
        {
            EClassId next_base;
            if (!ViewOffsetHelpers::getConfirmedViewBase(state, b, curr, next_base))
                break;
            if (!ViewOffsetHelpers::intersectViewAndBase(state, b, curr, next_base, worklist))
                return false;
            curr = next_base;
        }
        return true;
    }

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }

    std::string name() const override
    {
        return "ViewSelectOffsetPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            const auto &info = state.var_infos[changed];
            if (info.type != VarType::SELECTED)
                return true;
            return propagateEClass(state, info.bucket_idx, info.eclass_id, worklist);
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
