// tensor_graphs_cpp/core/plan/propagators/view_to_base_offset.hpp
#pragma once

#include "core/plan/propagators/base.hpp"
#include "core/plan/propagators/view_offset_helpers.hpp"

namespace plan
{

// ViewToBaseOffsetPropagator: see docs/core/propagators.md.
class ViewToBaseOffsetPropagator : public Propagator
{
    bool propagateEClass(SearchState &state, uint32_t b, EClassId cid, std::vector<VarId> &worklist)
    {
        EClassId base_cid;
        if (ViewOffsetHelpers::getConfirmedViewBase(state, b, cid, base_cid))
        {
            if (!ViewOffsetHelpers::intersectViewAndBase(state, b, cid, base_cid, worklist))
                return false;
        }
        else
        {
            const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
            for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
            {
                ENodeId en_id = cls.enodes[en_idx];
                bool is_view = en_id.value < state.bucket_enode_infos[b].size() &&
                               state.bucket_enode_infos[b][en_id.value].is_view;
                if (!is_view)
                    continue;
                const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                if (!enode.getChildren().empty())
                {
                    EClassId cand_base = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
                    if (!ViewOffsetHelpers::filterUnviableViewEnodes(state, b, cid, cand_base, worklist))
                        return false;
                }
            }
        }
        return true;
    }

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::OFFSET); }

    std::string name() const override
    {
        return "ViewToBaseOffsetPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            const auto &info = state.var_infos[changed];
            if (info.type != VarType::OFFSET)
                return true;
            return propagateEClass(state, info.bucket_idx, info.eclass_id, worklist);
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.offset_vars[b])
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
