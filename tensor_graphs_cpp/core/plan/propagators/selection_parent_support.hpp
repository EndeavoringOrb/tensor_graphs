#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// Runs immediately after SelectionReachabilityPropagator. The shared graph
// produces incremental proposals; this propagator owns their domain deductions.
class SelectionParentSupportPropagator : public Propagator
{
    std::string conflict_reason_;
    std::vector<std::pair<VarId, int32_t>> proposals_;

  public:
    std::string name() const override { return "SelectionParentSupportPropagator"; }
    std::string conflictReason() const override { return conflict_reason_; }
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }

    bool propagate(SearchState &state, VarId, std::vector<VarId> &worklist) override
    {
        conflict_reason_.clear();
        // Swap out before applying changes, including on the conflict path.
        proposals_.clear();
        proposals_.swap(state.pending_parent_selections);
        for (const auto &[parent_var, required_value] : proposals_)
        {
            const Domain &domain = state.domains.at(parent_var);
            if (!domain.contains(required_value))
            {
                conflict_reason_ = "single supporting parent " + state.var_infos.at(parent_var).name +
                    "=" + domain.toString() + " cannot select " + std::to_string(required_value);
                return false;
            }
            if (!domain.isFixed())
            {
                state.setDomain(parent_var, Domain::makeFixed(required_value, domain.is_mask));
                worklist.push_back(parent_var);
            }
        }
        return true;
    }
};

} // namespace plan
