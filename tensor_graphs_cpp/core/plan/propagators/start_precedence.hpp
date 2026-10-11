// tensor_graphs_cpp/core/plan/propagators/start_precedence.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// StartPrecedencePropagator: see docs/core/propagators.md.
class StartPrecedencePropagator : public Propagator
{
    std::string conflict_reason_;

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
        VarId st_var = st_it->second;
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
                if (ch_sel_dom.isFixed() && ch_sel_dom.fixedValue() == 0)
                    continue;
                if (!state.domains[ch_st_it->second].isEmpty())
                {
                    int32_t input_start_min = state.domains[ch_st_it->second].getMin();
                    enode_start_min = std::max(enode_start_min, input_start_min + 1);
                }
            }
            min_enode_start_min = std::min(min_enode_start_min, enode_start_min);
        }

        if (min_enode_start_min != INT32_MAX && min_enode_start_min > 0)
        {
            Domain st_dom = state.domains[st_var];
            if (st_dom.setMin(min_enode_start_min))
            {
                if (st_dom.isEmpty())
                {
                    conflict_reason_ = "consumer=" + state.var_infos[st_var].name +
                        state.domains[st_var].toString() + " selection=" + sel_dom.toString() +
                        " required-min=" + std::to_string(min_enode_start_min);
                    return false;
                }
                state.setDomain(st_var, st_dom);
                worklist.push_back(st_var);
            }
        }
        return true;
    }

  public:
    explicit StartPrecedencePropagator(bool /*fixed_starts_only*/ = false)
    {
    }

    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED) | varTypeMask(VarType::START); }

    std::string conflictReason() const override { return conflict_reason_; }

    std::string name() const override
    {
        return "StartPrecedencePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        conflict_reason_.clear();
        if (changed != kInvalidVarId)
        {
            const auto type = state.var_infos.at(changed).type;
            if (type != VarType::START && type != VarType::SELECTED)
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            if (state.var_infos[changed].type == VarType::SELECTED)
            {
                // Selection changes activate both forward and backward edges.
                state.schedulePropagation(state.start_vars[b].at(cid));
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
                    if (state.domains[changed].contains(static_cast<int32_t>(en_idx + 1)))
                        for (EClassId child : state.bucket_egraphs[b].getENode(cls.enodes[en_idx]).getChildren())
                        {
                            const auto it = state.start_vars[b].find(state.bucket_egraphs[b].findConst(child));
                            if (it != state.start_vars[b].end())
                                state.schedulePropagation(it->second);
                        }
            }
            if (!propagateEClass(state, b, cid, worklist))
                return false;
            state.ensurePropagationState();
            for (const auto &consumer : state.propagation.start_precedence[b].consumers_by_cid.at(cid.value))
                if (!propagateEClass(state, b, consumer.parent_cid, worklist))
                    return false;
            return true;
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
