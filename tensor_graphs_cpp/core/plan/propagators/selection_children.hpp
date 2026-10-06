// tensor_graphs_cpp/core/plan/propagators/selection_children.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// SelectionChildrenPropagator: see docs/core/propagators.md.
class SelectionChildrenPropagator : public Propagator
{
    bool propagateEClass(SearchState &state, uint32_t b, EClassId cid, int32_t fixed_val)
    {
        if (fixed_val <= 0)
            return true;
        uint32_t en_idx = static_cast<uint32_t>(fixed_val - 1);
        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        if (en_idx >= cls.enodes.size())
            return false;
        ENodeId en_id = cls.enodes[en_idx];
        const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
        for (EClassId child : enode.getChildren())
        {
            EClassId canon_child = state.bucket_egraphs[b].findConst(child);
            auto ch_it = state.selected_vars[b].find(canon_child);
            if (ch_it != state.selected_vars[b].end())
            {
                VarId ch_v = ch_it->second;
                Domain ch_dom = state.domains[ch_v];
                if (ch_dom.contains(0))
                {
                    ch_dom.remove(0);
                    if (ch_dom.isEmpty())
                        return false;
                    state.setDomain(ch_v, ch_dom);
                }
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "SelectionChildrenPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;
            const Domain &dom = state.domains[changed];
            if (dom.isEmpty())
                return false;
            if (dom.isFixed() && dom.fixedValue() > 0)
            {
                return propagateEClass(state, state.var_infos[changed].bucket_idx,
                                       state.var_infos[changed].eclass_id, dom.fixedValue());
            }
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    const Domain &dom = state.domains[pair.second];
                    if (dom.isFixed() && dom.fixedValue() > 0)
                    {
                        if (!propagateEClass(state, b, pair.first, dom.fixedValue()))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};

} // namespace plan
