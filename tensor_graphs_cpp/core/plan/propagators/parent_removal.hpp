// tensor_graphs_cpp/core/plan/propagators/parent_removal.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

class ParentRemovalPropagator : public Propagator
{
    struct ParentInfo
    {
        EClassId parent_cid;
        uint32_t en_idx;
    };
    std::vector<std::unordered_map<EClassId, std::vector<ParentInfo>>> bucket_parents;
    bool parents_initialized = false;

    void ensureParents(const SearchState &state)
    {
        if (parents_initialized)
            return;
        bucket_parents.resize(state.buckets.size());
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &pair : state.selected_vars[b])
            {
                EClassId p_cid = pair.first;
                const EClass &cls = state.bucket_egraphs[b].getEClass(p_cid);
                for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
                {
                    const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                    for (EClassId ch : enode.getChildren())
                    {
                        EClassId canon_ch = state.bucket_egraphs[b].findConst(ch);
                        bucket_parents[b][canon_ch].push_back(ParentInfo{p_cid, en_idx});
                    }
                }
            }
        }
        parents_initialized = true;
    }

    bool removeParentEnodes(SearchState &state, uint32_t b, EClassId unselected_cid)
    {
        ensureParents(state);
        auto it = bucket_parents[b].find(unselected_cid);
        if (it == bucket_parents[b].end())
            return true;

        for (const auto &p_info : it->second)
        {
            auto p_sel_it = state.selected_vars[b].find(p_info.parent_cid);
            if (p_sel_it == state.selected_vars[b].end())
                continue;
            VarId p_sel_v = p_sel_it->second;
            Domain p_sel_dom = state.domains[p_sel_v];
            if (p_sel_dom.contains(static_cast<int32_t>(p_info.en_idx + 1)))
            {
                p_sel_dom.remove(static_cast<int32_t>(p_info.en_idx + 1));
                if (p_sel_dom.isEmpty())
                    return false;
                state.setDomain(p_sel_v, p_sel_dom);
            }
        }
        return true;
    }

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }

    std::string name() const override
    {
        return "ParentRemovalPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;
            const Domain &dom = state.domains[changed];
            if (dom.isFixed() && dom.fixedValue() == 0)
            {
                return removeParentEnodes(state, state.var_infos[changed].bucket_idx,
                                          state.var_infos[changed].eclass_id);
            }
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    const Domain &dom = state.domains[pair.second];
                    if (dom.isFixed() && dom.fixedValue() == 0)
                    {
                        if (!removeParentEnodes(state, b, pair.first))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};


} // namespace plan
