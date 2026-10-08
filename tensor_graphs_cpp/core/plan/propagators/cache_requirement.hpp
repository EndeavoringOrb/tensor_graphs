// tensor_graphs_cpp/core/plan/propagators/cache_requirement.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

class CacheRequirementPropagator : public Propagator
{
    bool requireCacheForEClass(SearchState &state, uint32_t b, EClassId cid, int32_t val)
    {
        if (val <= 0)
            return true;
        uint32_t en_idx = static_cast<uint32_t>(val - 1);
        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        if (en_idx >= cls.enodes.size())
            return false;
        const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
        if (isOpRootScatterOrCache(enode))
        {
            auto it = state.cached_vars.find(cls.base_eclass_id);
            if (it != state.cached_vars.end())
            {
                VarId cv = it->second;
                Domain c_dom = state.domains[cv];
                if (!c_dom.contains(1))
                    return false;
                if (!c_dom.isFixed())
                {
                    state.setDomain(cv, Domain::makeFixed(1, c_dom.is_mask));
                }
            }
        }
        return true;
    }

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }

    std::string name() const override
    {
        return "CacheRequirementPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            const VarInfo &changed_info = state.var_infos[changed];
            if (changed_info.type != VarType::SELECTED)
                return true;
            const Domain &dom = state.domains[changed];
            if (dom.isFixed() && dom.fixedValue() > 0)
            {
                return requireCacheForEClass(state, state.var_infos[changed].bucket_idx,
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
                        if (!requireCacheForEClass(state, b, pair.first, dom.fixedValue()))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};


} // namespace plan
