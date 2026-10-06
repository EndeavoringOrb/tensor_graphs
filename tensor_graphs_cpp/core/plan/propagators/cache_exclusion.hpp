// tensor_graphs_cpp/core/plan/propagators/cache_exclusion.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// CacheExclusionPropagator: see docs/core/propagators.md.
class CacheExclusionPropagator : public Propagator
{
    bool excludeForBase(SearchState &state, BaseEClassId base_id)
    {
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &pair : state.selected_vars[b])
            {
                EClassId cid = pair.first;
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (cls.base_eclass_id != base_id)
                    continue;

                VarId sel_v = pair.second;
                Domain sel_dom = state.domains[sel_v];
                bool changed = false;
                for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
                {
                    int32_t val = static_cast<int32_t>(en_idx + 1);
                    if (!sel_dom.contains(val))
                        continue;
                    const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                    if (isOpCacheOrScatter(enode))
                    {
                        changed = sel_dom.remove(val) || changed;
                    }
                }
                if (changed)
                {
                    if (sel_dom.isEmpty())
                        return false;
                    state.setDomain(sel_v, sel_dom);
                }
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "CacheExclusionPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::CACHED)
                return true;
            const Domain &dom = state.domains[changed];
            if (dom.isFixed() && dom.fixedValue() == 0)
            {
                return excludeForBase(state, state.var_infos[changed].base_eclass_id);
            }
        }
        else
        {
            for (const auto &pair : state.cached_vars)
            {
                const Domain &dom = state.domains[pair.second];
                if (dom.isFixed() && dom.fixedValue() == 0)
                {
                    if (!excludeForBase(state, pair.first))
                        return false;
                }
            }
        }
        return true;
    }
};

} // namespace plan
