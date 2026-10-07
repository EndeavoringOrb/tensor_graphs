// tensor_graphs_cpp/core/plan/propagators/unselected_start_offset.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// UnselectedStartOffsetPropagator: see docs/core/propagators.md.
class UnselectedStartOffsetPropagator : public Propagator
{
    bool cleanupEClass(SearchState &state, uint32_t b, EClassId cid)
    {
        auto st_it = state.start_vars[b].find(cid);
        if (st_it != state.start_vars[b].end())
        {
            VarId st_v = st_it->second;
            const Domain &st_dom = state.domains[st_v];
            if (!st_dom.isFixed() || st_dom.fixedValue() != 0)
                state.setDomain(st_v, Domain::makeFixed(0, false));
        }
        auto off_it = state.offset_vars[b].find(cid);
        if (off_it != state.offset_vars[b].end())
        {
            VarId off_v = off_it->second;
            const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
            const auto prealloc_it = state.preallocated_pages.find(cls.mem_space);
            uint32_t min_p = (prealloc_it == state.preallocated_pages.end()) ? 0 : prealloc_it->second;
            const Domain &off_dom = state.domains[off_v];
            if (!off_dom.isFixed() || off_dom.fixedValue() != static_cast<int32_t>(min_p))
            {
                state.setDomain(off_v, Domain::makeFixed(min_p, false));
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "UnselectedStartOffsetPropagator";
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
                return cleanupEClass(state, state.var_infos[changed].bucket_idx,
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
                        if (!cleanupEClass(state, b, pair.first))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};

} // namespace plan
