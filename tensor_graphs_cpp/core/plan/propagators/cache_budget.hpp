// tensor_graphs_cpp/core/plan/propagators/cache_budget.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// CacheBudgetPropagator: see docs/core/propagators.md.
class CacheBudgetPropagator : public Propagator
{
  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::CACHED); }

    std::string name() const override
    {
        return "CacheBudgetPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId && state.var_infos[changed].type != VarType::CACHED)
            return true;

        std::unordered_map<MemSpace, uint64_t> fixed_cache_bytes;
        for (const auto &cand : state.candidates)
        {
            auto c_it = state.cached_vars.find(cand.base_eclass_id);
            if (c_it != state.cached_vars.end())
            {
                VarId cv = c_it->second;
                if (state.domains[cv].isFixed() && state.domains[cv].fixedValue() == 1)
                {
                    fixed_cache_bytes[cand.mem_space] += cand.size_bytes;
                }
            }
        }

        for (const auto &pair : fixed_cache_bytes)
        {
            uint64_t cap = state.getMemoryCap(pair.first);
            if (pair.second > cap)
                return false;
        }

        return true;
    }
};

} // namespace plan
