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
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::CACHED)
                return true;
            MemSpace space = state.var_infos[changed].mem_space;
            auto it = state.fixed_cache_bytes.find(space);
            uint64_t bytes = (it != state.fixed_cache_bytes.end()) ? it->second : 0;
            if (bytes > state.getMemoryCap(space))
                return false;
            return true;
        }

        for (const auto &pair : state.fixed_cache_bytes)
        {
            uint64_t cap = state.getMemoryCap(pair.first);
            if (pair.second > cap)
                return false;
        }

        return true;
    }
};

} // namespace plan
