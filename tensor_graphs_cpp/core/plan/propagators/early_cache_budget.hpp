// tensor_graphs_cpp/core/plan/propagators/early_cache_budget.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// EarlyCacheBudgetPropagator: see docs/core/propagators.md.
class EarlyCacheBudgetPropagator : public Propagator
{
    struct CandidateEntry
    {
        VarId cv;
        uint64_t size_bytes;
    };
    std::unordered_map<MemSpace, std::vector<CandidateEntry>> candidates_by_space_;
    bool initialized_ = false;

    void ensureIndex(const SearchState &state)
    {
        if (initialized_)
            return;
        for (const auto &cand : state.candidates)
        {
            auto c_it = state.cached_vars.find(cand.base_eclass_id);
            if (c_it != state.cached_vars.end())
            {
                candidates_by_space_[cand.mem_space].push_back({c_it->second, cand.size_bytes});
            }
        }
        for (auto &pair : candidates_by_space_)
        {
            std::sort(pair.second.begin(), pair.second.end(),
                      [](const CandidateEntry &a, const CandidateEntry &b) {
                          return a.size_bytes > b.size_bytes;
                      });
        }
        initialized_ = true;
    }

    bool pruneSpace(SearchState &state, MemSpace space, std::vector<VarId> &worklist)
    {
        auto it = state.fixed_cache_bytes.find(space);
        uint64_t current_fixed = (it != state.fixed_cache_bytes.end()) ? it->second : 0;
        uint64_t cap = state.getMemoryCap(space);
        if (cap == 0)
            return true;

        auto cands_it = candidates_by_space_.find(space);
        if (cands_it == candidates_by_space_.end())
            return true;

        const uint64_t remaining_budget = (current_fixed < cap) ? (cap - current_fixed) : 0;

        for (const auto &entry : cands_it->second)
        {
            if (current_fixed <= cap && entry.size_bytes <= remaining_budget)
            {
                // All remaining candidates are <= entry.size_bytes <= remaining_budget,
                // so none of them can exceed the cap.
                break;
            }

            Domain c_dom = state.domains[entry.cv];
            if (!c_dom.isFixed() && c_dom.contains(1))
            {
                c_dom.remove(1);
                if (c_dom.isEmpty())
                    return false;
                state.setDomain(entry.cv, c_dom);
                worklist.push_back(entry.cv);
            }
        }
        return true;
    }

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::CACHED); }

    std::string name() const override
    {
        return "EarlyCacheBudgetPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        ensureIndex(state);
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::CACHED)
                return true;
            const Domain &changed_dom = state.domains[changed];
            if (changed_dom.isFixed() && changed_dom.fixedValue() == 0)
                return true;

            MemSpace space = state.var_infos[changed].mem_space;
            return pruneSpace(state, space, worklist);
        }

        for (const auto &pair : candidates_by_space_)
        {
            if (!pruneSpace(state, pair.first, worklist))
                return false;
        }
        return true;
    }
};

} // namespace plan
