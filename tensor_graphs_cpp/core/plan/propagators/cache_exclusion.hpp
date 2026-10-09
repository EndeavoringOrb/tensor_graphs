// tensor_graphs_cpp/core/plan/propagators/cache_exclusion.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// CacheExclusionPropagator: see docs/core/propagators.md.
class CacheExclusionPropagator : public Propagator
{
    struct IndexedClass
    {
        uint32_t bucket_idx;
        EClassId cid;
        VarId sel_v;
        std::vector<int32_t> cache_values;
    };
    std::unordered_map<BaseEClassId, std::vector<IndexedClass>> base_to_classes_;
    bool initialized_ = false;

    void ensureIndex(const SearchState &state)
    {
        if (initialized_)
            return;
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            if (b >= state.selected_vars.size())
                continue;
            for (const auto &pair : state.selected_vars[b])
            {
                EClassId cid = pair.first;
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (cls.base_eclass_id == BaseEClassId{})
                    continue;

                std::vector<int32_t> cache_values;
                for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
                {
                    const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                    if (isOpCacheOrScatter(enode))
                    {
                        cache_values.push_back(static_cast<int32_t>(en_idx + 1));
                    }
                }
                if (!cache_values.empty())
                {
                    base_to_classes_[cls.base_eclass_id].push_back(
                        IndexedClass{b, cid, pair.second, std::move(cache_values)});
                }
            }
        }
        initialized_ = true;
    }

    bool excludeForBase(SearchState &state, BaseEClassId base_id)
    {
        ensureIndex(state);
        auto it = base_to_classes_.find(base_id);
        if (it == base_to_classes_.end())
            return true;

        for (const auto &entry : it->second)
        {
            VarId sel_v = entry.sel_v;
            Domain sel_dom = state.domains[sel_v];
            bool changed = false;
            for (int32_t val : entry.cache_values)
            {
                if (sel_dom.contains(val))
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
        return true;
    }

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::CACHED); }

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
