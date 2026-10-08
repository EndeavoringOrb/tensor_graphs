// tensor_graphs_cpp/core/plan/propagators/cached_offset.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

class CachedOffsetPropagator : public Propagator
{
    bool offset_index_initialized_ = false;
    std::unordered_map<BaseEClassId, std::vector<VarId>> offsets_by_base_;

    void ensureOffsetIndex(const SearchState &state)
    {
        if (offset_index_initialized_)
            return;
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &offset_pair : state.offset_vars[b])
            {
                const BaseEClassId base_id = state.bucket_egraphs[b].getEClass(offset_pair.first).base_eclass_id;
                offsets_by_base_[base_id].push_back(offset_pair.second);
            }
        }
        offset_index_initialized_ = true;
    }

    bool synchronizeOffset(SearchState &state, BaseEClassId base_id, VarId source_var)
    {
        const Domain &source_domain = state.domains[source_var];
        if (!source_domain.isFixed())
            return true;
        const int32_t shared_offset = source_domain.fixedValue();

        auto offsets_it = offsets_by_base_.find(base_id);
        if (offsets_it == offsets_by_base_.end())
            return true;
        for (VarId other_var : offsets_it->second)
        {
            if (other_var == source_var)
                continue;

            const Domain &other_domain = state.domains[other_var];
            if (!other_domain.contains(shared_offset))
                return false;
            state.setDomain(other_var, Domain::makeFixed(shared_offset));
        }
        return true;
    }

  public:
    uint8_t interestedVarTypes() const override
    {
        return varTypeMask(VarType::CACHED) | varTypeMask(VarType::OFFSET);
    }

    std::string name() const override
    {
        return "CachedOffsetPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed == kInvalidVarId)
            return true;

        ensureOffsetIndex(state);
        const VarInfo &info = state.var_infos[changed];
        BaseEClassId base_id;
        if (info.type == VarType::OFFSET)
        {
            base_id = state.bucket_egraphs[info.bucket_idx].getEClass(info.eclass_id).base_eclass_id;
            const auto cache_it = state.cached_vars.find(base_id);
            if (cache_it == state.cached_vars.end())
                return true;
            const Domain &cache_dom = state.domains[cache_it->second];
            if (!cache_dom.isFixed() || cache_dom.fixedValue() != 1)
                return true;
            return synchronizeOffset(state, base_id, changed);
        }

        if (info.type != VarType::CACHED)
            return true;
        const Domain &cache_dom = state.domains[changed];
        if (!cache_dom.isFixed() || cache_dom.fixedValue() != 1)
            return true;
        base_id = info.base_eclass_id;
        auto offsets_it = offsets_by_base_.find(base_id);
        if (offsets_it != offsets_by_base_.end())
        {
            for (VarId offset_var : offsets_it->second)
            {
                if (state.domains[offset_var].isFixed())
                    return synchronizeOffset(state, base_id, offset_var);
            }
        }
        return true;
    }
};


} // namespace plan
