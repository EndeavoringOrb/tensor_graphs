// tensor_graphs_cpp/core/plan/propagators/cached_offset_allocation.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

class CachedOffsetAllocationPropagator : public Propagator
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

    uint32_t getMaxSizePages(const SearchState &state, BaseEClassId base_id) const
    {
        uint32_t max_pages = 1;
        const auto cache_it = state.cached_vars.find(base_id);
        if (cache_it != state.cached_vars.end())
        {
            const VarInfo &cinfo = state.var_infos[cache_it->second];
            max_pages = std::max(max_pages, state.bytesToPages(cinfo.size_bytes, cinfo.mem_space));
        }
        const auto off_it = offsets_by_base_.find(base_id);
        if (off_it != offsets_by_base_.end())
        {
            for (VarId off_var : off_it->second)
            {
                max_pages = std::max(max_pages, state.var_infos[off_var].size_pages);
            }
        }
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            EClassId cid = state.bucket_egraphs[b].findEClassByBaseId(base_id);
            if (cid != EClassId{})
            {
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                max_pages = std::max(max_pages, state.bytesToPages(getSizeBytes(cls.shape, cls.dtype), cls.mem_space));
            }
        }
        return max_pages;
    }

    bool allocateCachedOffset(SearchState &state, BaseEClassId base_id, std::vector<VarId> &worklist)
    {
        const auto cache_it = state.cached_vars.find(base_id);
        const auto offsets_it = offsets_by_base_.find(base_id);
        if (cache_it == state.cached_vars.end() || offsets_it == offsets_by_base_.end() || offsets_it->second.empty())
            return true;

        const VarInfo &cache_info = state.var_infos[cache_it->second];
        const uint32_t size_pages = getMaxSizePages(state, base_id);

        int32_t min_offset = 0;
        int32_t max_offset = std::numeric_limits<int32_t>::max();
        int32_t fixed_offset = -1;
        for (VarId offset_var : offsets_it->second)
        {
            const Domain &domain = state.domains[offset_var];
            if (domain.isEmpty())
                return false;
            min_offset = std::max(min_offset, domain.getMin());
            max_offset = std::min(max_offset, domain.getMax());
            if (domain.isFixed())
            {
                if (fixed_offset >= 0 && fixed_offset != domain.fixedValue())
                    return false;
                fixed_offset = domain.fixedValue();
            }
        }

        if (fixed_offset >= 0)
        {
            for (VarId offset_var : offsets_it->second)
            {
                const Domain &domain = state.domains[offset_var];
                if (!domain.contains(fixed_offset))
                    return false;
                if (!domain.isFixed())
                {
                    if (state.setDomain(offset_var, Domain::makeFixed(fixed_offset)))
                        worklist.push_back(offset_var);
                }
            }
            return true;
        }

        // Approach A: Arrival-order sequential packing for cached nodes.
        // Pack cached variables sequentially in the dynamic arena starting at min_p
        // (after preallocated buffers).
        const auto prealloc_it = state.preallocated_pages.find(cache_info.mem_space);
        int64_t candidate = (prealloc_it != state.preallocated_pages.end()) ? prealloc_it->second : 0;
        candidate = std::max<int64_t>(candidate, min_offset);

        for (const auto &other_cache : state.cached_vars)
        {
            if (other_cache.first == base_id)
                continue;
            const Domain &cached_domain = state.domains[other_cache.second];
            if (!cached_domain.isFixed() || cached_domain.fixedValue() != 1)
                continue;

            const auto other_offsets_it = offsets_by_base_.find(other_cache.first);
            if (other_offsets_it == offsets_by_base_.end())
                continue;

            int32_t other_offset = -1;
            for (VarId offset_var : other_offsets_it->second)
            {
                const Domain &domain = state.domains[offset_var];
                if (domain.isFixed())
                {
                    other_offset = domain.fixedValue();
                    break;
                }
            }
            if (other_offset < 0)
                continue;

            const VarInfo &other_info = state.var_infos[other_cache.second];
            if (other_info.mem_space != cache_info.mem_space)
                continue;

            const uint32_t other_size = getMaxSizePages(state, other_cache.first);
            candidate = std::max<int64_t>(candidate, static_cast<int64_t>(other_offset) + other_size);
        }

        if (candidate > max_offset || candidate + size_pages - 1 > std::numeric_limits<int32_t>::max())
            return false;

        uint64_t mem_cap = state.getMemoryCap(cache_info.mem_space);
        uint32_t align = state.getPageAlignment(cache_info.mem_space);
        if (mem_cap > 0 && (static_cast<uint64_t>(candidate + size_pages) * align > mem_cap))
            return false;

        for (VarId offset_var : offsets_it->second)
        {
            const Domain &domain = state.domains[offset_var];
            if (!domain.contains(static_cast<int32_t>(candidate)))
                return false;
        }

        for (VarId offset_var : offsets_it->second)
        {
            if (state.setDomain(offset_var, Domain::makeFixed(static_cast<int32_t>(candidate))))
                worklist.push_back(offset_var);
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "CachedOffsetAllocationPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        ensureOffsetIndex(state);
        if (changed == kInvalidVarId)
        {
            std::vector<BaseEClassId> active_caches;
            for (const auto &cache_pair : state.cached_vars)
            {
                const Domain &domain = state.domains[cache_pair.second];
                if (domain.isFixed() && domain.fixedValue() == 1)
                    active_caches.push_back(cache_pair.first);
            }
            std::sort(active_caches.begin(), active_caches.end(), [](BaseEClassId a, BaseEClassId b) {
                return a.value < b.value;
            });
            for (BaseEClassId bid : active_caches)
            {
                if (!allocateCachedOffset(state, bid, worklist))
                    return false;
            }
            return true;
        }

        const VarInfo &info = state.var_infos[changed];
        if (info.type == VarType::CACHED)
        {
            const Domain &domain = state.domains[changed];
            return !domain.isFixed() || domain.fixedValue() != 1 ||
                   allocateCachedOffset(state, info.base_eclass_id, worklist);
        }
        if (info.type == VarType::OFFSET)
        {
            const BaseEClassId base_id = state.bucket_egraphs[info.bucket_idx].getEClass(info.eclass_id).base_eclass_id;
            const auto cache_it = state.cached_vars.find(base_id);
            if (cache_it == state.cached_vars.end())
                return true;
            const Domain &cache_domain = state.domains[cache_it->second];
            if (!cache_domain.isFixed() || cache_domain.fixedValue() != 1)
                return true;
            return allocateCachedOffset(state, base_id, worklist);
        }
        return true;
    }
};

// ============================================================================
// LOWER-BOUND PROPAGATORS
// ============================================================================


} // namespace plan
