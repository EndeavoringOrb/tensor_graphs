// tensor_graphs_cpp/core/plan/propagators/cache_freshness.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

class CacheFreshnessPropagator : public Propagator
{
    std::unordered_map<BaseEClassId, std::vector<VarId>> base_to_selected_;
    std::unordered_map<VarId, VarId> selected_to_cached_;
    bool initialized_ = false;

    void ensureIndex(const SearchState &state)
    {
        if (initialized_)
            return;
        for (const auto &[base_id, cached_vid] : state.cached_vars)
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                if (b >= state.selected_vars.size())
                    continue;
                for (const auto &[cid, sel_vid] : state.selected_vars[b])
                {
                    const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                    if (cls.base_eclass_id == base_id)
                    {
                        base_to_selected_[base_id].push_back(sel_vid);
                        selected_to_cached_[sel_vid] = cached_vid;
                    }
                }
            }
        }
        initialized_ = true;
    }

  public:
    uint8_t interestedVarTypes() const override
    {
        return varTypeMask(VarType::CACHED) | varTypeMask(VarType::SELECTED);
    }

    std::string name() const override
    {
        return "CacheFreshnessPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        ensureIndex(state);

        if (changed != kInvalidVarId)
        {
            const VarInfo &info = state.var_infos[changed];
            if (info.type == VarType::CACHED)
            {
                const Domain &dom = state.domains[changed];
                if (dom.isFixed() && dom.fixedValue() == 1)
                {
                    auto it = base_to_selected_.find(info.base_eclass_id);
                    if (it != base_to_selected_.end())
                    {
                        for (VarId sel_vid : it->second)
                        {
                            Domain sel_dom = state.domains[sel_vid];
                            if (sel_dom.contains(0))
                            {
                                sel_dom.remove(0);
                                if (sel_dom.isEmpty())
                                    return false;
                                state.setDomain(sel_vid, sel_dom);
                                worklist.push_back(sel_vid);
                            }
                        }
                    }
                }
            }
            else if (info.type == VarType::SELECTED)
            {
                const Domain &dom = state.domains[changed];
                if (dom.isFixed() && dom.fixedValue() == 0)
                {
                    auto it = selected_to_cached_.find(changed);
                    if (it != selected_to_cached_.end())
                    {
                        VarId cached_vid = it->second;
                        Domain c_dom = state.domains[cached_vid];
                        if (c_dom.contains(1))
                        {
                            c_dom.remove(1);
                            if (c_dom.isEmpty())
                                return false;
                            state.setDomain(cached_vid, c_dom);
                            worklist.push_back(cached_vid);
                        }
                    }
                }
            }
            return true;
        }

        // Initial propagation (changed == kInvalidVarId)
        for (const auto &[base_id, cached_vid] : state.cached_vars)
        {
            const Domain &c_dom = state.domains[cached_vid];
            if (c_dom.isFixed() && c_dom.fixedValue() == 1)
            {
                auto it = base_to_selected_.find(base_id);
                if (it != base_to_selected_.end())
                {
                    for (VarId sel_vid : it->second)
                    {
                        Domain sel_dom = state.domains[sel_vid];
                        if (sel_dom.contains(0))
                        {
                            sel_dom.remove(0);
                            if (sel_dom.isEmpty())
                                return false;
                            state.setDomain(sel_vid, sel_dom);
                            worklist.push_back(sel_vid);
                        }
                    }
                }
            }
        }

        for (const auto &[sel_vid, cached_vid] : selected_to_cached_)
        {
            const Domain &sel_dom = state.domains[sel_vid];
            if (sel_dom.isFixed() && sel_dom.fixedValue() == 0)
            {
                Domain c_dom = state.domains[cached_vid];
                if (c_dom.contains(1))
                {
                    c_dom.remove(1);
                    if (c_dom.isEmpty())
                        return false;
                    state.setDomain(cached_vid, c_dom);
                    worklist.push_back(cached_vid);
                }
            }
        }

        return true;
    }
};

} // namespace plan
