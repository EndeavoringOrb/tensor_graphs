// tensor_graphs_cpp/core/plan/propagators/start_unique.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// StartUniquePropagator: see docs/core/propagators.md.
class StartUniquePropagator : public Propagator
{
    // Per-bucket active lists, each rebuilt only when that bucket is used
    // after a selection change.
    std::vector<uint64_t> active_start_selection_revision_;
    std::vector<std::vector<VarId>> active_start_vars_;

    void ensureActiveStarts(SearchState &state, uint32_t b)
    {
        const uint64_t selection_revision = state.getSelectionRevision();
        if (active_start_vars_.size() != state.buckets.size())
        {
            active_start_vars_.assign(state.buckets.size(), {});
            active_start_selection_revision_.assign(state.buckets.size(), UINT64_MAX);
        }
        if (active_start_selection_revision_[b] == selection_revision)
            return;

        auto &active = active_start_vars_[b];
        active.clear();
        active.reserve(state.selected_vars[b].size());
        for (const auto &pair : state.selected_vars[b])
        {
            if (!state.domains[pair.second].contains(0))
                active.push_back(state.start_vars[b].at(pair.first));
        }
        std::sort(active.begin(), active.end());
        active_start_selection_revision_[b] = selection_revision;
    }

    bool removeStartVal(SearchState &state, uint32_t b, VarId source_st_v, int32_t st_val)
    {
        ensureActiveStarts(state, b);
        for (VarId other_st_v : active_start_vars_[b])
        {
            if (other_st_v == source_st_v)
                continue;
            const Domain &other_st_dom = state.domains[other_st_v];
            if (!other_st_dom.contains(st_val))
                continue;
            if (other_st_dom.isFixed())
                return false;
            Domain next_st_dom = other_st_dom;
            next_st_dom.remove(st_val);
            if (next_st_dom.isEmpty())
                return false;
            state.setDomain(other_st_v, next_st_dom);
        }
        return true;
    }

    // The bucket-level form of removeStartVal over every fixed active start:
    // fixed starts must be distinct, and their values leave the unfixed
    // active starts' domains. O(A log A) instead of O(A) per fixed start.
    bool propagateBucket(SearchState &state, uint32_t b)
    {
        ensureActiveStarts(state, b);
        std::vector<std::pair<int32_t, VarId>> fixed;
        std::vector<VarId> unfixed;
        for (VarId st_v : active_start_vars_[b])
        {
            const Domain &dom = state.domains[st_v];
            if (dom.isEmpty())
                return false;
            if (dom.isFixed())
                fixed.emplace_back(dom.fixedValue(), st_v);
            else
                unfixed.push_back(st_v);
        }
        std::sort(fixed.begin(), fixed.end());
        for (size_t i = 1; i < fixed.size(); ++i)
            if (fixed[i].first == fixed[i - 1].first)
                return false;
        for (VarId st_v : unfixed)
        {
            Domain next = state.domains[st_v];
            bool changed = false;
            for (const auto &[value, var] : fixed)
                if (next.contains(value))
                {
                    next.remove(value);
                    changed = true;
                }
            if (!changed)
                continue;
            if (next.isEmpty())
                return false;
            state.setDomain(st_v, next);
        }
        return true;
    }

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::START); }
    StartSelectionGuard startSelectionGuard() const override { return StartSelectionGuard::FIXED_START_POSITIVE; }
    // While a complete plan is validated, every active start is fixed: one
    // bucket-level pass per flush replaces a scan of the bucket per start.
    PropagationBatch propagationBatch(const SearchState &state, VarId) const override
    {
        return state.completing_plan ? PropagationBatch::BUCKET : PropagationBatch::NONE;
    }

    std::string name() const override
    {
        return "StartUniquePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::START)
                return true;
            const Domain &dom = state.domains[changed];
            if (!dom.isFixed())
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            if (state.completing_plan)
                return propagateBucket(state, b);
            EClassId cid = state.var_infos[changed].eclass_id;
            VarId sel_v = state.selected_vars[b].at(cid);
            const Domain &sel_dom = state.domains[sel_v];
            if (!sel_dom.isFixed() || sel_dom.fixedValue() <= 0)
                return true;
            return removeStartVal(state, b, changed, dom.fixedValue());
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    EClassId cid = pair.first;
                    VarId sel_v = pair.second;
                    const Domain &sel_dom = state.domains[sel_v];
                    if (!sel_dom.isFixed() || sel_dom.fixedValue() <= 0)
                        continue;
                    VarId st_v = state.start_vars[b].at(cid);
                    const Domain &st_dom = state.domains[st_v];
                    if (st_dom.isFixed())
                    {
                        if (!removeStartVal(state, b, st_v, st_dom.fixedValue()))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};

} // namespace plan
