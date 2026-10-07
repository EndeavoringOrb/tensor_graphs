// tensor_graphs_cpp/core/plan/propagators/consumer_start_precedence.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

// ConsumerStartPrecedencePropagator: see docs/core/propagators.md.
class ConsumerStartPrecedencePropagator : public Propagator
{
    bool fixed_starts_only;

    bool propagateStart(SearchState &state, uint32_t b, EClassId cid, int32_t min_start)
    {
        auto &precedence = state.propagation.start_precedence[b];
        ++precedence.stamp;
        if (precedence.stamp == 0)
        {
            std::fill(precedence.visited_stamp.begin(), precedence.visited_stamp.end(), 0);
            precedence.stamp = 1;
        }
        const uint32_t stamp = precedence.stamp;
        int32_t required_consumer_start = min_start + 1;
        auto &frontier = precedence.frontier;
        frontier.clear();
        frontier.push_back(cid);
        precedence.visited_stamp[cid.value] = stamp;

        for (size_t head = 0; head < frontier.size(); ++head)
        {
            EClassId current = frontier[head];
            auto it = precedence.parents.find(current);
            if (it == precedence.parents.end())
                continue;

            std::unordered_map<EClassId, std::unordered_set<uint32_t>> dependent_alternatives;
            std::unordered_map<EClassId, std::unordered_set<uint32_t>> view_alternatives;
            for (const auto &p_info : it->second)
            {
                if (p_info.selection_var == kInvalidVarId || p_info.start_var == kInvalidVarId)
                    continue;
                dependent_alternatives[p_info.parent_cid].insert(p_info.en_idx);
                if (p_info.is_view)
                    view_alternatives[p_info.parent_cid].insert(p_info.en_idx);
            }

            for (const auto &parent_entry : dependent_alternatives)
            {
                const EClassId p_cid = parent_entry.first;
                const VarId p_sel_v = state.selected_vars[b].at(p_cid);
                const auto p_info_it = std::find_if(it->second.begin(), it->second.end(), [&](const auto &p_info) {
                    return p_info.parent_cid == p_cid;
                });
                if (p_info_it == it->second.end())
                    continue;
                VarId p_st_v = p_info_it->start_var;
                Domain p_sel_dom = state.domains[p_sel_v];
                if (p_sel_dom.isFixed() && p_sel_dom.fixedValue() == 0)
                    continue;

                const EClass &p_cls = state.bucket_egraphs[b].getEClass(p_cid);
                bool all_candidates_depend = true;
                bool any_candidate = false;
                for (uint32_t p_en_idx = 0; p_en_idx < p_cls.enodes.size(); ++p_en_idx)
                {
                    const int32_t selection_value = static_cast<int32_t>(p_en_idx + 1);
                    if (!p_sel_dom.contains(selection_value))
                        continue;
                    any_candidate = true;
                    if (parent_entry.second.count(p_en_idx) == 0)
                    {
                        all_candidates_depend = false;
                        continue;
                    }
                    const Domain &start_domain = state.domains[p_st_v];
                    if (start_domain.isEmpty() || start_domain.getMax() < required_consumer_start)
                    {
                        if (p_sel_dom.isFixed())
                            return false;
                        p_sel_dom.remove(selection_value);
                        if (p_sel_dom.isEmpty())
                            return false;
                        state.setDomain(p_sel_v, p_sel_dom);
                    }
                }
                if (any_candidate && all_candidates_depend && !p_sel_dom.contains(0))
                {
                    Domain start_domain = state.domains[p_st_v];
                    if (start_domain.setMin(required_consumer_start))
                    {
                        if (start_domain.isEmpty())
                            return false;
                        state.setDomain(p_st_v, start_domain);
                    }
                }
                bool can_be_view = false;
                const auto view_it = view_alternatives.find(p_cid);
                if (view_it != view_alternatives.end())
                    for (uint32_t view_idx : view_it->second)
                        can_be_view = can_be_view || p_sel_dom.contains(static_cast<int32_t>(view_idx + 1));
                if (can_be_view && precedence.visited_stamp[p_cid.value] != stamp)
                {
                    precedence.visited_stamp[p_cid.value] = stamp;
                    frontier.push_back(p_cid);
                }
            }
        }
        return true;
    }

  public:
    explicit ConsumerStartPrecedencePropagator(bool fixed_starts_only = false)
        : fixed_starts_only(fixed_starts_only)
    {
    }

    std::string name() const override
    {
        return "ConsumerStartPrecedencePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        state.ensurePropagationState();
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::START)
                return true;
            if (fixed_starts_only && !state.domains[changed].isFixed())
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            VarId sel_v = state.selected_vars[b].at(cid);
            const Domain &sel_dom = state.domains[sel_v];
            if (sel_dom.contains(0))
                return true;
            return propagateStart(state, b, cid, state.domains[changed].getMin());
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
                    if (sel_dom.isFixed() && sel_dom.fixedValue() > 0)
                    {
                        VarId st_v = state.start_vars[b].at(cid);
                        if (fixed_starts_only && !state.domains[st_v].isFixed())
                            continue;
                        if (!propagateStart(state, b, cid, state.domains[st_v].getMin()))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};

} // namespace plan
