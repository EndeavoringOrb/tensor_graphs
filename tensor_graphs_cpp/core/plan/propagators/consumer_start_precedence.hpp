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

            for (const auto &p_info : it->second)
            {
                EClassId p_cid = p_info.parent_cid;
                uint32_t p_en_idx = p_info.en_idx;
                if (p_info.selection_var == kInvalidVarId || p_info.start_var == kInvalidVarId)
                    continue;
                VarId p_sel_v = p_info.selection_var;
                const Domain &p_sel_dom = state.domains[p_sel_v];
                if (!p_sel_dom.contains(static_cast<int32_t>(p_en_idx + 1)))
                    continue;

                VarId p_st_v = p_info.start_var;
                const Domain &current_start_domain = state.domains[p_st_v];
                if (current_start_domain.isEmpty() || current_start_domain.getMin() < required_consumer_start)
                {
                    Domain p_st_dom = current_start_domain;
                    if (p_st_dom.setMin(required_consumer_start))
                    {
                        if (p_st_dom.isEmpty())
                        {
                            if (p_sel_dom.isFixed())
                                return false;
                            Domain new_sel_dom = p_sel_dom;
                            new_sel_dom.remove(static_cast<int32_t>(p_en_idx + 1));
                            if (new_sel_dom.isEmpty())
                                return false;
                            state.setDomain(p_sel_v, new_sel_dom);
                        }
                        else
                        {
                            state.setDomain(p_st_v, p_st_dom);
                        }
                    }
                }
                if (p_info.is_view && precedence.visited_stamp[p_cid.value] != stamp)
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
            uint32_t en_idx = state.var_infos[changed].enode_idx;
            VarId sel_v = state.selected_vars[b].at(cid);
            const Domain &sel_dom = state.domains[sel_v];
            if (!sel_dom.isFixed() || sel_dom.fixedValue() != static_cast<int32_t>(en_idx + 1))
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
                        uint32_t en_idx = sel_dom.fixedValue() - 1;
                        VarId st_v = state.start_vars[b].at(cid)[en_idx];
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
