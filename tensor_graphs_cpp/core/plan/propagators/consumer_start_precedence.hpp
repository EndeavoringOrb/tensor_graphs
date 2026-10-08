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
        const int32_t required_consumer_start = min_start + 1;
        auto &frontier = precedence.frontier;
        frontier.clear();
        frontier.push_back(cid);
        if (cid.value < precedence.visited_stamp.size())
            precedence.visited_stamp[cid.value] = stamp;

        for (size_t head = 0; head < frontier.size(); ++head)
        {
            const EClassId current = frontier[head];
            if (current.value >= precedence.consumers_by_cid.size())
                continue;
            const auto &consumers = precedence.consumers_by_cid[current.value];
            if (consumers.empty())
                continue;

            for (const auto &consumer : consumers)
            {
                const VarId p_sel_v = consumer.selection_var;
                const Domain &p_sel_dom = state.domains[p_sel_v];
                if (p_sel_dom.isFixed() && p_sel_dom.fixedValue() == 0)
                    continue;

                const VarId p_st_v = consumer.start_var;
                const Domain &start_dom = state.domains[p_st_v];
                const bool cannot_start = start_dom.isEmpty() || start_dom.getMax() < required_consumer_start;

                if (cannot_start)
                {
                    Domain updated_selection = p_sel_dom;
                    bool domain_changed = false;
                    if (updated_selection.is_mask && updated_selection.min_val == 0 && consumer.total_enodes <= 31)
                    {
                        const uint32_t cand_bits = updated_selection.mask >> 1;
                        const uint32_t remove_bits = cand_bits & consumer.dep_mask;
                        if (remove_bits != 0)
                        {
                            if (updated_selection.isFixed())
                                return false;
                            updated_selection.mask &= ~(remove_bits << 1);
                            if (updated_selection.mask == 0)
                                return false;
                            domain_changed = true;
                        }
                    }
                    else
                    {
                        for (uint32_t dep_idx : consumer.dep_indices)
                        {
                            const int32_t sel_val = static_cast<int32_t>(dep_idx + 1);
                            if (updated_selection.contains(sel_val))
                            {
                                if (updated_selection.isFixed())
                                    return false;
                                updated_selection.remove(sel_val);
                                if (updated_selection.isEmpty())
                                    return false;
                                domain_changed = true;
                            }
                        }
                    }
                    if (domain_changed)
                    {
                        state.setDomain(p_sel_v, updated_selection);
                    }
                }
                else
                {
                    bool all_candidates_depend = true;
                    bool any_candidate = false;
                    bool can_be_view = false;

                    if (p_sel_dom.is_mask && p_sel_dom.min_val == 0 && consumer.total_enodes <= 31)
                    {
                        const uint32_t cand_bits = p_sel_dom.mask >> 1;
                        if (cand_bits != 0)
                        {
                            any_candidate = true;
                            all_candidates_depend = ((cand_bits & ~consumer.dep_mask) == 0);
                        }
                        else
                        {
                            all_candidates_depend = false;
                        }
                        can_be_view = (cand_bits & consumer.view_mask) != 0;
                    }
                    else
                    {
                        for (uint32_t p_en_idx = 0; p_en_idx < consumer.total_enodes; ++p_en_idx)
                        {
                            const int32_t sel_val = static_cast<int32_t>(p_en_idx + 1);
                            if (!p_sel_dom.contains(sel_val))
                                continue;
                            any_candidate = true;
                            if (p_en_idx < 32)
                            {
                                if ((consumer.dep_mask & (1u << p_en_idx)) == 0)
                                    all_candidates_depend = false;
                            }
                            else
                            {
                                if (std::find(consumer.dep_indices.begin(), consumer.dep_indices.end(), p_en_idx) ==
                                    consumer.dep_indices.end())
                                    all_candidates_depend = false;
                            }
                        }
                        for (uint32_t view_idx : consumer.view_indices)
                        {
                            if (p_sel_dom.contains(static_cast<int32_t>(view_idx + 1)))
                            {
                                can_be_view = true;
                                break;
                            }
                        }
                    }

                    if (any_candidate && all_candidates_depend && !p_sel_dom.contains(0))
                    {
                        Domain new_st_dom = state.domains[p_st_v];
                        if (new_st_dom.setMin(required_consumer_start))
                        {
                            if (new_st_dom.isEmpty())
                                return false;
                            state.setDomain(p_st_v, new_st_dom);
                        }
                    }

                    const EClassId p_cid = consumer.parent_cid;
                    if (can_be_view && p_cid.value < precedence.visited_stamp.size() &&
                        precedence.visited_stamp[p_cid.value] != stamp)
                    {
                        precedence.visited_stamp[p_cid.value] = stamp;
                        frontier.push_back(p_cid);
                    }
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

    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::START); }
    StartSelectionGuard startSelectionGuard() const override
    {
        return fixed_starts_only ? StartSelectionGuard::FIXED_NON_OPTIONAL_WITH_CONSUMERS
                                 : StartSelectionGuard::NON_OPTIONAL_WITH_CONSUMERS;
    }

    std::string name() const override
    {
        return "ConsumerStartPrecedencePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::START)
                return true;
            if (fixed_starts_only && !state.domains[changed].isFixed())
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            const auto &precedence = state.propagation.start_precedence[b];
            VarId sel_v = (cid.value < precedence.selected_by_cid.size())
                              ? precedence.selected_by_cid[cid.value]
                              : kInvalidVarId;
            if (sel_v == kInvalidVarId)
            {
                auto it = state.selected_vars[b].find(cid);
                if (it == state.selected_vars[b].end())
                    return true;
                sel_v = it->second;
            }
            const Domain &sel_dom = state.domains[sel_v];
            if (sel_dom.contains(0))
                return true;
            return propagateStart(state, b, cid, state.domains[changed].getMin());
        }
        else
        {
            state.ensurePropagationState();
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
