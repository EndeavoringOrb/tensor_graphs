// tensor_graphs_cpp/core/plan/propagators/input_prune_start_precedence.hpp
#pragma once

#include <array>

#include "core/plan/propagators/base.hpp"

namespace plan
{

// InputPruneStartPrecedencePropagator: see docs/core/propagators.md.
class InputPruneStartPrecedencePropagator : public Propagator
{
    struct CachedChild
    {
        EClassId cid;
        VarId selection_var = kInvalidVarId;
        VarId start_var = kInvalidVarId;
    };

    struct CachedEClass
    {
        std::vector<std::vector<CachedChild>> children_by_alternative;
    };

    std::vector<std::vector<CachedEClass>> cached_eclasses_;
    std::vector<uint32_t> input_occurrence_counts_;
    std::vector<uint32_t> input_occurrence_stamps_;
    std::vector<EClassId> touched_inputs_;
    uint32_t input_occurrence_epoch_ = 0;

    void prepareBucket(SearchState &state, uint32_t b)
    {
        const size_t num_classes = state.bucket_egraphs[b].classes.size();
        if (cached_eclasses_.size() <= b)
            cached_eclasses_.resize(b + 1);
        auto &cached_eclasses = cached_eclasses_[b];
        if (cached_eclasses.size() >= num_classes)
            return;
        const size_t old_size = cached_eclasses.size();
        cached_eclasses.resize(num_classes);
        for (const auto &selected_pair : state.selected_vars[b])
        {
            const EClassId cid = selected_pair.first;
            if (cid.value < old_size || cid.value >= num_classes)
                continue;
            const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
            auto &cached_class = cached_eclasses[cid.value];
            cached_class.children_by_alternative.resize(cls.enodes.size());
            for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
            {
                const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                auto &cached_children = cached_class.children_by_alternative[en_idx];
                cached_children.reserve(enode.getChildren().size());
                for (EClassId child : enode.getChildren())
                {
                    const EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                    const auto child_selection = state.selected_vars[b].find(canon_child);
                    const auto child_start = state.start_vars[b].find(canon_child);
                    cached_children.push_back({
                        canon_child,
                        child_selection == state.selected_vars[b].end() ? kInvalidVarId : child_selection->second,
                        child_start == state.start_vars[b].end() ? kInvalidVarId : child_start->second});
                }
            }
        }
    }

    bool pruneEClass(SearchState &state, uint32_t b, EClassId cid, std::vector<VarId> &worklist,
                     VarId known_start_var = kInvalidVarId, VarId known_selection_var = kInvalidVarId)
    {
        VarId sel_v = known_selection_var;
        if (sel_v == kInvalidVarId)
        {
            auto sel_it = state.selected_vars[b].find(cid);
            if (sel_it == state.selected_vars[b].end())
                return true;
            sel_v = sel_it->second;
        }
        const Domain &current_selection = state.domains[sel_v];
        if (current_selection.contains(0))
            return true;

        VarId st_var = known_start_var;
        if (st_var == kInvalidVarId)
        {
            auto st_it = state.start_vars[b].find(cid);
            if (st_it == state.start_vars[b].end())
                return true;
            st_var = st_it->second;
        }

        prepareBucket(state, b);
        if (cid.value >= cached_eclasses_[b].size())
            return true;
        const auto &cached_children_by_alternative = cached_eclasses_[b][cid.value].children_by_alternative;
        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        bool sel_modified = false;
        Domain sel_dom;
        const int32_t cand_start_max = state.domains[st_var].getMax();

        for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
        {
            int32_t cand_val = static_cast<int32_t>(en_idx + 1);
            if (!(sel_modified ? sel_dom : current_selection).contains(cand_val))
                continue;
            bool remove_cand = false;

            for (const CachedChild &child : cached_children_by_alternative[en_idx])
            {
                if (child.selection_var == kInvalidVarId || child.start_var == kInvalidVarId)
                    continue;

                const Domain &ch_sel_dom = state.domains[child.selection_var];
                if (ch_sel_dom.isFixed() && ch_sel_dom.fixedValue() == 0)
                    continue;
                int32_t input_start_min = state.domains[child.start_var].getMin();
                if (input_start_min != INT32_MAX && input_start_min >= cand_start_max)
                {
                    remove_cand = true;
                    break;
                }
            }

            if (remove_cand)
            {
                if ((sel_modified ? sel_dom : current_selection).isFixed())
                    return false;
                if (!sel_modified)
                    sel_dom = current_selection;
                sel_dom.remove(cand_val);
                sel_modified = true;
                if (sel_dom.isEmpty())
                    return false;
            }
        }

        if (sel_modified)
        {
            state.setDomain(sel_v, sel_dom);
            worklist.push_back(sel_v);
        }

        const Domain &remaining_selection = sel_modified ? sel_dom : current_selection;

        int32_t min_start_max_C = INT32_MAX;
        uint32_t num_remaining_cands = 0;
        auto &touched_inputs = touched_inputs_;
        touched_inputs.clear();
        const size_t num_classes = state.bucket_egraphs[b].classes.size();
        if (input_occurrence_counts_.size() < num_classes)
        {
            input_occurrence_counts_.resize(num_classes, 0);
            input_occurrence_stamps_.resize(num_classes, 0);
        }
        if (++input_occurrence_epoch_ == 0)
        {
            std::fill(input_occurrence_stamps_.begin(), input_occurrence_stamps_.end(), 0);
            input_occurrence_epoch_ = 1;
        }
        const uint32_t occurrence_epoch = input_occurrence_epoch_;

        for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
        {
            int32_t cand_val = static_cast<int32_t>(en_idx + 1);
            if (!remaining_selection.contains(cand_val))
                continue;
            ++num_remaining_cands;
            min_start_max_C = cand_start_max;

            std::array<EClassId, 8> seen_children{};
            size_t seen_child_count = 0;
            const auto &children = cached_children_by_alternative[en_idx];
            for (size_t child_idx = 0; child_idx < children.size(); ++child_idx)
            {
                const EClassId canon_child = children[child_idx].cid;
                bool duplicate = false;
                const size_t prior_count = std::min(child_idx, seen_child_count);
                for (size_t prior_idx = 0; prior_idx < prior_count; ++prior_idx)
                {
                    if (seen_children[prior_idx] == canon_child)
                    {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate && child_idx >= seen_children.size())
                {
                    for (size_t prior_idx = seen_children.size(); prior_idx < child_idx; ++prior_idx)
                    {
                        if (children[prior_idx].cid == canon_child)
                        {
                            duplicate = true;
                            break;
                        }
                    }
                }
                if (duplicate)
                    continue;
                if (seen_child_count < seen_children.size())
                    seen_children[seen_child_count++] = canon_child;
                if (canon_child.value >= input_occurrence_counts_.size())
                    continue;
                if (input_occurrence_stamps_[canon_child.value] != occurrence_epoch)
                {
                    input_occurrence_stamps_[canon_child.value] = occurrence_epoch;
                    input_occurrence_counts_[canon_child.value] = 0;
                    touched_inputs.push_back(canon_child);
                }
                ++input_occurrence_counts_[canon_child.value];
            }
        }

        if (num_remaining_cands > 0 && min_start_max_C != INT32_MAX)
        {
            int32_t max_allowed_input_start = min_start_max_C - 1;
            for (EClassId input_cid : touched_inputs)
            {
                if (input_occurrence_counts_[input_cid.value] == num_remaining_cands)
                {
                    auto in_st_it = state.start_vars[b].find(input_cid);
                    if (in_st_it != state.start_vars[b].end())
                    {
                        VarId in_st_v = in_st_it->second;
                        Domain in_st_dom = state.domains[in_st_v];
                        if (in_st_dom.setMax(max_allowed_input_start))
                        {
                            if (in_st_dom.isEmpty())
                                return false;
                            state.setDomain(in_st_v, in_st_dom);
                            worklist.push_back(in_st_v);
                        }
                    }
                }
            }
        }

        return true;
    }

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::START); }
    StartSelectionGuard startSelectionGuard() const override { return StartSelectionGuard::NON_OPTIONAL; }

    std::string name() const override
    {
        return "InputPruneStartPrecedencePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::START)
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            return pruneEClass(state, b, cid, worklist, changed,
                               state.var_infos[changed].selection_var);
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    if (!pruneEClass(state, b, pair.first, worklist))
                        return false;
                }
            }
        }
        return true;
    }
};

} // namespace plan
