// tensor_graphs_cpp/core/plan/propagators/required_reachability.hpp
#pragma once

#include "core/plan/propagators/base.hpp"
#include <vector>

namespace plan
{

// RequiredReachabilityPropagator: see docs/core/propagators.md.
class RequiredReachabilityPropagator : public Propagator
{
    std::vector<std::vector<std::vector<uint8_t>>> reachable_from_root_enode_;
    std::vector<VarId> root_vars_;
    bool initialized_ = false;

    void ensureIndex(const SearchState &state)
    {
        if (initialized_)
            return;

        const size_t num_buckets = state.buckets.size();
        reachable_from_root_enode_.resize(num_buckets);
        root_vars_.resize(num_buckets, kInvalidVarId);

        for (uint32_t b = 0; b < num_buckets; ++b)
        {
            if (b >= state.bucket_root_ids.size() || b >= state.selected_vars.size())
                continue;

            const EGraph &egraph = state.bucket_egraphs[b];
            EClassId root_cid = egraph.findConst(state.bucket_root_ids[b]);
            auto root_it = state.selected_vars[b].find(root_cid);
            if (root_it == state.selected_vars[b].end())
                continue;

            root_vars_[b] = root_it->second;
            const size_t num_classes = egraph.classes.size();
            const EClass &root_cls = egraph.getEClass(root_cid);
            const size_t num_root_enodes = root_cls.enodes.size();
            reachable_from_root_enode_[b].resize(num_root_enodes);

            for (size_t en_idx = 0; en_idx < num_root_enodes; ++en_idx)
            {
                auto &reach = reachable_from_root_enode_[b][en_idx];
                reach.assign(num_classes, 0);

                const ENode &enode = egraph.getENode(root_cls.enodes[en_idx]);
                std::vector<EClassId> q;
                for (EClassId ch : enode.getChildren())
                {
                    EClassId canon_ch = egraph.findConst(ch);
                    if (canon_ch.value < num_classes && !reach[canon_ch.value])
                    {
                        reach[canon_ch.value] = 1;
                        q.push_back(canon_ch);
                    }
                }

                for (size_t head = 0; head < q.size(); ++head)
                {
                    EClassId curr = q[head];
                    const EClass &curr_cls = egraph.getEClass(curr);
                    for (ENodeId c_en_id : curr_cls.enodes)
                    {
                        const ENode &c_enode = egraph.getENode(c_en_id);
                        for (EClassId ch : c_enode.getChildren())
                        {
                            EClassId canon_ch = egraph.findConst(ch);
                            if (canon_ch.value < num_classes && !reach[canon_ch.value])
                            {
                                reach[canon_ch.value] = 1;
                                q.push_back(canon_ch);
                            }
                        }
                    }
                }
            }
        }
        initialized_ = true;
    }

  public:
    uint8_t interestedVarTypes() const override
    {
        return varTypeMask(VarType::SELECTED);
    }

    std::string name() const override
    {
        return "RequiredReachabilityPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        ensureIndex(state);

        if (changed != kInvalidVarId)
        {
            const VarInfo &info = state.var_infos[changed];
            if (info.type != VarType::SELECTED)
                return true;

            const Domain &dom = state.domains[changed];
            if (dom.contains(0))
                return true;

            uint32_t b = info.bucket_idx;
            if (b >= root_vars_.size())
                return true;

            VarId root_var = root_vars_[b];
            if (root_var == kInvalidVarId || root_var == changed)
                return true;

            const Domain &root_dom = state.domains[root_var];
            if (root_dom.isEmpty())
                return false;
            if (root_dom.isFixed())
                return true;

            EClassId cid = info.eclass_id;
            Domain new_root_dom = root_dom;
            bool modified = false;

            for (int32_t val = root_dom.getMin(); val <= root_dom.getMax(); ++val)
            {
                if (!root_dom.contains(val) || val <= 0)
                    continue;

                uint32_t en_idx = static_cast<uint32_t>(val - 1);
                if (en_idx >= reachable_from_root_enode_[b].size())
                    continue;

                if (cid.value >= reachable_from_root_enode_[b][en_idx].size() ||
                    !reachable_from_root_enode_[b][en_idx][cid.value])
                {
                    new_root_dom.remove(val);
                    modified = true;
                }
            }

            if (modified)
            {
                if (new_root_dom.isEmpty())
                    return false;
                state.setDomain(root_var, new_root_dom);
                worklist.push_back(root_var);
            }
            return true;
        }

        // Initial propagation
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            if (b >= root_vars_.size())
                continue;
            VarId root_var = root_vars_[b];
            if (root_var == kInvalidVarId)
                continue;

            const Domain &root_dom = state.domains[root_var];
            if (root_dom.isEmpty())
                return false;
            if (root_dom.isFixed())
                continue;

            Domain new_root_dom = root_dom;
            bool modified = false;

            if (b < state.selected_vars.size())
            {
                for (const auto &[cid, sel_var] : state.selected_vars[b])
                {
                    if (sel_var == root_var)
                        continue;
                    const Domain &dom = state.domains[sel_var];
                    if (dom.contains(0))
                        continue;

                    for (int32_t val = new_root_dom.getMin(); val <= new_root_dom.getMax(); ++val)
                    {
                        if (!new_root_dom.contains(val) || val <= 0)
                            continue;

                        uint32_t en_idx = static_cast<uint32_t>(val - 1);
                        if (en_idx >= reachable_from_root_enode_[b].size())
                            continue;

                        if (cid.value >= reachable_from_root_enode_[b][en_idx].size() ||
                            !reachable_from_root_enode_[b][en_idx][cid.value])
                        {
                            new_root_dom.remove(val);
                            modified = true;
                        }
                    }
                }
            }

            if (modified)
            {
                if (new_root_dom.isEmpty())
                    return false;
                state.setDomain(root_var, new_root_dom);
                worklist.push_back(root_var);
            }
        }

        return true;
    }
};

} // namespace plan
