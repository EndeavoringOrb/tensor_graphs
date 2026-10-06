// tensor_graphs_cpp/core/plan/propagators/view_offset_helpers.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

struct ViewOffsetHelpers
{
    static void buildBucketViewUsers(const SearchState &state,
                                     std::vector<std::unordered_map<EClassId, std::vector<EClassId>>> &bucket_view_users)
    {
        bucket_view_users.clear();
        bucket_view_users.resize(state.buckets.size());
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &pair : state.selected_vars[b])
            {
                EClassId v_cid = pair.first;
                const EClass &cls = state.bucket_egraphs[b].getEClass(v_cid);
                for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
                {
                    ENodeId en_id = cls.enodes[en_idx];
                    bool is_view = en_id.value < state.bucket_enode_infos[b].size() &&
                                   state.bucket_enode_infos[b][en_id.value].is_view;
                    if (!is_view)
                        continue;
                    const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                    if (!enode.getChildren().empty())
                    {
                        EClassId base_cid = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
                        if (base_cid != v_cid)
                        {
                            auto &users = bucket_view_users[b][base_cid];
                            if (std::find(users.begin(), users.end(), v_cid) == users.end())
                            {
                                users.push_back(v_cid);
                            }
                        }
                    }
                }
            }
        }
    }

    static bool getConfirmedViewBase(const SearchState &state, uint32_t b, EClassId cid, EClassId &out_base_cid)
    {
        auto sel_it = state.selected_vars[b].find(cid);
        if (sel_it == state.selected_vars[b].end())
            return false;
        const Domain &sel_dom = state.domains[sel_it->second];
        if (!sel_dom.isFixed() || sel_dom.fixedValue() <= 0)
            return false;

        uint32_t en_idx = static_cast<uint32_t>(sel_dom.fixedValue() - 1);
        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        if (en_idx >= cls.enodes.size())
            return false;

        ENodeId en_id = cls.enodes[en_idx];
        if (en_id.value >= state.bucket_enode_infos[b].size() ||
            !state.bucket_enode_infos[b][en_id.value].is_view)
            return false;

        const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
        if (enode.getChildren().empty())
            return false;

        out_base_cid = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
        return out_base_cid != cid;
    }

    static bool intersectViewAndBase(SearchState &state, uint32_t b, EClassId view_cid, EClassId base_cid,
                                     std::vector<VarId> &worklist)
    {
        auto base_off_it = state.offset_vars[b].find(base_cid);
        auto view_off_it = state.offset_vars[b].find(view_cid);
        if (base_off_it == state.offset_vars[b].end() || view_off_it == state.offset_vars[b].end())
            return true;

        VarId base_off_v = base_off_it->second;
        VarId view_off_v = view_off_it->second;

        const Domain &dom_base = state.domains[base_off_v];
        const Domain &dom_view = state.domains[view_off_v];

        if (dom_base.isEmpty() || dom_view.isEmpty())
            return false;

        int32_t common_min = std::max(dom_base.getMin(), dom_view.getMin());
        int32_t common_max = std::min(dom_base.getMax(), dom_view.getMax());

        if (common_min > common_max)
            return false; // Contradiction!

        if (dom_base.getMin() != common_min || dom_base.getMax() != common_max)
        {
            if (state.setDomain(base_off_v, Domain::makeRange(common_min, common_max)))
            {
                worklist.push_back(base_off_v);
            }
        }

        if (dom_view.getMin() != common_min || dom_view.getMax() != common_max)
        {
            if (state.setDomain(view_off_v, Domain::makeRange(common_min, common_max)))
            {
                worklist.push_back(view_off_v);
            }
        }

        return true;
    }

    static bool filterUnviableViewEnodes(SearchState &state, uint32_t b, EClassId v_cid, EClassId base_cid,
                                         std::vector<VarId> &worklist)
    {
        auto base_off_it = state.offset_vars[b].find(base_cid);
        auto view_off_it = state.offset_vars[b].find(v_cid);
        if (base_off_it == state.offset_vars[b].end() || view_off_it == state.offset_vars[b].end())
            return true;

        VarId base_off_v = base_off_it->second;
        VarId view_off_v = view_off_it->second;
        const Domain &dom_base = state.domains[base_off_v];
        const Domain &dom_view = state.domains[view_off_v];

        if (dom_base.isEmpty() || dom_view.isEmpty())
            return false;

        int32_t common_min = std::max(dom_base.getMin(), dom_view.getMin());
        int32_t common_max = std::min(dom_base.getMax(), dom_view.getMax());

        if (common_min > common_max)
        {
            auto sel_it = state.selected_vars[b].find(v_cid);
            if (sel_it == state.selected_vars[b].end())
                return true;
            VarId sel_v = sel_it->second;
            Domain sel_dom = state.domains[sel_v];
            if (sel_dom.isFixed() && sel_dom.fixedValue() == 0)
                return true;

            const EClass &cls = state.bucket_egraphs[b].getEClass(v_cid);
            bool domain_modified = false;
            for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
            {
                if (!sel_dom.contains(en_idx + 1))
                    continue;
                ENodeId en_id = cls.enodes[en_idx];
                bool is_view = en_id.value < state.bucket_enode_infos[b].size() &&
                               state.bucket_enode_infos[b][en_id.value].is_view;
                if (!is_view)
                    continue;
                const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                if (!enode.getChildren().empty() &&
                    state.bucket_egraphs[b].findConst(enode.getChildren()[0]) == base_cid)
                {
                    if (sel_dom.isFixed())
                        return false;
                    sel_dom.remove(en_idx + 1);
                    domain_modified = true;
                }
            }
            if (domain_modified)
            {
                if (sel_dom.isEmpty())
                    return false;
                if (state.setDomain(sel_v, sel_dom))
                    worklist.push_back(sel_v);
            }
        }
        return true;
    }
};

} // namespace plan
