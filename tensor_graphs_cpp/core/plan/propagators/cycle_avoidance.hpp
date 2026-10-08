// tensor_graphs_cpp/core/plan/propagators/cycle_avoidance.hpp
#pragma once

#include "core/plan/propagators/base.hpp"
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace plan
{

class CycleAvoidancePropagator : public Propagator
{
    struct ParentInfo
    {
        EClassId parent_cid;
        uint32_t en_idx;
    };
    std::vector<std::unordered_map<EClassId, std::vector<ParentInfo>>> bucket_parents;
    bool parents_initialized = false;

    void ensureParents(const SearchState &state)
    {
        if (parents_initialized)
            return;
        bucket_parents.resize(state.buckets.size());
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &pair : state.selected_vars[b])
            {
                EClassId p_cid = pair.first;
                const EClass &cls = state.bucket_egraphs[b].getEClass(p_cid);
                for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
                {
                    const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                    for (EClassId ch : enode.getChildren())
                    {
                        EClassId canon_ch = state.bucket_egraphs[b].findConst(ch);
                        bucket_parents[b][canon_ch].push_back(ParentInfo{p_cid, en_idx});
                    }
                }
            }
        }
        parents_initialized = true;
    }

    void getFixedAncestors(const SearchState &state, uint32_t b, EClassId cid,
                           std::unordered_set<EClassId> &ancestors)
    {
        std::vector<EClassId> frontier = {cid};
        ancestors.insert(cid);

        for (size_t head = 0; head < frontier.size(); ++head)
        {
            EClassId curr = frontier[head];
            auto it = bucket_parents[b].find(curr);
            if (it == bucket_parents[b].end())
                continue;

            for (const auto &p_info : it->second)
            {
                EClassId p_cid = p_info.parent_cid;
                auto sel_it = state.selected_vars[b].find(p_cid);
                if (sel_it == state.selected_vars[b].end())
                    continue;

                const Domain &p_sel_dom = state.domains[sel_it->second];
                if (!p_sel_dom.isFixed() || p_sel_dom.fixedValue() != static_cast<int32_t>(p_info.en_idx + 1))
                    continue;

                if (ancestors.insert(p_cid).second)
                {
                    frontier.push_back(p_cid);
                }
            }
        }
    }

    bool checkIncrementalCycle(const SearchState &state, uint32_t b, EClassId cid, uint32_t en_idx)
    {
        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        if (en_idx >= cls.enodes.size())
            return false;
        const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);

        std::unordered_set<EClassId> targets;
        for (EClassId ch : enode.getChildren())
        {
            EClassId canon_ch = state.bucket_egraphs[b].findConst(ch);
            if (canon_ch == cid)
                return false; // Direct self-cycle
            targets.insert(canon_ch);
        }
        if (targets.empty())
            return true;

        std::vector<EClassId> frontier = {cid};
        std::unordered_set<EClassId> visited = {cid};

        for (size_t head = 0; head < frontier.size(); ++head)
        {
            EClassId curr = frontier[head];
            auto it = bucket_parents[b].find(curr);
            if (it == bucket_parents[b].end())
                continue;

            for (const auto &p_info : it->second)
            {
                EClassId p_cid = p_info.parent_cid;
                auto sel_it = state.selected_vars[b].find(p_cid);
                if (sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &p_sel_dom = state.domains[sel_it->second];
                if (!p_sel_dom.isFixed() || p_sel_dom.fixedValue() != static_cast<int32_t>(p_info.en_idx + 1))
                    continue;

                if (targets.count(p_cid))
                    return false;

                if (visited.insert(p_cid).second)
                {
                    frontier.push_back(p_cid);
                }
            }
        }
        return true;
    }

    bool propagateFixed(SearchState &state, uint32_t b, EClassId initial_cid, uint32_t initial_en_idx)
    {
        ensureParents(state);
        std::vector<std::pair<EClassId, uint32_t>> pending = {{initial_cid, initial_en_idx}};

        while (!pending.empty())
        {
            auto [cid, en_idx] = pending.back();
            pending.pop_back();

            const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
            if (en_idx >= cls.enodes.size())
                return false;
            const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);

            // 1. Compute all fixed ancestors of cid (including cid).
            std::unordered_set<EClassId> ancestors;
            getFixedAncestors(state, b, cid, ancestors);

            // 2. Check each child of the newly fixed e-node.
            for (EClassId ch : enode.getChildren())
            {
                EClassId canon_ch = state.bucket_egraphs[b].findConst(ch);
                if (ancestors.count(canon_ch))
                {
                    return false;
                }

                // 3. Collect all descendants reachable from canon_ch via fixed selections.
                std::vector<EClassId> desc_frontier = {canon_ch};
                std::unordered_set<EClassId> descendants = {canon_ch};

                for (size_t head = 0; head < desc_frontier.size(); ++head)
                {
                    EClassId curr = desc_frontier[head];
                    auto sel_it = state.selected_vars[b].find(curr);
                    if (sel_it == state.selected_vars[b].end())
                        continue;

                    const Domain &curr_dom = state.domains[sel_it->second];
                    if (curr_dom.isFixed() && curr_dom.fixedValue() > 0)
                    {
                        uint32_t curr_en_idx = static_cast<uint32_t>(curr_dom.fixedValue() - 1);
                        const EClass &curr_cls = state.bucket_egraphs[b].getEClass(curr);
                        if (curr_en_idx < curr_cls.enodes.size())
                        {
                            const ENode &curr_enode = state.bucket_egraphs[b].getENode(curr_cls.enodes[curr_en_idx]);
                            for (EClassId next_ch : curr_enode.getChildren())
                            {
                                EClassId canon_next = state.bucket_egraphs[b].findConst(next_ch);
                                if (ancestors.count(canon_next))
                                {
                                    return false;
                                }
                                if (descendants.insert(canon_next).second)
                                {
                                    desc_frontier.push_back(canon_next);
                                }
                            }
                        }
                    }
                }

                // 4. For each descendant: prune any candidate e-node that closes a cycle.
                for (EClassId desc_cid : descendants)
                {
                    auto sel_it = state.selected_vars[b].find(desc_cid);
                    if (sel_it == state.selected_vars[b].end())
                        continue;

                    VarId sel_var = sel_it->second;
                    Domain dom = state.domains[sel_var];
                    if (dom.isEmpty() || dom.isFixed())
                        continue;

                    const EClass &desc_cls = state.bucket_egraphs[b].getEClass(desc_cid);
                    bool modified = false;

                    for (uint32_t cand_idx = 0; cand_idx < desc_cls.enodes.size(); ++cand_idx)
                    {
                        int32_t val = static_cast<int32_t>(cand_idx + 1);
                        if (!dom.contains(val))
                            continue;

                        const ENode &cand_enode = state.bucket_egraphs[b].getENode(desc_cls.enodes[cand_idx]);
                        bool closes_cycle = false;
                        for (EClassId cand_ch : cand_enode.getChildren())
                        {
                            EClassId canon_cand_ch = state.bucket_egraphs[b].findConst(cand_ch);
                            if (canon_cand_ch == desc_cid || ancestors.count(canon_cand_ch))
                            {
                                closes_cycle = true;
                                break;
                            }
                        }

                        if (!closes_cycle && !checkIncrementalCycle(state, b, desc_cid, cand_idx))
                        {
                            closes_cycle = true;
                        }

                        if (closes_cycle)
                        {
                            dom.remove(val);
                            modified = true;
                        }
                    }

                    if (modified)
                    {
                        if (dom.isEmpty())
                            return false;
                        state.setDomain(sel_var, dom);

                        if (dom.isFixed() && dom.fixedValue() > 0)
                        {
                            pending.push_back({desc_cid, static_cast<uint32_t>(dom.fixedValue() - 1)});
                        }
                    }
                }
            }
        }
        return true;
    }

    bool checkCycles(SearchState &state, uint32_t b)
    {
        std::vector<EClassId> active;
        for (const auto &pair : state.selected_vars[b])
        {
            const Domain &dom = state.domains[pair.second];
            if (dom.isFixed() && dom.fixedValue() > 0)
                active.push_back(pair.first);
        }

        std::unordered_map<EClassId, std::vector<EClassId>> adj;
        for (EClassId cid : active)
        {
            uint32_t en_idx = static_cast<uint32_t>(state.domains[state.selected_vars[b].at(cid)].fixedValue() - 1);
            const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
            if (en_idx >= cls.enodes.size())
                continue;
            const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
            for (EClassId ch : enode.getChildren())
            {
                EClassId canon_ch = state.bucket_egraphs[b].findConst(ch);
                if (canon_ch == cid)
                    return false;
                adj[canon_ch].push_back(cid);
            }
        }

        std::unordered_map<EClassId, int> color;
        std::function<bool(EClassId)> dfs = [&](EClassId u) -> bool {
            color[u] = 1;
            auto it = adj.find(u);
            if (it != adj.end())
            {
                for (EClassId v : it->second)
                {
                    int c = color[v];
                    if (c == 1)
                        return false;
                    if (c == 0 && !dfs(v))
                        return false;
                }
            }
            color[u] = 2;
            return true;
        };

        for (EClassId cid : active)
        {
            if (color[cid] == 0 && !dfs(cid))
                return false;
        }
        return true;
    }

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }

    std::string name() const override
    {
        return "CycleAvoidancePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;

            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            const Domain &dom = state.domains[changed];

            if (dom.isEmpty())
                return false;

            if (dom.isFixed() && dom.fixedValue() > 0)
            {
                return propagateFixed(state, b, cid, static_cast<uint32_t>(dom.fixedValue() - 1));
            }
            return true;
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                if (!checkCycles(state, b))
                    return false;

                for (const auto &pair : state.selected_vars[b])
                {
                    const Domain &dom = state.domains[pair.second];
                    if (dom.isFixed() && dom.fixedValue() > 0)
                    {
                        if (!propagateFixed(state, b, pair.first, static_cast<uint32_t>(dom.fixedValue() - 1)))
                            return false;
                    }
                }
            }
            return true;
        }
    }
};

} // namespace plan
