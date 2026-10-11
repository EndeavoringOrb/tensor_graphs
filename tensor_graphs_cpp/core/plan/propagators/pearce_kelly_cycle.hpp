// tensor_graphs_cpp/core/plan/propagators/pearce_kelly_cycle.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

class PearceKellyCyclePropagator : public Propagator
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

    bool checkIncrementalCycle(SearchState &state, uint32_t b, EClassId cid, uint32_t en_idx)
    {
        ensureParents(state);
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
                    return false; // Direct self-cycle
                adj[canon_ch].push_back(cid);
            }
        }

        std::unordered_map<EClassId, int> color; // 0=unvisited, 1=visiting, 2=visited
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

    bool filterEClassDomain(SearchState &state, uint32_t b, EClassId cid, VarId sel_var)
    {
        Domain dom = state.domains[sel_var];
        if (dom.isEmpty())
            return false;

        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        bool modified = false;

        for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
        {
            int32_t val = static_cast<int32_t>(en_idx + 1);
            if (dom.contains(val))
            {
                if (!checkIncrementalCycle(state, b, cid, en_idx))
                {
                    dom.remove(val);
                    modified = true;
                }
            }
        }

        if (modified)
        {
            if (dom.isEmpty())
                return false;
            state.setDomain(sel_var, dom);
        }
        return true;
    }

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }

    std::string name() const override
    {
        return "PearceKellyCyclePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;

            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;

            if (!filterEClassDomain(state, b, cid, changed))
                return false;

            const Domain &dom = state.domains[changed];
            if (dom.isFixed() && dom.fixedValue() > 0)
            {
                uint32_t en_idx = static_cast<uint32_t>(dom.fixedValue() - 1);
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (en_idx < cls.enodes.size())
                {
                    const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                    for (EClassId ch : enode.getChildren())
                    {
                        EClassId canon_ch = state.bucket_egraphs[b].findConst(ch);
                        auto ch_it = state.selected_vars[b].find(canon_ch);
                        if (ch_it != state.selected_vars[b].end())
                        {
                            if (!filterEClassDomain(state, b, canon_ch, ch_it->second))
                                return false;
                        }
                    }
                }
            }
            return true;
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                if (!checkCycles(state, b))
                    return false;
            }
        }
        return true;
    }
};


} // namespace plan
