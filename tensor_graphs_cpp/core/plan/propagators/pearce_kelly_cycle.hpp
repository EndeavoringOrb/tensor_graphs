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
        VarId parent_selection_var;
    };
    // Per bucket, indexed by canonical child class id.
    std::vector<std::vector<std::vector<ParentInfo>>> bucket_parents;
    bool parents_initialized = false;

    void ensureParents(const SearchState &state)
    {
        if (parents_initialized)
            return;
        bucket_parents.resize(state.buckets.size());
        // Buckets are independent: each fills only its own map.
        ThreadPool::get().parallel_for(static_cast<uint32_t>(state.buckets.size()), [&](uint32_t b) {
            auto &parents = bucket_parents[b];
            parents.resize(state.bucket_egraphs[b].classes.size());
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
                        parents.at(canon_ch.value).push_back(ParentInfo{p_cid, en_idx, pair.second});
                    }
                }
            }
        });
        parents_initialized = true;
    }

    // Marks cid and every class reachable from it upward along parent edges
    // whose parent selection is fixed to the edge's alternative. Choosing an
    // alternative of cid closes a cycle exactly when one of its children is
    // marked (cid itself included: a direct self-cycle). The walk does not
    // depend on which alternative is tested, so a class's alternatives share
    // one walk.
    std::vector<uint32_t> ancestor_stamp_;
    uint32_t ancestor_epoch_ = 0;
    std::vector<EClassId> ancestor_frontier_;
    void markFixedAncestors(const SearchState &state, uint32_t b, EClassId cid)
    {
        ensureParents(state);
        const size_t num_classes = state.bucket_egraphs[b].classes.size();
        if (ancestor_stamp_.size() < num_classes)
            ancestor_stamp_.resize(num_classes, 0);
        if (++ancestor_epoch_ == 0)
        {
            std::fill(ancestor_stamp_.begin(), ancestor_stamp_.end(), 0);
            ancestor_epoch_ = 1;
        }
        ancestor_frontier_.clear();
        ancestor_frontier_.push_back(cid);
        ancestor_stamp_[cid.value] = ancestor_epoch_;
        const auto &parents = bucket_parents[b];
        for (size_t head = 0; head < ancestor_frontier_.size(); ++head)
        {
            const uint32_t current = ancestor_frontier_[head].value;
            if (current >= parents.size())
                continue;
            for (const auto &p_info : parents[current])
            {
                const Domain &p_sel_dom = state.domains[p_info.parent_selection_var];
                if (!p_sel_dom.isFixed() || p_sel_dom.fixedValue() != static_cast<int32_t>(p_info.en_idx + 1))
                    continue;
                if (ancestor_stamp_[p_info.parent_cid.value] != ancestor_epoch_)
                {
                    ancestor_stamp_[p_info.parent_cid.value] = ancestor_epoch_;
                    ancestor_frontier_.push_back(p_info.parent_cid);
                }
            }
        }
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

        markFixedAncestors(state, b, cid);
        for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
        {
            int32_t val = static_cast<int32_t>(en_idx + 1);
            if (!dom.contains(val))
                continue;
            for (EClassId ch : state.bucket_egraphs[b].getENode(cls.enodes[en_idx]).getChildren())
                if (ancestor_stamp_[state.bucket_egraphs[b].findConst(ch).value] == ancestor_epoch_)
                {
                    dom.remove(val);
                    modified = true;
                    break;
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


    // While the root heuristic validates a complete plan (every selection is
    // fixed before propagation), nothing is left to prune: one cycle check
    // of the selected graph per selection revision replaces the per-variable
    // ancestor searches (quadratic over a plan). Returns false (no
    // shortcut) when a selection of the bucket is still open.
    std::vector<uint64_t> verified_revision_;
    bool completePlanCheck(SearchState &state, uint32_t b, bool &consistent)
    {
        if (!state.completing_plan)
            return false;
        if (verified_revision_.size() < state.buckets.size())
            verified_revision_.resize(state.buckets.size(), UINT64_MAX);
        if (verified_revision_[b] == state.getSelectionRevision())
        {
            consistent = true;
            return true;
        }
        for (const auto &pair : state.selected_vars[b])
            if (!state.domains[pair.second].isFixed())
                return false;
        consistent = checkCycles(state, b);
        if (consistent)
            verified_revision_[b] = state.getSelectionRevision();
        return true;
    }

  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }
    void prepare(const SearchState &state) override { ensureParents(state); }

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
            bool consistent = true;
            if (completePlanCheck(state, b, consistent))
                return consistent;

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
