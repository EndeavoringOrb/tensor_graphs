// tensor_graphs_cpp/core/plan/propagator.hpp
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/constants.hpp"
#include "core/plan/domain.hpp"
#include "core/plan/search_state.hpp"

namespace plan
{

class Propagator
{
  public:
    virtual ~Propagator() = default;
    virtual std::string name() const = 0;

    // Shrinks variable domains in state. Returns false on contradiction.
    virtual bool propagate(SearchState &state) = 0;

    // Returns a lower bound on total makespan / cost.
    virtual float computeLowerBound(const SearchState &state)
    {
        return 0.0f;
    }
};

inline const std::vector<EClassId> &getReachableCids(const SearchState &state, uint32_t b)
{
    static const std::vector<EClassId> empty;
    return (b < state.reachable_cids.size() && !state.reachable_cids[b].empty())
               ? state.reachable_cids[b]
               : empty;
}

class SelectionPropagator : public Propagator
{
  public:
    std::string name() const override
    {
        return "SelectionPropagator";
    }

    bool propagate(SearchState &state) override
    {
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            EClassId root_id = state.bucket_root_ids[b];
            VarId root_var = state.selected_vars[b].at(root_id);

            // Constraint: root eclass must be selected (cannot be 0)
            Domain root_dom = state.domains[root_var];
            if (root_dom.contains(0))
            {
                root_dom.remove(0);
                if (root_dom.isEmpty())
                    return false;
                state.setDomain(root_var, root_dom);
            }

            // Propagate selection implications:
            // For each eclass that is definitely selected and fixed to enode e,
            // all children of enode e cannot be 0.
            for (EClassId cid : getReachableCids(state, b))
            {
                VarId v = state.selected_vars[b].at(cid);
                const Domain &dom = state.domains[v];
                if (dom.isEmpty())
                    return false;

                if (dom.isFixed() && dom.fixedValue() > 0)
                {
                    uint32_t en_idx = static_cast<uint32_t>(dom.fixedValue() - 1);
                    const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                    if (en_idx < cls.enodes.size())
                    {
                        ENodeId en_id = cls.enodes[en_idx];
                        const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                        for (EClassId child : enode.getChildren())
                        {
                            EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                            auto ch_it = state.selected_vars[b].find(canon_child);
                            if (ch_it != state.selected_vars[b].end())
                            {
                                VarId ch_v = ch_it->second;
                                Domain ch_dom = state.domains[ch_v];
                                if (ch_dom.contains(0))
                                {
                                    ch_dom.remove(0);
                                    if (ch_dom.isEmpty())
                                        return false;
                                    state.setDomain(ch_v, ch_dom);
                                }
                            }
                        }
                    }
                }
            }

            // An e-class is only allowed to remain selectable when some
            // currently possible nonzero parent selection can reach it from
            // the root. Without this support pass, every structurally
            // reachable e-class remains in {0, 1, ...} forever, forcing the
            // brancher to enumerate alternatives that are already outside
            // the current hyperbox's selected DAG.
            std::unordered_set<EClassId> potentially_reachable;
            std::vector<EClassId> frontier = {root_id};
            potentially_reachable.insert(root_id);
            size_t frontier_head = 0;
            while (frontier_head < frontier.size())
            {
                EClassId cid = frontier[frontier_head++];
                VarId sel_v = state.selected_vars[b].at(cid);
                const Domain &sel_dom = state.domains[sel_v];
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);

                for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
                {
                    int32_t value = static_cast<int32_t>(en_idx + 1);
                    if (!sel_dom.contains(value))
                        continue;

                    const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                    for (EClassId child : enode.getChildren())
                    {
                        EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                        if (state.selected_vars[b].find(canon_child) != state.selected_vars[b].end() &&
                            potentially_reachable.insert(canon_child).second)
                        {
                            frontier.push_back(canon_child);
                        }
                    }
                }
            }

            for (EClassId cid : getReachableCids(state, b))
            {
                if (cid == root_id || potentially_reachable.count(cid) != 0)
                    continue;

                VarId sel_v = state.selected_vars[b].at(cid);
                Domain sel_dom = state.domains[sel_v];
                if (!sel_dom.contains(0))
                    return false;
                if (!sel_dom.isFixed())
                    state.setDomain(sel_v, Domain::makeFixed(0, true));
            }
        }
        return true;
    }
};

class CachePropagator : public Propagator
{
  public:
    std::string name() const override
    {
        return "CachePropagator";
    }

    bool propagate(SearchState &state) override
    {
        // 1. Check CACHE and SCATTER dependencies on cached_vars
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (EClassId cid : getReachableCids(state, b))
            {
                VarId v = state.selected_vars[b].at(cid);
                Domain dom = state.domains[v];
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);

                for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
                {
                    int32_t val = static_cast<int32_t>(en_idx + 1);
                    if (dom.contains(val))
                    {
                        ENodeId en_id = cls.enodes[en_idx];
                        const ENode &enode = state.bucket_egraphs[b].getENode(en_id);

                        if (enode.getOpType() == OpType::CACHE)
                        {
                            BaseEClassId base_id = cls.base_eclass_id;
                            auto c_it = state.cached_vars.find(base_id);
                            if (c_it != state.cached_vars.end())
                            {
                                VarId cv = c_it->second;
                                // If not cached, cannot choose CACHE enode
                                if (state.domains[cv].isFixed() && state.domains[cv].fixedValue() == 0)
                                {
                                    dom.remove(val);
                                    if (dom.isEmpty())
                                        return false;
                                    state.setDomain(v, dom);
                                }
                                else if (dom.isFixed() && dom.fixedValue() == val)
                                {
                                    // Definitely selected CACHE enode -> must be cached
                                    Domain c_dom = state.domains[cv];
                                    c_dom.remove(0);
                                    if (c_dom.isEmpty())
                                        return false;
                                    state.setDomain(cv, c_dom);
                                }
                            }
                        }
                        else if (enode.getOpType() == OpType::SCATTER)
                        {
                            BaseEClassId base_id = cls.base_eclass_id;
                            auto c_it = state.cached_vars.find(base_id);
                            if (c_it != state.cached_vars.end())
                            {
                                VarId cv = c_it->second;
                                if (state.domains[cv].isFixed() && state.domains[cv].fixedValue() == 0)
                                {
                                    dom.remove(val);
                                    if (dom.isEmpty())
                                        return false;
                                    state.setDomain(v, dom);
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Cache memory budget per MemSpace
        std::unordered_map<MemSpace, uint64_t> fixed_cache_bytes;
        for (const auto &cand : state.candidates)
        {
            auto c_it = state.cached_vars.find(cand.base_eclass_id);
            if (c_it != state.cached_vars.end())
            {
                VarId cv = c_it->second;
                if (state.domains[cv].isFixed() && state.domains[cv].fixedValue() == 1)
                {
                    fixed_cache_bytes[cand.mem_space] += cand.size_bytes;
                }
            }
        }

        for (const auto &pair : fixed_cache_bytes)
        {
            uint64_t cap = state.getMemoryCap(pair.first);
            if (pair.second > cap)
                return false;
        }

        // Prune candidates that would exceed cap
        for (const auto &cand : state.candidates)
        {
            auto c_it = state.cached_vars.find(cand.base_eclass_id);
            if (c_it != state.cached_vars.end())
            {
                VarId cv = c_it->second;
                Domain c_dom = state.domains[cv];
                if (!c_dom.isFixed() && c_dom.contains(1))
                {
                    uint64_t cap = state.getMemoryCap(cand.mem_space);
                    if (fixed_cache_bytes[cand.mem_space] + cand.size_bytes > cap)
                    {
                        c_dom.remove(1);
                        if (c_dom.isEmpty())
                            return false;
                        state.setDomain(cv, c_dom);
                    }
                }
            }
        }

        return true;
    }
};

class TopologicalOrderPropagator : public Propagator
{
  public:
    std::string name() const override
    {
        return "TopologicalOrderPropagator";
    }

    bool propagate(SearchState &state) override
    {
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            const auto &reachable = getReachableCids(state, b);

            auto has_fixed_path = [&](EClassId from, EClassId target) {
                std::vector<EClassId> frontier = {from};
                std::unordered_set<EClassId> visited;
                visited.insert(from);

                for (size_t i = 0; i < frontier.size(); ++i)
                {
                    EClassId cid = frontier[i];
                    if (cid == target)
                        return true;

                    auto sel_it = state.selected_vars[b].find(cid);
                    if (sel_it == state.selected_vars[b].end())
                        continue;

                    const Domain &selection = state.domains[sel_it->second];
                    if (!selection.isFixed() || selection.fixedValue() <= 0)
                        continue;

                    uint32_t enode_idx = static_cast<uint32_t>(selection.fixedValue() - 1);
                    const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                    if (enode_idx >= cls.enodes.size())
                        continue;

                    const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[enode_idx]);
                    for (EClassId child : enode.getChildren())
                    {
                        EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                        if (visited.insert(canon_child).second)
                            frontier.push_back(canon_child);
                    }
                }
                return false;
            };

            // Remove enodes that would immediately create a cycle with the
            // already fixed selected graph. This makes the brancher's first
            // choice topologically meaningful instead of waiting for a full
            // selection assignment to discover the cycle.
            for (EClassId cid : reachable)
            {
                VarId sel_v = state.selected_vars[b].at(cid);
                Domain sel_dom = state.domains[sel_v];
                if (sel_dom.isFixed())
                    continue;

                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                bool changed = false;
                for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
                {
                    int32_t value = static_cast<int32_t>(en_idx + 1);
                    if (!sel_dom.contains(value))
                        continue;

                    const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                    bool incompatible = false;
                    for (EClassId child : enode.getChildren())
                    {
                        EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                        auto child_it = state.selected_vars[b].find(canon_child);
                        if (canon_child == cid || child_it == state.selected_vars[b].end() ||
                            state.domains[child_it->second].getMax() <= 0 ||
                            has_fixed_path(canon_child, cid))
                        {
                            incompatible = true;
                            break;
                        }
                    }

                    if (incompatible)
                        changed = sel_dom.remove(value) || changed;
                }

                if (changed)
                {
                    if (sel_dom.isEmpty())
                        return false;
                    state.setDomain(sel_v, sel_dom);
                }
            }

            std::unordered_set<EClassId> active;
            std::unordered_map<EClassId, int> in_degree;
            std::unordered_map<EClassId, std::vector<EClassId>> parents;

            for (EClassId cid : reachable)
            {
                VarId sel_v = state.selected_vars[b].at(cid);
                const Domain &sel_dom = state.domains[sel_v];
                if (sel_dom.isFixed() && sel_dom.fixedValue() > 0)
                {
                    active.insert(cid);
                    in_degree[cid] = 0;
                }
            }

            // Build the selected DAG. The previous implementation walked
            // reachable e-classes in hash-derived order, so a long dependency
            // chain could require many outer propagation rounds before a
            // lower bound reached its consumer.
            for (EClassId cid : active)
            {
                VarId sel_v = state.selected_vars[b].at(cid);
                uint32_t en_idx = static_cast<uint32_t>(state.domains[sel_v].fixedValue() - 1);
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (en_idx >= cls.enodes.size())
                    continue;

                const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                std::unordered_set<EClassId> unique_children;
                for (EClassId child : enode.getChildren())
                {
                    EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                    if (active.count(canon_child) != 0 && unique_children.insert(canon_child).second)
                    {
                        ++in_degree[cid];
                        parents[canon_child].push_back(cid);
                    }
                }
            }

            std::vector<EClassId> queue;
            for (const auto &entry : in_degree)
            {
                if (entry.second == 0)
                    queue.push_back(entry.first);
            }

            std::vector<EClassId> topo_order;
            size_t queue_head = 0;
            while (queue_head < queue.size())
            {
                EClassId child = queue[queue_head++];
                topo_order.push_back(child);
                for (EClassId parent : parents[child])
                {
                    if (--in_degree[parent] == 0)
                        queue.push_back(parent);
                }
            }

            // A selected strict-precedence cycle cannot be scheduled.
            if (topo_order.size() != active.size())
                return false;

            for (EClassId cid : topo_order)
            {
                VarId sel_v = state.selected_vars[b].at(cid);
                uint32_t en_idx = static_cast<uint32_t>(state.domains[sel_v].fixedValue() - 1);
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (en_idx >= cls.enodes.size())
                    continue;

                VarId parent_st_v = state.start_vars[b].at(cid)[en_idx];
                Domain parent_st_dom = state.domains[parent_st_v];
                const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                for (EClassId child : enode.getChildren())
                {
                    EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                    auto child_sel_it = state.selected_vars[b].find(canon_child);
                    if (child_sel_it == state.selected_vars[b].end())
                        continue;

                    VarId child_sel_v = child_sel_it->second;
                    const Domain &child_sel_dom = state.domains[child_sel_v];
                    if (!child_sel_dom.isFixed() || child_sel_dom.fixedValue() <= 0)
                        continue;

                    uint32_t child_en_idx = static_cast<uint32_t>(child_sel_dom.fixedValue() - 1);
                    auto child_start_it = state.start_vars[b].find(canon_child);
                    if (child_start_it == state.start_vars[b].end() ||
                        child_en_idx >= child_start_it->second.size())
                        continue;

                    Domain child_st_dom = state.domains[child_start_it->second[child_en_idx]];
                    if (parent_st_dom.setMin(child_st_dom.getMin() + 1))
                    {
                        if (parent_st_dom.isEmpty())
                            return false;
                        state.setDomain(parent_st_v, parent_st_dom);
                    }
                }
            }
        }
        return true;
    }
};

class EngineSchedulePropagator : public Propagator
{
  public:
    std::string name() const override
    {
        return "EngineSchedulePropagator";
    }

    bool propagate(SearchState &state) override
    {
        // Enforce: only one op can run at a time per engine
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            std::unordered_map<Engine, std::unordered_set<int32_t>> fixed_starts;

            // First collect all slots that are already occupied.  The old
            // implementation only used this table to detect a contradiction
            // after both operations had been fixed, which made a blocked
            // prefix look like a sequence of independent search failures.
            for (EClassId cid : getReachableCids(state, b))
            {
                VarId sel_v = state.selected_vars[b].at(cid);
                const Domain &sel_dom = state.domains[sel_v];
                if (sel_dom.isFixed() && sel_dom.fixedValue() > 0)
                {
                    uint32_t en_idx = static_cast<uint32_t>(sel_dom.fixedValue() - 1);
                    const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                    if (en_idx < cls.enodes.size())
                    {
                        ENodeId en_id = cls.enodes[en_idx];
                        const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                        VarId st_v = state.start_vars[b].at(cid)[en_idx];
                        const Domain &st_dom = state.domains[st_v];

                        if (st_dom.isFixed())
                        {
                            int32_t st_val = st_dom.fixedValue();
                            for (const Engine &eng : enode.getEngines())
                            {
                                if (!fixed_starts[eng].insert(st_val).second)
                                {
                                    // Engine conflict: two ops assigned exact same start time
                                    return false;
                                }
                            }
                        }
                    }
                }
            }

            auto isBlocked = [&](const ENode &enode, int32_t start) {
                for (const Engine &eng : enode.getEngines())
                {
                    auto starts_it = fixed_starts.find(eng);
                    if (starts_it != fixed_starts.end() && starts_it->second.count(start) != 0)
                        return true;
                }
                return false;
            };

            // Start domains are intervals.  We cannot represent arbitrary
            // holes in one Domain, but we can soundly skip every occupied
            // slot at the lower edge in one pass.  This is exactly the case
            // that otherwise causes min-first branching to test 90, 91, ...
            // one node at a time.
            for (EClassId cid : getReachableCids(state, b))
            {
                VarId sel_v = state.selected_vars[b].at(cid);
                const Domain &sel_dom = state.domains[sel_v];
                if (!sel_dom.isFixed() || sel_dom.fixedValue() <= 0)
                    continue;

                uint32_t en_idx = static_cast<uint32_t>(sel_dom.fixedValue() - 1);
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (en_idx >= cls.enodes.size())
                    continue;

                const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                VarId st_v = state.start_vars[b].at(cid)[en_idx];
                Domain st_dom = state.domains[st_v];
                if (st_dom.isEmpty())
                    return false;

                if (st_dom.isFixed())
                    continue;

                int64_t candidate = st_dom.getMin();
                const int32_t upper = st_dom.getMax();
                while (candidate <= upper && isBlocked(enode, static_cast<int32_t>(candidate)))
                    ++candidate;

                if (candidate > upper)
                    return false;

                if (candidate > st_dom.getMin())
                {
                    st_dom.setMin(static_cast<int32_t>(candidate));
                    state.setDomain(st_v, st_dom);
                }
            }
        }
        return true;
    }
};

class MemoryNonOverlapPropagator : public Propagator
{
  public:
    std::string name() const override
    {
        return "MemoryNonOverlapPropagator";
    }

    bool propagate(SearchState &state) override
    {
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            // Propagate view offsets
            for (EClassId cid : getReachableCids(state, b))
            {
                VarId sel_v = state.selected_vars[b].at(cid);
                const Domain &sel_dom = state.domains[sel_v];
                if (sel_dom.isFixed() && sel_dom.fixedValue() > 0)
                {
                    uint32_t en_idx = static_cast<uint32_t>(sel_dom.fixedValue() - 1);
                    const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                    if (en_idx < cls.enodes.size())
                    {
                        ENodeId en_id = cls.enodes[en_idx];
                        bool is_view = (en_id.value < state.bucket_enode_infos[b].size()) &&
                                       state.bucket_enode_infos[b][en_id.value].is_view;

                        if (is_view)
                        {
                            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                            if (!enode.getChildren().empty())
                            {
                                EClassId child_cid = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
                                auto off_view_it = state.offset_vars[b].find(cid);
                                auto off_ch_it = state.offset_vars[b].find(child_cid);

                                if (off_view_it != state.offset_vars[b].end() &&
                                    off_ch_it != state.offset_vars[b].end())
                                {
                                    VarId view_off_v = off_view_it->second;
                                    VarId ch_off_v = off_ch_it->second;
                                    Domain view_dom = state.domains[view_off_v];
                                    Domain ch_dom = state.domains[ch_off_v];

                                    // View offset is aligned with child offset
                                    if (ch_dom.isFixed())
                                    {
                                        view_dom.setRange(ch_dom.fixedValue(), ch_dom.fixedValue());
                                        state.setDomain(view_off_v, view_dom);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Memory non-overlap between allocations with overlapping
            // lifetimes in the same MemSpace.  A tensor is live until its
            // last consumer has finished, not merely until the instruction
            // which produces it has finished.  Treating every allocation as
            // [start, start + 1] permits a consumer's output to overwrite an
            // input while that input is still being read; that was the source
            // of valid-looking plans which produced corrupted model output.
            struct AllocEntry
            {
                EClassId cid;
                MemSpace ms;
                int32_t start_time;
                int32_t end_time;
                uint32_t page_offset;
                uint32_t page_size;
            };

            std::unordered_set<EClassId> active;
            std::unordered_map<EClassId, uint32_t> selected_enodes;
            std::unordered_map<EClassId, bool> selected_views;
            std::unordered_map<EClassId, std::vector<EClassId>> parents;

            for (EClassId cid : getReachableCids(state, b))
            {
                VarId sel_v = state.selected_vars[b].at(cid);
                const Domain &sel_dom = state.domains[sel_v];
                if (!sel_dom.isFixed() || sel_dom.fixedValue() <= 0)
                    continue;

                uint32_t en_idx = static_cast<uint32_t>(sel_dom.fixedValue() - 1);
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (en_idx >= cls.enodes.size())
                    continue;

                ENodeId en_id = cls.enodes[en_idx];
                active.insert(cid);
                selected_enodes[cid] = en_idx;
                selected_views[cid] = en_id.value < state.bucket_enode_infos[b].size() &&
                                       state.bucket_enode_infos[b][en_id.value].is_view;
            }

            // Build the selected dependency DAG.  Edges point from a value to
            // the operations which consume it.
            for (EClassId cid : active)
            {
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                uint32_t en_idx = selected_enodes.at(cid);
                const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                for (EClassId child : enode.getChildren())
                {
                    EClassId canon_child = state.bucket_egraphs[b].findConst(child);
                    if (active.count(canon_child) != 0)
                        parents[canon_child].push_back(cid);
                }
            }

            // Follow selected view nodes to the physical value they alias.
            // This is deliberately local to the propagator to avoid making
            // the search headers depend on planner.hpp.
            auto physical_base = [&](EClassId start) {
                EClassId current = start;
                std::unordered_set<EClassId> visited;
                while (visited.insert(current).second && selected_views[current])
                {
                    const EClass &cls = state.bucket_egraphs[b].getEClass(current);
                    const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[selected_enodes.at(current)]);
                    if (enode.getChildren().empty())
                        break;
                    current = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
                    if (active.count(current) == 0)
                        break;
                }
                return current;
            };

            // For each physical value, collect non-view operations which
            // consume it, walking through zero-cost view nodes.
            std::unordered_map<EClassId, std::unordered_set<EClassId>> consumers;
            for (EClassId child : active)
            {
                EClassId base = physical_base(child);
                std::vector<EClassId> frontier;
                std::unordered_set<EClassId> visited;
                for (EClassId parent : parents[child])
                    frontier.push_back(parent);

                for (size_t i = 0; i < frontier.size(); ++i)
                {
                    EClassId parent = frontier[i];
                    if (!visited.insert(parent).second)
                        continue;
                    if (selected_views[parent])
                    {
                        for (EClassId next : parents[parent])
                            frontier.push_back(next);
                    }
                    else
                    {
                        consumers[base].insert(parent);
                    }
                }
            }

            int32_t max_finish_time = 0;
            for (EClassId cid : active)
            {
                if (selected_views[cid])
                    continue;
                uint32_t en_idx = selected_enodes.at(cid);
                VarId st_v = state.start_vars[b].at(cid)[en_idx];
                if (!state.domains[st_v].isFixed())
                    continue;
                max_finish_time = std::max(max_finish_time, state.domains[st_v].fixedValue() + 1);
            }

            std::vector<AllocEntry> allocs;
            for (EClassId cid : active)
            {
                auto off_it = state.offset_vars[b].find(cid);
                if (off_it == state.offset_vars[b].end())
                    continue;

                if (selected_views[cid])
                    continue;

                uint32_t en_idx = selected_enodes.at(cid);
                VarId off_v = off_it->second;
                VarId st_v = state.start_vars[b].at(cid)[en_idx];
                if (!state.domains[off_v].isFixed() || !state.domains[st_v].isFixed())
                    continue;

                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                uint32_t page_size = state.bytesToPages(getSizeBytes(cls.shape, cls.dtype), cls.mem_space);
                if (page_size == 0)
                    page_size = 1;

                uint32_t page_offset = static_cast<uint32_t>(state.domains[off_v].fixedValue());
                // Inputs and already-materialized values use their physical
                // preallocated offset.  Their search offset variable exists
                // for uniformity but is not the runtime location.
                if (cls.base_eclass_id != BaseEClassId{})
                {
                    auto pre_it = state.preallocated_buffers.find(cls.base_eclass_id);
                    if (pre_it != state.preallocated_buffers.end() && pre_it->second.offset >= 0)
                    {
                        uint32_t align = state.getPageAlignment(cls.mem_space);
                        page_offset = static_cast<uint32_t>(pre_it->second.offset / align);
                        page_size = state.bytesToPages(pre_it->second.size, cls.mem_space);
                    }
                }

                int32_t start_time = state.domains[st_v].fixedValue();
                int32_t end_time = start_time + 1;
                auto consumer_it = consumers.find(cid);
                if (consumer_it == consumers.end() || consumer_it->second.empty())
                {
                    end_time = std::max(end_time, max_finish_time);
                }
                else
                {
                    for (EClassId consumer : consumer_it->second)
                    {
                        uint32_t consumer_en_idx = selected_enodes.at(consumer);
                        VarId consumer_st_v = state.start_vars[b].at(consumer)[consumer_en_idx];
                        if (!state.domains[consumer_st_v].isFixed())
                        {
                            end_time = std::max(end_time, max_finish_time);
                            continue;
                        }
                        end_time = std::max(end_time, state.domains[consumer_st_v].fixedValue() + 1);
                    }
                }

                allocs.push_back({cid, cls.mem_space, start_time, end_time, page_offset, page_size});
            }

            // Check non-overlap for allocations with overlapping lifetimes
            for (size_t i = 0; i < allocs.size(); ++i)
            {
                for (size_t j = i + 1; j < allocs.size(); ++j)
                {
                    if (allocs[i].ms == allocs[j].ms)
                    {
                        // Overlapping lifetime check
                        bool time_overlap = !(allocs[i].end_time <= allocs[j].start_time ||
                                              allocs[j].end_time <= allocs[i].start_time);
                        if (time_overlap)
                        {
                            bool space_overlap = !(allocs[i].page_offset + allocs[i].page_size <= allocs[j].page_offset ||
                                                   allocs[j].page_offset + allocs[j].page_size <= allocs[i].page_offset);
                            if (space_overlap)
                            {
                                LOG(DEBUG) << "[MemoryNonOverlapPropagator] Lifetime overlap: eclass "
                                           << allocs[i].cid << " [" << allocs[i].start_time << ","
                                           << allocs[i].end_time << "] and eclass " << allocs[j].cid << " ["
                                           << allocs[j].start_time << "," << allocs[j].end_time << "] share pages "
                                           << allocs[i].page_offset << ".."
                                           << (allocs[i].page_offset + allocs[i].page_size) << " and "
                                           << allocs[j].page_offset << ".."
                                           << (allocs[j].page_offset + allocs[j].page_size);
                                return false; // Overlap contradiction!
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
};

class CostLowerBoundPropagator : public Propagator
{
  private:
    float best_cost = TGConstants::INF;

  public:
    std::string name() const override
    {
        return "CostLowerBoundPropagator";
    }

    void setBestCost(float c)
    {
        best_cost = c;
    }

    float computeLowerBound(const SearchState &state) override
    {
        float total_lb = 0.0f;
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            float w = (b < state.bucket_weights.size()) ? state.bucket_weights[b] : 1.0f;
            if (w <= 0.0f)
                continue;

            std::unordered_map<Engine, float> engine_work;
            for (EClassId cid : getReachableCids(state, b))
            {
                VarId sel_v = state.selected_vars[b].at(cid);
                const Domain &sel_dom = state.domains[sel_v];
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);

                if (sel_dom.isFixed() && sel_dom.fixedValue() > 0)
                {
                    uint32_t en_idx = static_cast<uint32_t>(sel_dom.fixedValue() - 1);
                    if (en_idx < cls.enodes.size())
                    {
                        ENodeId en_id = cls.enodes[en_idx];
                        const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                        float c = 0.0f;
                        if (en_id.value < state.bucket_enode_infos[b].size())
                            c = state.bucket_enode_infos[b][en_id.value].cost;
                        if (c > 0.0f && c < TGConstants::INF)
                        {
                            for (const Engine &eng : enode.getEngines())
                                engine_work[eng] += c;
                        }
                    }
                }
            }

            float bucket_lb = 0.0f;
            for (const auto &pair : engine_work)
                bucket_lb = std::max(bucket_lb, pair.second);

            total_lb += w * bucket_lb;
        }
        return total_lb;
    }

    bool propagate(SearchState &state) override
    {
        float lb = computeLowerBound(state);
        return lb < best_cost;
    }
};

} // namespace plan
