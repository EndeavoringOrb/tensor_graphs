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
                        VarId parent_st_v = state.start_vars[b].at(cid)[en_idx];
                        Domain parent_st_dom = state.domains[parent_st_v];

                        for (EClassId child : enode.getChildren())
                        {
                            EClassId canon_ch = state.bucket_egraphs[b].findConst(child);
                            auto ch_sel_it = state.selected_vars[b].find(canon_ch);
                            if (ch_sel_it != state.selected_vars[b].end())
                            {
                                VarId ch_sel_v = ch_sel_it->second;
                                const Domain &ch_sel_dom = state.domains[ch_sel_v];
                                if (ch_sel_dom.isFixed() && ch_sel_dom.fixedValue() > 0)
                                {
                                    uint32_t ch_en_idx = static_cast<uint32_t>(ch_sel_dom.fixedValue() - 1);
                                    VarId child_st_v = state.start_vars[b].at(canon_ch)[ch_en_idx];
                                    Domain child_st_dom = state.domains[child_st_v];

                                    // start[parent] >= start[child] + 1
                                    if (parent_st_dom.setMin(child_st_dom.getMin() + 1))
                                    {
                                        if (parent_st_dom.isEmpty())
                                            return false;
                                        state.setDomain(parent_st_v, parent_st_dom);
                                    }
                                }
                            }
                        }
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

            // Memory non-overlap between concurrent allocations in same MemSpace
            // Collect fixed allocations
            struct AllocEntry
            {
                EClassId cid;
                MemSpace ms;
                int32_t start_time;
                int32_t end_time;
                uint32_t page_offset;
                uint32_t page_size;
            };

            std::vector<AllocEntry> allocs;
            for (EClassId cid : getReachableCids(state, b))
            {
                auto off_it = state.offset_vars[b].find(cid);
                if (off_it == state.offset_vars[b].end())
                    continue;

                VarId sel_v = state.selected_vars[b].at(cid);
                const Domain &sel_dom = state.domains[sel_v];
                if (sel_dom.isFixed() && sel_dom.fixedValue() > 0)
                {
                    uint32_t en_idx = static_cast<uint32_t>(sel_dom.fixedValue() - 1);
                    VarId off_v = off_it->second;
                    VarId st_v = state.start_vars[b].at(cid)[en_idx];

                    if (state.domains[off_v].isFixed() && state.domains[st_v].isFixed())
                    {
                        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                        ENodeId en_id = cls.enodes[en_idx];
                        bool is_view = (en_id.value < state.bucket_enode_infos[b].size()) &&
                                       state.bucket_enode_infos[b][en_id.value].is_view;
                        if (!is_view)
                        {
                            uint32_t p_size = state.bytesToPages(getSizeBytes(cls.shape, cls.dtype), cls.mem_space);
                            int32_t st = state.domains[st_v].fixedValue();
                            allocs.push_back({cid, cls.mem_space, st, st + 1,
                                              static_cast<uint32_t>(state.domains[off_v].fixedValue()), p_size});
                        }
                    }
                }
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
                                return false; // Overlap contradiction!
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
