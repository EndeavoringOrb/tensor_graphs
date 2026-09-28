// tensor_graphs_cpp/core/plan/propagator.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/common/constants.hpp"
#include "core/kernels.hpp"
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
    virtual bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) = 0;

    // Returns a lower bound on total makespan / cost.
    virtual float computeLowerBound(const SearchState &state)
    {
        return 0.0f;
    }
};

// ============================================================================
// Helper functions for op inspection and graph queries
// ============================================================================

inline bool isOpCacheOrScatter(const ENode &enode)
{
    if (enode.getOpType() == OpType::CACHE || enode.getOpType() == OpType::SCATTER)
        return true;
    KernelId kid = enode.getKernelId();
    if (kid.value != 0 && KernelRegistry::get().hasKernel(kid))
    {
        const auto &entry = KernelRegistry::get().getKernel(kid);
        if (entry.opType == OpType::CACHE || entry.opType == OpType::SCATTER)
            return true;
        if (entry.opType == OpType::FUSED && entry.refFactory)
        {
            Graph kGraph;
            std::vector<LogicalId> kInputs;
            for (uint64_t i = 0; i < entry.min_num_inputs; ++i)
                kInputs.push_back(kGraph.input(entry.dummyShapes.size() > i ? entry.dummyShapes[i] : std::vector<uint32_t>{1},
                                               entry.dtypes.size() > i ? entry.dtypes[i] : DType::FLOAT32));
            entry.refFactory(kInputs, kGraph);
            for (const auto &pair : kGraph.nodes)
            {
                if (pair.second.opType == OpType::CACHE || pair.second.opType == OpType::SCATTER)
                    return true;
            }
        }
    }
    return false;
}

inline bool isOpRootScatterOrCache(const ENode &enode)
{
    if (enode.getOpType() == OpType::CACHE || enode.getOpType() == OpType::SCATTER)
        return true;
    KernelId kid = enode.getKernelId();
    if (kid.value != 0 && KernelRegistry::get().hasKernel(kid))
    {
        const auto &entry = KernelRegistry::get().getKernel(kid);
        if (entry.opType == OpType::SCATTER)
            return true;
        if (entry.opType == OpType::FUSED && entry.refFactory)
        {
            Graph kGraph;
            std::vector<LogicalId> kInputs;
            for (uint64_t i = 0; i < entry.min_num_inputs; ++i)
                kInputs.push_back(kGraph.input(entry.dummyShapes.size() > i ? entry.dummyShapes[i] : std::vector<uint32_t>{1},
                                               entry.dtypes.size() > i ? entry.dtypes[i] : DType::FLOAT32));
            LogicalId rootId = entry.refFactory(kInputs, kGraph);
            if (kGraph.getNode(rootId).opType == OpType::SCATTER)
                return true;
        }
    }
    return false;
}

// ============================================================================
// CORRECTNESS PROPAGATORS (1 - 11)
// ============================================================================

// 1. if VarType::SELECTED, update reachable, fix unreachable to {0}
class SelectionReachabilityPropagator : public Propagator
{
  public:
    std::string name() const override
    {
        return "SelectionReachabilityPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;
            std::vector<VarId> unreachable;
            state.updateSelectionReachability(changed, unreachable);
            for (VarId sel_v : unreachable)
            {
                const Domain &sel_dom = state.domains[sel_v];
                if (!sel_dom.contains(0))
                    return false;
                if (!sel_dom.isFixed())
                    state.setDomain(sel_v, Domain::makeFixed(0, sel_dom.is_mask));
            }
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                if (b >= state.bucket_root_ids.size())
                    continue;
                EClassId root_cid = state.bucket_root_ids[b];
                auto it = state.selected_vars[b].find(root_cid);
                if (it == state.selected_vars[b].end())
                    continue;
                VarId root_var = it->second;
                std::vector<VarId> unreachable;
                state.updateSelectionReachability(root_var, unreachable);
                for (VarId sel_v : unreachable)
                {
                    const Domain &sel_dom = state.domains[sel_v];
                    if (!sel_dom.contains(0))
                        return false;
                    if (!sel_dom.isFixed())
                        state.setDomain(sel_v, Domain::makeFixed(0, sel_dom.is_mask));
                }
            }
        }
        return true;
    }
};

// 2. if VarType::SELECTED && dom.isFixed() && dom.fixedValue() > 0, for each eclass that is
// definitely selected and fixed to enode e, all children of enode e cannot be 0.
class SelectionChildrenPropagator : public Propagator
{
    bool propagateEClass(SearchState &state, uint32_t b, EClassId cid, int32_t fixed_val)
    {
        if (fixed_val <= 0)
            return true;
        uint32_t en_idx = static_cast<uint32_t>(fixed_val - 1);
        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        if (en_idx >= cls.enodes.size())
            return false;
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
        return true;
    }

  public:
    std::string name() const override
    {
        return "SelectionChildrenPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;
            const Domain &dom = state.domains[changed];
            if (dom.isEmpty())
                return false;
            if (dom.isFixed() && dom.fixedValue() > 0)
            {
                return propagateEClass(state, state.var_infos[changed].bucket_idx,
                                       state.var_infos[changed].eclass_id, dom.fixedValue());
            }
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    const Domain &dom = state.domains[pair.second];
                    if (dom.isFixed() && dom.fixedValue() > 0)
                    {
                        if (!propagateEClass(state, b, pair.first, dom.fixedValue()))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};

// 3. if VarType::SELECTED && dom.isFixed() && dom.fixedValue() == 0, fix start to {0} and offset
// to {min_p} because they don't matter.
class UnselectedStartOffsetPropagator : public Propagator
{
    bool cleanupEClass(SearchState &state, uint32_t b, EClassId cid)
    {
        auto st_it = state.start_vars[b].find(cid);
        if (st_it != state.start_vars[b].end())
        {
            for (VarId st_v : st_it->second)
            {
                const Domain &st_dom = state.domains[st_v];
                if (!st_dom.isFixed() || st_dom.fixedValue() != 0)
                {
                    state.setDomain(st_v, Domain::makeFixed(0, false));
                }
            }
        }
        auto off_it = state.offset_vars[b].find(cid);
        if (off_it != state.offset_vars[b].end())
        {
            VarId off_v = off_it->second;
            const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
            const auto prealloc_it = state.preallocated_pages.find(cls.mem_space);
            uint32_t min_p = (prealloc_it == state.preallocated_pages.end()) ? 0 : prealloc_it->second;
            const Domain &off_dom = state.domains[off_v];
            if (!off_dom.isFixed() || off_dom.fixedValue() != static_cast<int32_t>(min_p))
            {
                state.setDomain(off_v, Domain::makeFixed(min_p, false));
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "UnselectedStartOffsetPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;
            const Domain &dom = state.domains[changed];
            if (dom.isFixed() && dom.fixedValue() == 0)
            {
                return cleanupEClass(state, state.var_infos[changed].bucket_idx,
                                     state.var_infos[changed].eclass_id);
            }
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    const Domain &dom = state.domains[pair.second];
                    if (dom.isFixed() && dom.fixedValue() == 0)
                    {
                        if (!cleanupEClass(state, b, pair.first))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};

// 4. if VarType::CACHED && dom.isFixed() && dom.fixedValue() == 0, remove CACHE/SCATTER (and
// FUSED with CACHE/SCATTER in refFactory graph) from corresponding selected domain in all buckets
class CacheExclusionPropagator : public Propagator
{
    bool excludeForBase(SearchState &state, BaseEClassId base_id)
    {
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &pair : state.selected_vars[b])
            {
                EClassId cid = pair.first;
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                if (cls.base_eclass_id != base_id)
                    continue;

                VarId sel_v = pair.second;
                Domain sel_dom = state.domains[sel_v];
                bool changed = false;
                for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
                {
                    int32_t val = static_cast<int32_t>(en_idx + 1);
                    if (!sel_dom.contains(val))
                        continue;
                    const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
                    if (isOpCacheOrScatter(enode))
                    {
                        changed = sel_dom.remove(val) || changed;
                    }
                }
                if (changed)
                {
                    if (sel_dom.isEmpty())
                        return false;
                    state.setDomain(sel_v, sel_dom);
                }
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "CacheExclusionPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::CACHED)
                return true;
            const Domain &dom = state.domains[changed];
            if (dom.isFixed() && dom.fixedValue() == 0)
            {
                return excludeForBase(state, state.var_infos[changed].base_eclass_id);
            }
        }
        else
        {
            for (const auto &pair : state.cached_vars)
            {
                const Domain &dom = state.domains[pair.second];
                if (dom.isFixed() && dom.fixedValue() == 0)
                {
                    if (!excludeForBase(state, pair.first))
                        return false;
                }
            }
        }
        return true;
    }
};

// 5. if VarType::START, and corresponding selected (dom.isFixed() && dom.fixedValue() > 0), all
// consumers (including through views) starts setMin(changed start + 1)
class StartPrecedencePropagator : public Propagator
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

    bool propagateStart(SearchState &state, uint32_t b, EClassId cid, uint32_t en_idx, int32_t min_start)
    {
        ensureParents(state);
        int32_t required_consumer_start = min_start + 1;
        std::vector<EClassId> frontier = {cid};
        std::unordered_set<EClassId> visited = {cid};

        for (size_t head = 0; head < frontier.size(); ++head)
        {
            EClassId current = frontier[head];
            auto it = bucket_parents[b].find(current);
            if (it == bucket_parents[b].end())
                continue;

            for (const auto &p_info : it->second)
            {
                EClassId p_cid = p_info.parent_cid;
                uint32_t p_en_idx = p_info.en_idx;

                auto sel_it = state.selected_vars[b].find(p_cid);
                if (sel_it == state.selected_vars[b].end())
                    continue;
                VarId p_sel_v = sel_it->second;
                const Domain &p_sel_dom = state.domains[p_sel_v];
                if (!p_sel_dom.contains(static_cast<int32_t>(p_en_idx + 1)))
                    continue;

                auto st_it = state.start_vars[b].find(p_cid);
                if (st_it == state.start_vars[b].end() || p_en_idx >= st_it->second.size())
                    continue;

                VarId p_st_v = st_it->second[p_en_idx];
                Domain p_st_dom = state.domains[p_st_v];
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
                ENodeId p_en_id = state.bucket_egraphs[b].getEClass(p_cid).enodes[p_en_idx];
                bool is_view = p_en_id.value < state.bucket_enode_infos[b].size() &&
                               state.bucket_enode_infos[b][p_en_id.value].is_view;
                if (is_view && visited.insert(p_cid).second)
                {
                    frontier.push_back(p_cid);
                }
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "StartPrecedencePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::START)
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            uint32_t en_idx = state.var_infos[changed].enode_idx;
            VarId sel_v = state.selected_vars[b].at(cid);
            const Domain &sel_dom = state.domains[sel_v];
            if (!sel_dom.isFixed() || sel_dom.fixedValue() != static_cast<int32_t>(en_idx + 1))
                return true;
            return propagateStart(state, b, cid, en_idx, state.domains[changed].getMin());
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
                        if (!propagateStart(state, b, cid, en_idx, state.domains[st_v].getMin()))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};

// 6. if (VarType::START and dom.isFixed()) and corresponding selected (dom.isFixed() &&
// dom.fixedValue() > 0), remove start from domain of all other start domains in the same bucket
// where corresponding selected is not fixed to 0.
class StartUniquePropagator : public Propagator
{
    bool removeStartVal(SearchState &state, uint32_t b, VarId source_st_v, int32_t st_val)
    {
        for (const auto &pair : state.selected_vars[b])
        {
            EClassId other_cid = pair.first;
            VarId other_sel_v = pair.second;
            const Domain &other_sel_dom = state.domains[other_sel_v];
            if (other_sel_dom.isFixed() && other_sel_dom.fixedValue() == 0)
                continue;

            for (VarId other_st_v : state.start_vars[b].at(other_cid))
            {
                if (other_st_v == source_st_v)
                    continue;
                Domain other_st_dom = state.domains[other_st_v];
                if (other_st_dom.contains(st_val))
                {
                    if (other_st_dom.isFixed())
                        return false;
                    other_st_dom.remove(st_val);
                    if (other_st_dom.isEmpty())
                        return false;
                    state.setDomain(other_st_v, other_st_dom);
                }
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "StartUniquePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::START)
                return true;
            const Domain &dom = state.domains[changed];
            if (!dom.isFixed())
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            uint32_t en_idx = state.var_infos[changed].enode_idx;
            VarId sel_v = state.selected_vars[b].at(cid);
            const Domain &sel_dom = state.domains[sel_v];
            if (!sel_dom.isFixed() || sel_dom.fixedValue() != static_cast<int32_t>(en_idx + 1))
                return true;
            return removeStartVal(state, b, changed, dom.fixedValue());
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
                    if (!sel_dom.isFixed() || sel_dom.fixedValue() <= 0)
                        continue;
                    uint32_t en_idx = sel_dom.fixedValue() - 1;
                    VarId st_v = state.start_vars[b].at(cid)[en_idx];
                    const Domain &st_dom = state.domains[st_v];
                    if (st_dom.isFixed())
                    {
                        if (!removeStartVal(state, b, st_v, st_dom.fixedValue()))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};

// 7. WRITE AFTER READ if (VarType::OFFSET and dom.isFixed() and corresponding start is fixed ||
// VarType::START and dom.isFixed() and corresponding offset is fixed) and corresponding selected
// is not fixed to 0, and overlaps with another fixed start+offset+size+selected_not_{0} in the
// same bucket and mem_space.
// Sort by start to get A, B. let R(A) be all readers of a direct or through view(s), use
// max(reader_start+1) as end.
// - if both A and B are views, return true. ignore
// - if B is view of A and (B.offset >= A.offset && B.offset+B.size <= A.offset+A.size), or A is a
// view of B and A is within B, return true. ignore
// - if A is INPUT/CACHE/ROOT, or B is INPUT/CACHE/ROOT, return false.
// - if B is a reader (direct or through view(s)) of A, make sure A is in safe_inplace_idxs. if
// not, return false.
// - if B is a reader, but not (B.offset >= A.offset && B.offset+B.size <= A.offset+A.size) return
// false.
// - for every reader C = R(A)/B, B.setMin(reader.start.getMin + 1), C.setMax(start_B - 1)
class WriteAfterReadPropagator : public Propagator
{
    struct FixedAlloc
    {
        EClassId cid;
        uint32_t bucket_idx;
        MemSpace mem_space;
        uint32_t offset;
        uint32_t size;
        int32_t start;
        uint32_t en_idx;
        ENodeId en_id;
        VarId start_var;
        VarId offset_var;
        bool is_view;
        bool is_input_or_cache;
        bool is_root;
    };

    bool getFixedAlloc(const SearchState &state, uint32_t b, EClassId cid, FixedAlloc &out) const
    {
        auto sel_it = state.selected_vars[b].find(cid);
        if (sel_it == state.selected_vars[b].end())
            return false;
        const Domain &sel_dom = state.domains[sel_it->second];
        if (sel_dom.isFixed() && sel_dom.fixedValue() == 0)
            return false;

        uint32_t en_idx = 0;
        if (sel_dom.isFixed() && sel_dom.fixedValue() > 0)
        {
            en_idx = static_cast<uint32_t>(sel_dom.fixedValue() - 1);
        }
        else
        {
            int32_t single_val = -1;
            for (int32_t v = 1; v <= 31; ++v)
            {
                if (sel_dom.contains(v))
                {
                    if (single_val != -1)
                    {
                        single_val = -1;
                        break;
                    }
                    single_val = v;
                }
            }
            if (single_val <= 0)
                return false;
            en_idx = static_cast<uint32_t>(single_val - 1);
        }

        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        if (en_idx >= cls.enodes.size())
            return false;
        ENodeId en_id = cls.enodes[en_idx];
        const ENode &enode = state.bucket_egraphs[b].getENode(en_id);

        auto off_it = state.offset_vars[b].find(cid);
        if (off_it == state.offset_vars[b].end())
            return false;
        VarId off_v = off_it->second;
        const Domain &off_dom = state.domains[off_v];
        if (!off_dom.isFixed())
            return false;

        auto st_it = state.start_vars[b].find(cid);
        if (st_it == state.start_vars[b].end() || en_idx >= st_it->second.size())
            return false;
        VarId st_v = st_it->second[en_idx];
        const Domain &st_dom = state.domains[st_v];
        if (!st_dom.isFixed())
            return false;

        out.cid = cid;
        out.bucket_idx = b;
        out.mem_space = cls.mem_space;
        out.offset = static_cast<uint32_t>(off_dom.fixedValue());
        uint32_t psize = state.bytesToPages(getSizeBytes(cls.shape, cls.dtype), cls.mem_space);
        out.size = (psize == 0) ? 1 : psize;
        out.start = st_dom.fixedValue();
        out.en_idx = en_idx;
        out.en_id = en_id;
        out.start_var = st_v;
        out.offset_var = off_v;
        out.is_view = en_id.value < state.bucket_enode_infos[b].size() &&
                     state.bucket_enode_infos[b][en_id.value].is_view;
        out.is_input_or_cache = (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE ||
                                (cls.base_eclass_id != BaseEClassId{} && state.preallocated_buffers.count(cls.base_eclass_id)));
        out.is_root = (b < state.bucket_root_ids.size() &&
                      state.bucket_egraphs[b].findConst(cid) == state.bucket_egraphs[b].findConst(state.bucket_root_ids[b]));
        return true;
    }

    bool isViewOf(const SearchState &state, uint32_t b, EClassId base, EClassId view_cand) const
    {
        EClassId curr = view_cand;
        std::unordered_set<EClassId> visited;
        while (visited.insert(curr).second && curr != base)
        {
            auto sel_it = state.selected_vars[b].find(curr);
            if (sel_it == state.selected_vars[b].end())
                break;
            const Domain &dom = state.domains[sel_it->second];
            if (!dom.isFixed() || dom.fixedValue() <= 0)
                break;
            uint32_t en_idx = dom.fixedValue() - 1;
            const EClass &cls = state.bucket_egraphs[b].getEClass(curr);
            if (en_idx >= cls.enodes.size())
                break;
            ENodeId en_id = cls.enodes[en_idx];
            bool is_view = en_id.value < state.bucket_enode_infos[b].size() &&
                           state.bucket_enode_infos[b][en_id.value].is_view;
            if (!is_view)
                break;
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
            if (enode.getChildren().empty())
                break;
            curr = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
        }
        return curr == base;
    }

    std::unordered_set<EClassId> getAllReaders(const SearchState &state, uint32_t b, EClassId base) const
    {
        std::vector<EClassId> all_aliases = {base};
        std::unordered_set<EClassId> visited_aliases = {base};
        for (size_t i = 0; i < all_aliases.size(); ++i)
        {
            EClassId target = all_aliases[i];
            for (const auto &pair : state.selected_vars[b])
            {
                EClassId cand = pair.first;
                if (visited_aliases.count(cand) != 0)
                    continue;
                const Domain &dom = state.domains[pair.second];
                if (!dom.isFixed() || dom.fixedValue() <= 0)
                    continue;
                uint32_t en_idx = dom.fixedValue() - 1;
                const EClass &cls = state.bucket_egraphs[b].getEClass(cand);
                if (en_idx >= cls.enodes.size())
                    continue;
                ENodeId en_id = cls.enodes[en_idx];
                bool is_view = en_id.value < state.bucket_enode_infos[b].size() &&
                               state.bucket_enode_infos[b][en_id.value].is_view;
                if (is_view)
                {
                    const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                    if (!enode.getChildren().empty() &&
                        state.bucket_egraphs[b].findConst(enode.getChildren()[0]) == target)
                    {
                        visited_aliases.insert(cand);
                        all_aliases.push_back(cand);
                    }
                }
            }
        }

        std::unordered_set<EClassId> readers;
        for (const auto &pair : state.selected_vars[b])
        {
            EClassId cand = pair.first;
            const Domain &dom = state.domains[pair.second];
            if (dom.isFixed() && dom.fixedValue() == 0)
                continue;
            const EClass &cls = state.bucket_egraphs[b].getEClass(cand);
            for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
            {
                if (!dom.contains(en_idx + 1))
                    continue;
                ENodeId en_id = cls.enodes[en_idx];
                const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                for (EClassId ch : enode.getChildren())
                {
                    if (visited_aliases.count(state.bucket_egraphs[b].findConst(ch)) != 0)
                    {
                        readers.insert(cand);
                        break;
                    }
                }
            }
        }
        return readers;
    }

    bool checkPair(SearchState &state, FixedAlloc A, FixedAlloc B)
    {
        if (A.start > B.start)
            std::swap(A, B);
        else if (A.start == B.start)
            return false;

        // - if both A and B are views, return true. ignore
        if (A.is_view && B.is_view)
            return true;

        // - if B is view of A and within A, or A is view of B and within B, return true. ignore
        if (isViewOf(state, A.bucket_idx, A.cid, B.cid) &&
            B.offset >= A.offset && B.offset + B.size <= A.offset + A.size)
            return true;
        if (isViewOf(state, A.bucket_idx, B.cid, A.cid) &&
            A.offset >= B.offset && A.offset + A.size <= B.offset + B.size)
            return true;

        // - if A is INPUT/CACHE/ROOT, or B is INPUT/CACHE/ROOT, return false.
        if (A.is_input_or_cache || A.is_root || B.is_input_or_cache || B.is_root)
            return false;

        auto readers_A = getAllReaders(state, A.bucket_idx, A.cid);
        bool b_is_reader = readers_A.count(B.cid) != 0;

        if (b_is_reader)
        {
            // - if B is a reader (direct or through view(s)) of A, make sure A is in safe_inplace_idxs. if not, return false.
            const EClass &b_cls = state.bucket_egraphs[B.bucket_idx].getEClass(B.cid);
            const ENode &b_enode = state.bucket_egraphs[B.bucket_idx].getENode(b_cls.enodes[B.en_idx]);
            KernelId b_kid = b_enode.getKernelId();
            std::vector<uint32_t> safe_inplace;
            if (b_kid.value != 0 && KernelRegistry::get().hasKernel(b_kid))
                safe_inplace = KernelRegistry::get().getKernel(b_kid).safe_inplace_idxs;

            bool found_safe = false;
            for (size_t in_idx = 0; in_idx < b_enode.getChildren().size(); ++in_idx)
            {
                EClassId child_cid = state.bucket_egraphs[B.bucket_idx].findConst(b_enode.getChildren()[in_idx]);
                if (child_cid == A.cid || isViewOf(state, A.bucket_idx, A.cid, child_cid))
                {
                    if (std::find(safe_inplace.begin(), safe_inplace.end(), static_cast<uint32_t>(in_idx)) != safe_inplace.end())
                    {
                        found_safe = true;
                        break;
                    }
                }
            }
            if (!found_safe)
                return false;

            // - if B is a reader, but not (B.offset >= A.offset && B.offset+B.size <= A.offset+A.size) return false.
            if (!(B.offset >= A.offset && B.offset + B.size <= A.offset + A.size))
                return false;
        }

        // - for every reader C = R(A)/B, B.setMin(reader.start.getMin + 1), C.setMax(start_B - 1)
        for (EClassId c_cid : readers_A)
        {
            if (c_cid == B.cid)
                continue;
            auto c_sel_it = state.selected_vars[A.bucket_idx].find(c_cid);
            if (c_sel_it == state.selected_vars[A.bucket_idx].end())
                continue;
            const Domain &c_sel_dom = state.domains[c_sel_it->second];
            if (c_sel_dom.isFixed() && c_sel_dom.fixedValue() == 0)
                continue;

            auto c_st_it = state.start_vars[A.bucket_idx].find(c_cid);
            if (c_st_it == state.start_vars[A.bucket_idx].end())
                continue;

            for (uint32_t c_en = 0; c_en < c_st_it->second.size(); ++c_en)
            {
                if (!c_sel_dom.contains(c_en + 1))
                    continue;
                VarId c_st_v = c_st_it->second[c_en];
                Domain c_st_dom = state.domains[c_st_v];
                Domain b_st_dom = state.domains[B.start_var];

                if (b_st_dom.setMin(c_st_dom.getMin() + 1))
                {
                    if (b_st_dom.isEmpty())
                        return false;
                    state.setDomain(B.start_var, b_st_dom);
                }
                if (c_st_dom.setMax(B.start - 1))
                {
                    if (c_st_dom.isEmpty())
                        return false;
                    state.setDomain(c_st_v, c_st_dom);
                }
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "WriteAfterReadPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            VarType type = state.var_infos[changed].type;
            if (type != VarType::OFFSET && type != VarType::START)
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            FixedAlloc curr;
            if (!getFixedAlloc(state, b, cid, curr))
                return true;

            for (const auto &pair : state.selected_vars[b])
            {
                EClassId other_cid = pair.first;
                if (other_cid == cid)
                    continue;
                FixedAlloc other;
                if (!getFixedAlloc(state, b, other_cid, other))
                    continue;
                if (curr.mem_space != other.mem_space)
                    continue;
                if (std::max(curr.offset, other.offset) < std::min(curr.offset + curr.size, other.offset + other.size))
                {
                    if (!checkPair(state, curr, other))
                        return false;
                }
            }
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                std::vector<FixedAlloc> allocs;
                for (const auto &pair : state.selected_vars[b])
                {
                    FixedAlloc a;
                    if (getFixedAlloc(state, b, pair.first, a))
                        allocs.push_back(a);
                }
                for (size_t i = 0; i < allocs.size(); ++i)
                {
                    for (size_t j = i + 1; j < allocs.size(); ++j)
                    {
                        if (allocs[i].mem_space == allocs[j].mem_space &&
                            std::max(allocs[i].offset, allocs[j].offset) < std::min(allocs[i].offset + allocs[i].size, allocs[j].offset + allocs[j].size))
                        {
                            if (!checkPair(state, allocs[i], allocs[j]))
                                return false;
                        }
                    }
                }
            }
        }
        return true;
    }
};

// 8. if (VarType::OFFSET and dom.isFixed() and corresponding start is fixed and corresponding
// selected is not fixed to 0), any reader views should have offset fixed to equal this offset
// (or plus a bit like with a slice).
class ViewOffsetPropagator : public Propagator
{
    bool propagateViewOffset(SearchState &state, uint32_t b, EClassId cid, int32_t off_val)
    {
        for (const auto &pair : state.selected_vars[b])
        {
            EClassId v_cid = pair.first;
            if (v_cid == cid)
                continue;
            VarId v_sel_v = pair.second;
            Domain v_sel_dom = state.domains[v_sel_v];
            if (v_sel_dom.isFixed() && v_sel_dom.fixedValue() == 0)
                continue;

            const EClass &v_cls = state.bucket_egraphs[b].getEClass(v_cid);
            for (uint32_t en_idx = 0; en_idx < v_cls.enodes.size(); ++en_idx)
            {
                if (!v_sel_dom.contains(en_idx + 1))
                    continue;
                ENodeId en_id = v_cls.enodes[en_idx];
                bool is_view = en_id.value < state.bucket_enode_infos[b].size() &&
                               state.bucket_enode_infos[b][en_id.value].is_view;
                if (!is_view)
                    continue;
                const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                if (!enode.getChildren().empty() &&
                    state.bucket_egraphs[b].findConst(enode.getChildren()[0]) == cid)
                {
                    auto off_it = state.offset_vars[b].find(v_cid);
                    if (off_it != state.offset_vars[b].end())
                    {
                        VarId v_off_v = off_it->second;
                        Domain v_off_dom = state.domains[v_off_v];
                        if (!v_off_dom.contains(off_val))
                        {
                            if (v_sel_dom.isFixed())
                                return false;
                            v_sel_dom.remove(en_idx + 1);
                            if (v_sel_dom.isEmpty())
                                return false;
                            state.setDomain(v_sel_v, v_sel_dom);
                        }
                        else if (v_sel_dom.isFixed() && v_sel_dom.fixedValue() == static_cast<int32_t>(en_idx + 1))
                        {
                            if (!v_off_dom.isFixed() || v_off_dom.fixedValue() != off_val)
                            {
                                state.setDomain(v_off_v, Domain::makeFixed(off_val, false));
                            }
                        }
                    }
                }
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "ViewOffsetPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::OFFSET)
                return true;
            const Domain &dom = state.domains[changed];
            if (!dom.isFixed())
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            VarId sel_v = state.selected_vars[b].at(cid);
            const Domain &sel_dom = state.domains[sel_v];
            if (sel_dom.isFixed() && sel_dom.fixedValue() == 0)
                return true;
            if (sel_dom.isFixed() && sel_dom.fixedValue() > 0)
            {
                uint32_t en_idx = sel_dom.fixedValue() - 1;
                VarId st_v = state.start_vars[b].at(cid)[en_idx];
                if (!state.domains[st_v].isFixed())
                    return true;
            }
            return propagateViewOffset(state, b, cid, dom.fixedValue());
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    EClassId cid = pair.first;
                    const Domain &sel_dom = state.domains[pair.second];
                    if (!sel_dom.isFixed() || sel_dom.fixedValue() <= 0)
                        continue;
                    uint32_t en_idx = sel_dom.fixedValue() - 1;
                    VarId st_v = state.start_vars[b].at(cid)[en_idx];
                    if (!state.domains[st_v].isFixed())
                        continue;
                    auto off_it = state.offset_vars[b].find(cid);
                    if (off_it == state.offset_vars[b].end())
                        continue;
                    const Domain &off_dom = state.domains[off_it->second];
                    if (!off_dom.isFixed())
                        continue;
                    if (!propagateViewOffset(state, b, cid, off_dom.fixedValue()))
                        return false;
                }
            }
        }
        return true;
    }
};

// 9. if VarType::SELECTED && dom.isFixed() && dom.fixedValue() > 0, pearce-kelly cycle detection
// given other fixed nonzero selections.
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

  public:
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
            const Domain &dom = state.domains[changed];
            if (!dom.isFixed() || dom.fixedValue() <= 0)
                return true;
            return checkIncrementalCycle(state, state.var_infos[changed].bucket_idx,
                                         state.var_infos[changed].eclass_id,
                                         static_cast<uint32_t>(dom.fixedValue() - 1));
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

// 10. contrapositive of 2. if VarType::SELECTED && dom.isFixed() && dom.fixedValue() == 0, any
// parent enode must be removed from eclass selection domain
class ParentRemovalPropagator : public Propagator
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

    bool removeParentEnodes(SearchState &state, uint32_t b, EClassId unselected_cid)
    {
        ensureParents(state);
        auto it = bucket_parents[b].find(unselected_cid);
        if (it == bucket_parents[b].end())
            return true;

        for (const auto &p_info : it->second)
        {
            auto p_sel_it = state.selected_vars[b].find(p_info.parent_cid);
            if (p_sel_it == state.selected_vars[b].end())
                continue;
            VarId p_sel_v = p_sel_it->second;
            Domain p_sel_dom = state.domains[p_sel_v];
            if (p_sel_dom.contains(static_cast<int32_t>(p_info.en_idx + 1)))
            {
                p_sel_dom.remove(static_cast<int32_t>(p_info.en_idx + 1));
                if (p_sel_dom.isEmpty())
                    return false;
                state.setDomain(p_sel_v, p_sel_dom);
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "ParentRemovalPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;
            const Domain &dom = state.domains[changed];
            if (dom.isFixed() && dom.fixedValue() == 0)
            {
                return removeParentEnodes(state, state.var_infos[changed].bucket_idx,
                                          state.var_infos[changed].eclass_id);
            }
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    const Domain &dom = state.domains[pair.second];
                    if (dom.isFixed() && dom.fixedValue() == 0)
                    {
                        if (!removeParentEnodes(state, b, pair.first))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};

// 11. if VarType::SELECTED && dom.isFixed() && dom.fixedValue() > 0 && optype CACHE/SCATTER (or
// FUSED with root SCATTER in refFactory), fix corresponding cached var to {1}
class CacheRequirementPropagator : public Propagator
{
    bool requireCacheForEClass(SearchState &state, uint32_t b, EClassId cid, int32_t val)
    {
        if (val <= 0)
            return true;
        uint32_t en_idx = static_cast<uint32_t>(val - 1);
        const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
        if (en_idx >= cls.enodes.size())
            return false;
        const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
        if (isOpRootScatterOrCache(enode))
        {
            auto it = state.cached_vars.find(cls.base_eclass_id);
            if (it != state.cached_vars.end())
            {
                VarId cv = it->second;
                Domain c_dom = state.domains[cv];
                if (!c_dom.contains(1))
                    return false;
                if (!c_dom.isFixed())
                {
                    state.setDomain(cv, Domain::makeFixed(1, c_dom.is_mask));
                }
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "CacheRequirementPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;
            const Domain &dom = state.domains[changed];
            if (dom.isFixed() && dom.fixedValue() > 0)
            {
                return requireCacheForEClass(state, state.var_infos[changed].bucket_idx,
                                             state.var_infos[changed].eclass_id, dom.fixedValue());
            }
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    const Domain &dom = state.domains[pair.second];
                    if (dom.isFixed() && dom.fixedValue() > 0)
                    {
                        if (!requireCacheForEClass(state, b, pair.first, dom.fixedValue()))
                            return false;
                    }
                }
            }
        }
        return true;
    }
};

// ============================================================================
// EXTRA PROPAGATORS (12 - 15)
// ============================================================================

// 12. if VarType::SELECTED, update dynamic programming bottom up critical path cp = cost +
// max(children cp), lower_bound = max(critical_path, lower_bound), if lower_bound > incumbent
// return false
class CriticalPathPropagator : public Propagator
{
    float computeBucketCriticalPath(const SearchState &state, uint32_t b) const
    {
        if (b >= state.bucket_root_ids.size())
            return 0.0f;
        EClassId root = state.bucket_root_ids[b];
        std::unordered_map<EClassId, float> memo;
        std::unordered_set<EClassId> visiting;

        std::function<float(EClassId)> getCp = [&](EClassId cid) -> float {
            cid = state.bucket_egraphs[b].findConst(cid);
            auto it = memo.find(cid);
            if (it != memo.end())
                return it->second;
            if (!visiting.insert(cid).second)
                return 0.0f;

            auto sel_it = state.selected_vars[b].find(cid);
            if (sel_it == state.selected_vars[b].end())
            {
                visiting.erase(cid);
                return memo[cid] = 0.0f;
            }
            const Domain &dom = state.domains[sel_it->second];
            if (dom.isFixed() && dom.fixedValue() == 0)
            {
                visiting.erase(cid);
                return memo[cid] = 0.0f;
            }

            const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
            float cp_val = 0.0f;

            if (dom.isFixed() && dom.fixedValue() > 0)
            {
                uint32_t en_idx = static_cast<uint32_t>(dom.fixedValue() - 1);
                if (en_idx < cls.enodes.size())
                {
                    ENodeId en_id = cls.enodes[en_idx];
                    const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                    float cost = (b < state.bucket_enode_infos.size() && en_id.value < state.bucket_enode_infos[b].size())
                                     ? state.bucket_enode_infos[b][en_id.value].cost
                                     : 0.0f;
                    if (cost == TGConstants::INF || std::isnan(cost))
                        cost = 0.0f;
                    bool is_view = (b < state.bucket_enode_infos.size() && en_id.value < state.bucket_enode_infos[b].size()) &&
                                   state.bucket_enode_infos[b][en_id.value].is_view;
                    if (is_view || enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
                        cost = 0.0f;

                    float max_child = 0.0f;
                    for (EClassId ch : enode.getChildren())
                    {
                        max_child = std::max(max_child, getCp(ch));
                    }
                    cp_val = cost + max_child;
                }
            }
            else if (dom.contains(0))
            {
                cp_val = 0.0f;
            }
            else
            {
                float min_enode_cp = TGConstants::INF;
                for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
                {
                    if (!dom.contains(en_idx + 1))
                        continue;
                    ENodeId en_id = cls.enodes[en_idx];
                    const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                    float cost = (b < state.bucket_enode_infos.size() && en_id.value < state.bucket_enode_infos[b].size())
                                     ? state.bucket_enode_infos[b][en_id.value].cost
                                     : 0.0f;
                    if (cost == TGConstants::INF || std::isnan(cost))
                        cost = 0.0f;
                    bool is_view = (b < state.bucket_enode_infos.size() && en_id.value < state.bucket_enode_infos[b].size()) &&
                                   state.bucket_enode_infos[b][en_id.value].is_view;
                    if (is_view || enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
                        cost = 0.0f;

                    float max_child = 0.0f;
                    for (EClassId ch : enode.getChildren())
                    {
                        max_child = std::max(max_child, getCp(ch));
                    }
                    min_enode_cp = std::min(min_enode_cp, cost + max_child);
                }
                cp_val = (min_enode_cp < TGConstants::INF) ? min_enode_cp : 0.0f;
            }

            visiting.erase(cid);
            return memo[cid] = cp_val;
        };

        return getCp(root);
    }

  public:
    std::string name() const override
    {
        return "CriticalPathPropagator";
    }

    float computeLowerBound(const SearchState &state) override
    {
        float total_lb = 0.0f;
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            float w = (b < state.bucket_weights.size()) ? state.bucket_weights[b] : 1.0f;
            if (w > 0.0f)
            {
                total_lb += w * computeBucketCriticalPath(state, b);
            }
        }
        return total_lb;
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (state.best_cost == TGConstants::INF)
            return true;
        if (changed != kInvalidVarId && state.var_infos[changed].type != VarType::SELECTED)
            return true;
        float lb = computeLowerBound(state);
        return lb < state.best_cost;
    }
};

// 13. if VarType::CACHED && dom.isFixed() && dom.fixedValue() == 1, state.cache_sums[mem_space] +=
// size if state.cache_sum > mem_cap
class CacheBudgetPropagator : public Propagator
{
  public:
    std::string name() const override
    {
        return "CacheBudgetPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId && state.var_infos[changed].type != VarType::CACHED)
            return true;

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

// 14. restrictOffset / restrictStart: generalizes overlap constraints to restrict domain of
// multiple allocations across lifetimes and memory spans
class RestrictOffsetPropagator : public Propagator
{
    struct AllocDesc
    {
        EClassId cid;
        uint32_t bucket_idx;
        MemSpace mem_space;
        VarId sel_var;
        VarId offset_var;
        std::vector<VarId> start_vars;
        uint32_t size; // in pages
        std::vector<bool> enode_is_view;
    };

    bool initialized_ = false;
    size_t num_vars_ = 0;
    std::vector<std::unordered_map<MemSpace, std::vector<AllocDesc>>> bucket_allocs_;
    std::vector<const AllocDesc *> var_to_alloc_;

    void init(const SearchState &state)
    {
        num_vars_ = state.var_infos.size();
        var_to_alloc_.assign(num_vars_, nullptr);
        bucket_allocs_.clear();
        bucket_allocs_.resize(state.buckets.size());

        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            if (b >= state.bucket_egraphs.size())
                continue;
            const auto &egraph = state.bucket_egraphs[b];

            for (const auto &pair : state.selected_vars[b])
            {
                EClassId cid = pair.first;
                VarId sel_v = pair.second;

                auto off_it = state.offset_vars[b].find(cid);
                if (off_it == state.offset_vars[b].end())
                    continue;
                VarId off_v = off_it->second;

                auto st_it = state.start_vars[b].find(cid);
                std::vector<VarId> start_vars;
                if (st_it != state.start_vars[b].end())
                    start_vars = st_it->second;

                const EClass &cls = egraph.getEClass(cid);
                uint32_t psize = state.bytesToPages(getSizeBytes(cls.shape, cls.dtype), cls.mem_space);
                uint32_t alloc_size = (psize == 0) ? 1 : psize;

                std::vector<bool> enode_is_view;
                for (ENodeId en_id : cls.enodes)
                {
                    bool is_v = (en_id.value < state.bucket_enode_infos[b].size()) &&
                                state.bucket_enode_infos[b][en_id.value].is_view;
                    enode_is_view.push_back(is_v);
                }

                AllocDesc desc;
                desc.cid = cid;
                desc.bucket_idx = b;
                desc.mem_space = cls.mem_space;
                desc.sel_var = sel_v;
                desc.offset_var = off_v;
                desc.start_vars = std::move(start_vars);
                desc.size = alloc_size;
                desc.enode_is_view = std::move(enode_is_view);

                bucket_allocs_[b][cls.mem_space].push_back(std::move(desc));
            }
        }

        for (uint32_t b = 0; b < bucket_allocs_.size(); ++b)
        {
            for (auto &pair : bucket_allocs_[b])
            {
                for (auto &desc : pair.second)
                {
                    if (desc.offset_var < var_to_alloc_.size())
                        var_to_alloc_[desc.offset_var] = &desc;
                    for (VarId st_v : desc.start_vars)
                    {
                        if (st_v < var_to_alloc_.size())
                            var_to_alloc_[st_v] = &desc;
                    }
                }
            }
        }
        initialized_ = true;
    }

  public:
    std::string name() const override
    {
        return "RestrictOffsetPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (!initialized_ || num_vars_ != state.var_infos.size())
            init(state);

        if (changed != kInvalidVarId)
        {
            if (changed >= var_to_alloc_.size())
                return true;
            const AllocDesc *target = var_to_alloc_[changed];
            if (!target)
                return true;

            const Domain &sel_dom = state.domains[target->sel_var];
            if (sel_dom.isFixed() && sel_dom.fixedValue() == 0)
                return true;

            uint32_t en_idx = (sel_dom.isFixed() && sel_dom.fixedValue() > 0) ? (sel_dom.fixedValue() - 1) : 0;
            if (en_idx < target->enode_is_view.size() && target->enode_is_view[en_idx])
                return true;

            const Domain &target_off_dom = state.domains[target->offset_var];
            bool target_offset_fixed = target_off_dom.isFixed();
            VarType type = state.var_infos[changed].type;

            // Changing START only constrains other memory intervals when offset is already fixed
            if (type == VarType::START && !target_offset_fixed)
                return true;

            uint32_t target_offset = target_offset_fixed ? static_cast<uint32_t>(target_off_dom.fixedValue()) : 0;

            bool target_start_fixed = false;
            int32_t target_start = 0;
            int32_t target_end = 0;
            if (en_idx < target->start_vars.size())
            {
                const Domain &st_dom = state.domains[target->start_vars[en_idx]];
                target_start_fixed = st_dom.isFixed();
                target_start = st_dom.getMin();
                target_end = (target_start_fixed ? st_dom.fixedValue() : st_dom.getMax()) + 1;
            }

            auto space_it = bucket_allocs_[target->bucket_idx].find(target->mem_space);
            if (space_it == bucket_allocs_[target->bucket_idx].end())
                return true;

            for (const auto &other : space_it->second)
            {
                if (other.cid == target->cid)
                    continue;

                const Domain &other_sel = state.domains[other.sel_var];
                if (other_sel.isFixed() && other_sel.fixedValue() == 0)
                    continue;

                const Domain &other_off_dom = state.domains[other.offset_var];
                bool other_offset_fixed = other_off_dom.isFixed();
                if (!target_offset_fixed && !other_offset_fixed)
                    continue;

                uint32_t other_en_idx = (other_sel.isFixed() && other_sel.fixedValue() > 0) ? (other_sel.fixedValue() - 1) : 0;
                if (other_en_idx < other.enode_is_view.size() && other.enode_is_view[other_en_idx])
                    continue;

                bool other_start_fixed = false;
                int32_t other_start = 0;
                int32_t other_end = 0;
                if (other_en_idx < other.start_vars.size())
                {
                    const Domain &st_dom = state.domains[other.start_vars[other_en_idx]];
                    other_start_fixed = st_dom.isFixed();
                    other_start = st_dom.getMin();
                    other_end = (other_start_fixed ? st_dom.fixedValue() : st_dom.getMax()) + 1;
                }

                uint32_t other_offset = other_offset_fixed ? static_cast<uint32_t>(other_off_dom.fixedValue()) : 0;

                // Restrict offsets if lifetimes can overlap
                bool lifetimes_can_overlap = !(target_end <= other_start || other_end <= target_start);
                if (lifetimes_can_overlap)
                {
                    if (target_offset_fixed && !other_offset_fixed)
                    {
                        Domain off_dom = state.domains[other.offset_var];
                        int64_t before_max = static_cast<int64_t>(target_offset) - other.size;
                        int64_t after_min = static_cast<int64_t>(target_offset) + target->size;
                        bool modified = false;
                        if (off_dom.getMin() > before_max)
                            modified = off_dom.setMin(static_cast<int32_t>(after_min)) || modified;
                        else if (off_dom.getMax() < after_min)
                            modified = off_dom.setMax(static_cast<int32_t>(before_max)) || modified;
                        if (off_dom.isEmpty())
                            return false;
                        if (modified)
                            state.setDomain(other.offset_var, off_dom);
                    }
                    else if (!target_offset_fixed && other_offset_fixed)
                    {
                        Domain off_dom = state.domains[target->offset_var];
                        int64_t before_max = static_cast<int64_t>(other_offset) - target->size;
                        int64_t after_min = static_cast<int64_t>(other_offset) + other.size;
                        bool modified = false;
                        if (off_dom.getMin() > before_max)
                            modified = off_dom.setMin(static_cast<int32_t>(after_min)) || modified;
                        else if (off_dom.getMax() < after_min)
                            modified = off_dom.setMax(static_cast<int32_t>(before_max)) || modified;
                        if (off_dom.isEmpty())
                            return false;
                        if (modified)
                        {
                            state.setDomain(target->offset_var, off_dom);
                            target_offset_fixed = off_dom.isFixed();
                            if (off_dom.isFixed())
                                target_offset = static_cast<uint32_t>(off_dom.fixedValue());
                        }
                    }
                }

                // Restrict start times if offsets overlap
                if (target_offset_fixed && other_offset_fixed)
                {
                    bool memory_overlaps = !(target_offset + target->size <= other_offset ||
                                            other_offset + other.size <= target_offset);
                    if (memory_overlaps)
                    {
                        if (target_start_fixed && other_start_fixed)
                        {
                            if (!(target_end <= other_start || other_end <= target_start))
                                return false;
                        }
                        else if (target_start_fixed && !other_start_fixed)
                        {
                            for (VarId j_st_v : other.start_vars)
                            {
                                Domain j_st_dom = state.domains[j_st_v];
                                if (j_st_dom.getMin() >= target_start)
                                {
                                    if (j_st_dom.setMin(target_end))
                                    {
                                        if (j_st_dom.isEmpty())
                                            return false;
                                        state.setDomain(j_st_v, j_st_dom);
                                    }
                                }
                            }
                        }
                        else if (!target_start_fixed && other_start_fixed)
                        {
                            for (VarId i_st_v : target->start_vars)
                            {
                                Domain i_st_dom = state.domains[i_st_v];
                                if (i_st_dom.getMin() >= other_start)
                                {
                                    if (i_st_dom.setMin(other_end))
                                    {
                                        if (i_st_dom.isEmpty())
                                            return false;
                                        state.setDomain(i_st_v, i_st_dom);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return true;
        }
        else
        {
            for (uint32_t b = 0; b < bucket_allocs_.size(); ++b)
            {
                for (const auto &space_pair : bucket_allocs_[b])
                {
                    const auto &allocs = space_pair.second;
                    std::vector<size_t> fixed_indices;
                    std::vector<size_t> active_indices;

                    for (size_t i = 0; i < allocs.size(); ++i)
                    {
                        const auto &a = allocs[i];
                        const Domain &sel_dom = state.domains[a.sel_var];
                        if (sel_dom.isFixed() && sel_dom.fixedValue() == 0)
                            continue;
                        uint32_t en_idx = (sel_dom.isFixed() && sel_dom.fixedValue() > 0) ? (sel_dom.fixedValue() - 1) : 0;
                        if (en_idx < a.enode_is_view.size() && a.enode_is_view[en_idx])
                            continue;

                        active_indices.push_back(i);
                        if (state.domains[a.offset_var].isFixed())
                            fixed_indices.push_back(i);
                    }

                    if (fixed_indices.empty())
                        continue;

                    for (size_t fi : fixed_indices)
                    {
                        const auto &fa = allocs[fi];
                        const Domain &fa_sel = state.domains[fa.sel_var];
                        uint32_t fa_en_idx = (fa_sel.isFixed() && fa_sel.fixedValue() > 0) ? (fa_sel.fixedValue() - 1) : 0;
                        int32_t fa_start = 0, fa_end = 0;
                        if (fa_en_idx < fa.start_vars.size())
                        {
                            const Domain &st_dom = state.domains[fa.start_vars[fa_en_idx]];
                            fa_start = st_dom.getMin();
                            fa_end = (st_dom.isFixed() ? st_dom.fixedValue() : st_dom.getMax()) + 1;
                        }
                        uint32_t fa_off = static_cast<uint32_t>(state.domains[fa.offset_var].fixedValue());

                        for (size_t oi : active_indices)
                        {
                            if (fi == oi)
                                continue;
                            const auto &oa = allocs[oi];
                            const Domain &oa_off_dom = state.domains[oa.offset_var];
                            if (oa_off_dom.isFixed())
                                continue;

                            const Domain &oa_sel = state.domains[oa.sel_var];
                            uint32_t oa_en_idx = (oa_sel.isFixed() && oa_sel.fixedValue() > 0) ? (oa_sel.fixedValue() - 1) : 0;
                            int32_t oa_start = 0, oa_end = 0;
                            if (oa_en_idx < oa.start_vars.size())
                            {
                                const Domain &st_dom = state.domains[oa.start_vars[oa_en_idx]];
                                oa_start = st_dom.getMin();
                                oa_end = (st_dom.isFixed() ? st_dom.fixedValue() : st_dom.getMax()) + 1;
                            }

                            bool lifetimes_can_overlap = !(fa_end <= oa_start || oa_end <= fa_start);
                            if (lifetimes_can_overlap)
                            {
                                Domain off_dom = state.domains[oa.offset_var];
                                int64_t before_max = static_cast<int64_t>(fa_off) - oa.size;
                                int64_t after_min = static_cast<int64_t>(fa_off) + fa.size;
                                bool modified = false;
                                if (off_dom.getMin() > before_max)
                                    modified = off_dom.setMin(static_cast<int32_t>(after_min)) || modified;
                                else if (off_dom.getMax() < after_min)
                                    modified = off_dom.setMax(static_cast<int32_t>(before_max)) || modified;
                                if (off_dom.isEmpty())
                                    return false;
                                if (modified)
                                    state.setDomain(oa.offset_var, off_dom);
                            }
                        }
                    }
                }
            }
        }
        return true;
    }
};

// 15. if VarType::SELECTED, update per engine workload, lower_bound = max(lower_bound,
// engine_workload) for engine_workload in workloads
class EngineWorkloadPropagator : public Propagator
{
  public:
    std::string name() const override
    {
        return "EngineWorkloadPropagator";
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
            for (const auto &pair : state.selected_vars[b])
            {
                EClassId cid = pair.first;
                VarId sel_v = pair.second;
                const Domain &sel_dom = state.domains[sel_v];
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);

                if (sel_dom.isFixed() && sel_dom.fixedValue() > 0)
                {
                    uint32_t en_idx = static_cast<uint32_t>(sel_dom.fixedValue() - 1);
                    if (en_idx < cls.enodes.size())
                    {
                        ENodeId en_id = cls.enodes[en_idx];
                        const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                        float c = (en_id.value < state.bucket_enode_infos[b].size())
                                      ? state.bucket_enode_infos[b][en_id.value].cost
                                      : 0.0f;
                        if (c > 0.0f && c < TGConstants::INF &&
                            enode.getOpType() != OpType::INPUT &&
                            enode.getOpType() != OpType::CACHE)
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

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (state.best_cost == TGConstants::INF)
            return true;
        if (changed != kInvalidVarId && state.var_infos[changed].type != VarType::SELECTED)
            return true;
        float lb = computeLowerBound(state);
        return lb < state.best_cost;
    }
};

// Aliases for compatibility
class SelectionPropagator : public Propagator
{
    SelectionReachabilityPropagator reachability;
    SelectionChildrenPropagator children;

  public:
    std::string name() const override
    {
        return "SelectionPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        return reachability.propagate(state, changed, worklist) &&
               children.propagate(state, changed, worklist);
    }
};

// ============================================================================
// Registration Helpers
// ============================================================================

template <typename EngineT>
inline void addBasePropagators(EngineT &engine)
{
    engine.addPropagator(std::make_unique<SelectionReachabilityPropagator>());
    engine.addPropagator(std::make_unique<SelectionChildrenPropagator>());
    engine.addPropagator(std::make_unique<UnselectedStartOffsetPropagator>());
    engine.addPropagator(std::make_unique<CacheExclusionPropagator>());
    engine.addPropagator(std::make_unique<StartPrecedencePropagator>());
    engine.addPropagator(std::make_unique<StartUniquePropagator>());
    engine.addPropagator(std::make_unique<WriteAfterReadPropagator>());
    engine.addPropagator(std::make_unique<ViewOffsetPropagator>());
    engine.addPropagator(std::make_unique<PearceKellyCyclePropagator>());
    engine.addPropagator(std::make_unique<ParentRemovalPropagator>());
    engine.addPropagator(std::make_unique<CacheRequirementPropagator>());
}

template <typename EngineT>
inline void addExtraPropagators(EngineT &engine)
{
    engine.addPropagator(std::make_unique<CriticalPathPropagator>());
    engine.addPropagator(std::make_unique<CacheBudgetPropagator>());
    engine.addPropagator(std::make_unique<RestrictOffsetPropagator>());
    engine.addPropagator(std::make_unique<EngineWorkloadPropagator>());
}

template <typename EngineT>
inline void addAllPropagators(EngineT &engine)
{
    addBasePropagators(engine);
    addExtraPropagators(engine);
}

} // namespace plan
