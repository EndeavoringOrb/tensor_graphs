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
#include "core/logging.hpp"
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
        int32_t required_consumer_start = min_start + 1;
        auto &frontier = precedence.frontier;
        frontier.clear();
        frontier.push_back(cid);
        precedence.visited_stamp[cid.value] = stamp;

        for (size_t head = 0; head < frontier.size(); ++head)
        {
            EClassId current = frontier[head];
            auto it = precedence.parents.find(current);
            if (it == precedence.parents.end())
                continue;

            for (const auto &p_info : it->second)
            {
                EClassId p_cid = p_info.parent_cid;
                uint32_t p_en_idx = p_info.en_idx;
                if (p_info.selection_var == kInvalidVarId || p_info.start_var == kInvalidVarId)
                    continue;
                VarId p_sel_v = p_info.selection_var;
                const Domain &p_sel_dom = state.domains[p_sel_v];
                if (!p_sel_dom.contains(static_cast<int32_t>(p_en_idx + 1)))
                    continue;

                VarId p_st_v = p_info.start_var;
                const Domain &current_start_domain = state.domains[p_st_v];
                if (current_start_domain.isEmpty() || current_start_domain.getMin() < required_consumer_start)
                {
                    Domain p_st_dom = current_start_domain;
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
                }
                if (p_info.is_view && precedence.visited_stamp[p_cid.value] != stamp)
                {
                    precedence.visited_stamp[p_cid.value] = stamp;
                    frontier.push_back(p_cid);
                }
            }
        }
        return true;
    }

  public:
    explicit StartPrecedencePropagator(bool fixed_starts_only = false)
        : fixed_starts_only(fixed_starts_only)
    {
    }

    std::string name() const override
    {
        return "StartPrecedencePropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        state.ensurePropagationState();
        if (changed != kInvalidVarId)
        {
            if (state.var_infos[changed].type != VarType::START)
                return true;
            if (fixed_starts_only && !state.domains[changed].isFixed())
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            uint32_t en_idx = state.var_infos[changed].enode_idx;
            VarId sel_v = state.selected_vars[b].at(cid);
            const Domain &sel_dom = state.domains[sel_v];
            if (!sel_dom.isFixed() || sel_dom.fixedValue() != static_cast<int32_t>(en_idx + 1))
                return true;
            return propagateStart(state, b, cid, state.domains[changed].getMin());
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
  public:
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

    static inline const std::vector<EClassId> empty_readers{};

    static bool getAlloc(const SearchState &state, uint32_t b, EClassId cid, FixedAlloc &out,
                         bool require_fixed_offset = true, bool require_fixed_start = true)
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
        if (off_dom.isEmpty())
            return false;
        if (require_fixed_offset && !off_dom.isFixed())
            return false;

        auto st_it = state.start_vars[b].find(cid);
        if (st_it == state.start_vars[b].end() || en_idx >= st_it->second.size())
            return false;
        VarId st_v = st_it->second[en_idx];
        const Domain &st_dom = state.domains[st_v];
        if (st_dom.isEmpty())
            return false;
        if (require_fixed_start && !st_dom.isFixed())
            return false;

        out.cid = cid;
        out.bucket_idx = b;
        out.mem_space = cls.mem_space;
        out.offset = static_cast<uint32_t>(off_dom.getMin());
        uint32_t psize = state.bytesToPages(getSizeBytes(cls.shape, cls.dtype), cls.mem_space);
        out.size = (psize == 0) ? 1 : psize;
        out.start = st_dom.isFixed() ? st_dom.fixedValue() : st_dom.getMin();
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

    static bool getFixedAlloc(const SearchState &state, uint32_t b, EClassId cid, FixedAlloc &out)
    {
        return getAlloc(state, b, cid, out, true, true);
    }

    static void buildBucketInfo(const SearchState &state, uint32_t b)
    {
        auto &schedule = state.propagation.write_after_read[b];
        if (!schedule.dirty)
            return;
        auto &class_info = schedule.class_info;
        auto &active_cids = schedule.active_cids;
        auto &touched_cids = schedule.touched_cids;
        const bool rebuild_structure = schedule.structure_dirty;
        std::vector<EClassId> affected_cids;
        const size_t num_classes = state.bucket_egraphs[b].classes.size();
        if (class_info.size() < num_classes)
            class_info.resize(num_classes);
        schedule.temporal_overlaps.resize(num_classes);
        if (rebuild_structure)
        {
            for (EClassId cid : active_cids)
            {
                if (cid.value < schedule.temporal_overlaps.size())
                    schedule.temporal_overlaps[cid.value].clear();
            }
        }

        if (rebuild_structure)
        {
            for (EClassId cid : touched_cids)
            {
                if (cid.value < class_info.size())
                {
                    auto &info = class_info[cid.value];
                    info.is_active = false;
                    info.is_view = false;
                    info.is_input_or_cache = false;
                    info.is_root = false;
                    info.view_parent = EClassId{UINT32_MAX};
                    info.base_cid = EClassId{UINT32_MAX};
                    info.start_max = -1;
                    info.max_reader_start_max = -1;
                    info.readers.clear();
                    info.reader_dependents.clear();
                }
            }
            touched_cids.clear();
            active_cids.clear();

            for (const auto &pair : state.selected_vars[b])
            {
            EClassId cid = pair.first;
            if (cid.value >= class_info.size())
                continue;

            const Domain &sel_dom = state.domains[pair.second];
            if (sel_dom.isFixed() && sel_dom.fixedValue() == 0)
                continue;

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
                    continue;
                en_idx = static_cast<uint32_t>(single_val - 1);
            }

            const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
            if (en_idx >= cls.enodes.size())
                continue;
            ENodeId en_id = cls.enodes[en_idx];
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);

            auto &info = class_info[cid.value];
            info.is_active = true;
            info.en_idx = en_idx;
            info.en_id = en_id;
            info.is_view = en_id.value < state.bucket_enode_infos[b].size() &&
                           state.bucket_enode_infos[b][en_id.value].is_view;
            info.is_input_or_cache = (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE ||
                                     (cls.base_eclass_id != BaseEClassId{} && state.preallocated_buffers.count(cls.base_eclass_id)));
            info.is_root = (b < state.bucket_root_ids.size() &&
                           state.bucket_egraphs[b].findConst(cid) == state.bucket_egraphs[b].findConst(state.bucket_root_ids[b]));

            if (info.is_view && !enode.getChildren().empty())
            {
                info.view_parent = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
            }
            else
            {
                info.view_parent = EClassId{UINT32_MAX};
            }

            info.start_max = -1;
            auto st_it = state.start_vars[b].find(cid);
            if (st_it != state.start_vars[b].end())
            {
                for (uint32_t c_en = 0; c_en < st_it->second.size(); ++c_en)
                {
                    if (sel_dom.contains(c_en + 1))
                    {
                        int32_t mx = state.domains[st_it->second[c_en]].getMax();
                        if (mx > info.start_max)
                            info.start_max = mx;
                    }
                }
            }

            info.max_reader_start_max = -1;
            info.readers.clear();
            active_cids.push_back(cid);
            touched_cids.push_back(cid);
            }

            // Resolve base_cid for each active eclass
            for (EClassId cid : active_cids)
            {
            EClassId curr = cid;
            int step = 0;
            for (; step < 32; ++step)
            {
                if (curr.value >= class_info.size() || !class_info[curr.value].is_active || !class_info[curr.value].is_view)
                    break;
                EClassId p = class_info[curr.value].view_parent;
                if (p.value == UINT32_MAX)
                    break;
                curr = p;
            }
            if (step >= 32)
            {
                Error::throw_err("WriteAfterReadPropagator::buildBucketInfo: view chain for base_cid exceeded 32 steps");
            }
            class_info[cid.value].base_cid = curr;
            }

            // Connect readers by examining children of each candidate
            for (const auto &pair : state.selected_vars[b])
            {
            EClassId cand = pair.first;
            const Domain &dom = state.domains[pair.second];
            if (dom.isFixed() && dom.fixedValue() == 0)
                continue;

            int32_t cand_st_max = -1;
            auto st_it = state.start_vars[b].find(cand);
            if (st_it != state.start_vars[b].end())
            {
                for (uint32_t c_en = 0; c_en < st_it->second.size(); ++c_en)
                {
                    if (dom.contains(c_en + 1))
                    {
                        int32_t mx = state.domains[st_it->second[c_en]].getMax();
                        if (mx > cand_st_max)
                            cand_st_max = mx;
                    }
                }
            }

            if (cand.value < class_info.size())
            {
                if (!class_info[cand.value].is_active && class_info[cand.value].start_max == -1)
                {
                    touched_cids.push_back(cand);
                }
                class_info[cand.value].start_max = cand_st_max;
            }

            const EClass &cls = state.bucket_egraphs[b].getEClass(cand);
            for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
            {
                if (!dom.contains(en_idx + 1))
                    continue;
                ENodeId en_id = cls.enodes[en_idx];
                const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
                for (EClassId ch : enode.getChildren())
                {
                    EClassId curr = state.bucket_egraphs[b].findConst(ch);
                    int step = 0;
                    for (; step < 32; ++step)
                    {
                        if (curr.value >= class_info.size() || !class_info[curr.value].is_active)
                            break;
                        auto &target_info = class_info[curr.value];
                        target_info.readers.push_back(cand);
                        if (!target_info.is_view || target_info.view_parent.value == UINT32_MAX)
                            break;
                        curr = target_info.view_parent;
                    }
                    if (step >= 32)
                    {
                        Error::throw_err("WriteAfterReadPropagator::buildBucketInfo: view chain for readers exceeded 32 steps");
                    }
                }
            }
            }

            // Deduplicate readers for each active eclass
            for (EClassId cid : active_cids)
            {
            auto &r = class_info[cid.value].readers;
            if (r.size() > 1)
            {
                std::sort(r.begin(), r.end(), [](EClassId x, EClassId y) { return x.value < y.value; });
                r.erase(std::unique(r.begin(), r.end()), r.end());
            }
            }
            for (EClassId cid : active_cids)
            {
                for (EClassId reader : class_info[cid.value].readers)
                {
                    if (reader.value < class_info.size())
                        class_info[reader.value].reader_dependents.push_back(cid);
                }
            }
            schedule.structure_dirty = false;
        }
        else
        {
            // A start-domain change only affects its eclass start maximum,
            // the latest-reader bounds of values it reads, and overlap rows
            // for those affected classes.
            schedule.affected_stamp.resize(num_classes, 0);
            ++schedule.affected_epoch;
            if (schedule.affected_epoch == 0)
            {
                std::fill(schedule.affected_stamp.begin(), schedule.affected_stamp.end(), 0);
                schedule.affected_epoch = 1;
            }
            auto markAffected = [&](EClassId cid) {
                if (cid.value >= num_classes || schedule.affected_stamp[cid.value] == schedule.affected_epoch)
                    return;
                schedule.affected_stamp[cid.value] = schedule.affected_epoch;
                affected_cids.push_back(cid);
            };
            for (EClassId cid : schedule.dirty_start_cids)
            {
                auto sel_it = state.selected_vars[b].find(cid);
                if (sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &selection = state.domains[sel_it->second];
                int32_t start_max = -1;
                auto st_it = state.start_vars[b].find(cid);
                if (st_it != state.start_vars[b].end())
                {
                    for (uint32_t en_idx = 0; en_idx < st_it->second.size(); ++en_idx)
                    {
                        if (selection.contains(en_idx + 1))
                            start_max = std::max(start_max, state.domains[st_it->second[en_idx]].getMax());
                    }
                }
                if (cid.value < class_info.size())
                    class_info[cid.value].start_max = start_max;

                markAffected(cid);
                for (EClassId dependent : class_info[cid.value].reader_dependents)
                    markAffected(dependent);
            }

            for (EClassId cid : affected_cids)
            {
                if (cid.value >= class_info.size() || !class_info[cid.value].is_active)
                    continue;
                auto &info = class_info[cid.value];
                info.max_reader_start_max = -1;
                for (EClassId reader : info.readers)
                {
                    if (reader.value < class_info.size())
                        info.max_reader_start_max = std::max(info.max_reader_start_max,
                                                             class_info[reader.value].start_max);
                }
            }
        }

        // A structural rebuild starts with every temporal row invalid. A
        // start-only update clears just the rows whose interval changed.
        if (!rebuild_structure)
        {
            for (EClassId cid : affected_cids)
            {
                if (cid.value >= num_classes || !class_info[cid.value].is_active)
                    continue;
                auto &row = schedule.temporal_overlaps[cid.value];
                for (EClassId neighbor : row)
                {
                    if (neighbor.value >= schedule.temporal_overlaps.size())
                        continue;
                    auto &neighbor_row = schedule.temporal_overlaps[neighbor.value];
                    neighbor_row.erase(std::remove(neighbor_row.begin(), neighbor_row.end(), cid), neighbor_row.end());
                }
                row.clear();
            }
        }
        else
        {
            for (EClassId cid : active_cids)
            {
                auto &info = class_info[cid.value];
                info.max_reader_start_max = -1;
                for (EClassId reader : info.readers)
                {
                    if (reader.value < class_info.size())
                        info.max_reader_start_max = std::max(info.max_reader_start_max,
                                                             class_info[reader.value].start_max);
                }
            }
        }

        auto addTemporalPair = [&](EClassId a, EClassId c) {
            schedule.temporal_overlaps[a.value].push_back(c);
            schedule.temporal_overlaps[c.value].push_back(a);
        };

        auto temporalPairOverlaps = [&](EClassId a_cid, EClassId c_cid) {
            return mayOverlapInTime(state, b, a_cid, c_cid);
        };

        if (rebuild_structure)
        {
            // Fixed, non-persistent lifetimes form intervals in dispatch
            // order. Sweep them instead of testing every active pair.
            struct TemporalCandidate
            {
                EClassId cid;
                int32_t start;
                int32_t last_reader_start;
                bool is_broad;
            };
            std::vector<TemporalCandidate> temporal_candidates;
            temporal_candidates.reserve(active_cids.size());
            std::vector<size_t> broad_candidates;
            std::vector<size_t> fixed_candidates;
            for (EClassId cid : active_cids)
            {
                const auto &info = class_info[cid.value];
                const auto st_it = state.start_vars[b].find(cid);
                const bool start_fixed = st_it != state.start_vars[b].end() &&
                                         info.en_idx < st_it->second.size() &&
                                         state.domains[st_it->second[info.en_idx]].isFixed();
                const bool broad = !start_fixed || info.is_input_or_cache || info.is_root;
                int32_t start = 0;
                if (start_fixed)
                    start = state.domains[st_it->second[info.en_idx]].fixedValue();
                temporal_candidates.push_back({cid, start, info.max_reader_start_max, broad});
                const size_t idx = temporal_candidates.size() - 1;
                (broad ? broad_candidates : fixed_candidates).push_back(idx);
            }

            // Pairs involving a broad candidate are retained unconditionally,
            // matching the previous conservative behavior.
            for (size_t broad_idx : broad_candidates)
            {
                const EClassId broad_cid = temporal_candidates[broad_idx].cid;
                for (const auto &candidate : temporal_candidates)
                {
                    if (broad_cid != candidate.cid &&
                        (!candidate.is_broad || broad_cid.value < candidate.cid.value))
                        addTemporalPair(broad_cid, candidate.cid);
                }
            }

            std::sort(fixed_candidates.begin(), fixed_candidates.end(), [&](size_t lhs, size_t rhs) {
                const auto &a = temporal_candidates[lhs];
                const auto &c = temporal_candidates[rhs];
                return (a.start != c.start) ? a.start < c.start : a.cid.value < c.cid.value;
            });
            for (size_t i = 0; i < fixed_candidates.size(); ++i)
            {
                const auto &earlier = temporal_candidates[fixed_candidates[i]];
                for (size_t j = i + 1; j < fixed_candidates.size(); ++j)
                {
                    const auto &later = temporal_candidates[fixed_candidates[j]];
                    // Equal starts were conservatively considered overlapping.
                    if (later.start != earlier.start && later.start > earlier.last_reader_start)
                        break;
                    addTemporalPair(earlier.cid, later.cid);
                }
            }
        }
        else
        {
            // Rebuild only overlap rows touched by a changed start bound.
            for (EClassId cid : affected_cids)
            {
                if (cid.value >= class_info.size() || !class_info[cid.value].is_active)
                    continue;
                for (EClassId other : active_cids)
                {
                    if (other == cid)
                        continue;
                    if (schedule.affected_stamp[other.value] == schedule.affected_epoch &&
                        cid.value > other.value)
                        continue;
                    if (temporalPairOverlaps(cid, other))
                        addTemporalPair(cid, other);
                }
            }
        }
        schedule.dirty_start_cids.clear();
        schedule.dirty = false;
    }

    static void buildFixedOffsetIndex(const SearchState &state, uint32_t b)
    {
        auto &schedule = state.propagation.write_after_read[b];
        if (!schedule.spatial_dirty)
            return;

        for (auto &space_entries : schedule.fixed_offset_allocations)
            space_entries.second.clear();
        schedule.max_fixed_allocation_size.clear();
        for (const auto &pair : state.selected_vars[b])
        {
            FixedAlloc alloc;
            if (!getAlloc(state, b, pair.first, alloc,
                          /*require_fixed_offset=*/true, /*require_fixed_start=*/false))
                continue;

            auto &entries = schedule.fixed_offset_allocations[alloc.mem_space];
            entries.push_back({alloc.cid, alloc.mem_space, alloc.offset,
                               static_cast<uint64_t>(alloc.offset) + alloc.size,
                               alloc.size, alloc.en_idx, alloc.en_id, alloc.start_var,
                               alloc.offset_var, alloc.is_view,
                               alloc.is_input_or_cache, alloc.is_root});
            auto &max_size = schedule.max_fixed_allocation_size[alloc.mem_space];
            max_size = std::max(max_size, alloc.size);
        }

        for (auto &space_entries : schedule.fixed_offset_allocations)
        {
            auto &entries = space_entries.second;
            std::sort(entries.begin(), entries.end(), [](const auto &lhs, const auto &rhs) {
                return (lhs.offset != rhs.offset) ? lhs.offset < rhs.offset : lhs.cid.value < rhs.cid.value;
            });
        }
        schedule.spatial_dirty = false;
    }

    static bool mayOverlapInTime(const SearchState &state, uint32_t b, EClassId a_cid, EClassId c_cid)
    {
        const auto &class_info = state.propagation.write_after_read[b].class_info;
        if (a_cid.value >= class_info.size() || c_cid.value >= class_info.size())
            return true;
        const auto &a_info = class_info[a_cid.value];
        const auto &c_info = class_info[c_cid.value];
        const auto a_st_it = state.start_vars[b].find(a_cid);
        const auto c_st_it = state.start_vars[b].find(c_cid);
        const bool a_fixed = a_st_it != state.start_vars[b].end() &&
                             a_info.en_idx < a_st_it->second.size() &&
                             state.domains[a_st_it->second[a_info.en_idx]].isFixed();
        const bool c_fixed = c_st_it != state.start_vars[b].end() &&
                             c_info.en_idx < c_st_it->second.size() &&
                             state.domains[c_st_it->second[c_info.en_idx]].isFixed();
        if (!a_fixed || !c_fixed || a_info.is_input_or_cache || a_info.is_root ||
            c_info.is_input_or_cache || c_info.is_root)
            return true;

        const int32_t a_start = state.domains[a_st_it->second[a_info.en_idx]].fixedValue();
        const int32_t c_start = state.domains[c_st_it->second[c_info.en_idx]].fixedValue();
        if (a_start == c_start)
            return true;
        if (a_start < c_start)
            return c_start <= a_info.max_reader_start_max;
        return a_start <= c_info.max_reader_start_max;
    }

    static bool isViewOf(const SearchState &state, uint32_t b, EClassId base, EClassId view_cand)
    {
        const auto &class_info = state.propagation.write_after_read[b].class_info;
        if (base == view_cand)
            return true;
        EClassId curr = view_cand;
        int step = 0;
        for (; step < 32 && curr != base; ++step)
        {
            if (curr.value < class_info.size() && class_info[curr.value].is_active)
            {
                if (!class_info[curr.value].is_view)
                    return false;
                EClassId p = class_info[curr.value].view_parent;
                if (p.value == UINT32_MAX)
                    return false;
                curr = p;
                continue;
            }
            auto sel_it = state.selected_vars[b].find(curr);
            if (sel_it == state.selected_vars[b].end())
                return false;
            const Domain &dom = state.domains[sel_it->second];
            if (!dom.isFixed() || dom.fixedValue() <= 0)
                return false;
            uint32_t en_idx = static_cast<uint32_t>(dom.fixedValue() - 1);
            const EClass &cls = state.bucket_egraphs[b].getEClass(curr);
            if (en_idx >= cls.enodes.size())
                return false;
            ENodeId en_id = cls.enodes[en_idx];
            bool is_view = en_id.value < state.bucket_enode_infos[b].size() &&
                           state.bucket_enode_infos[b][en_id.value].is_view;
            if (!is_view)
                return false;
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
            if (enode.getChildren().empty())
                return false;
            curr = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
        }
        if (curr == base)
            return true;
        if (step >= 32)
        {
            Error::throw_err("WriteAfterReadPropagator::isViewOf: view chain exceeded 32 steps (possible cycle or excessively deep view)");
        }
        return false;
    }

  private:
    bool canShare(const SearchState &state, FixedAlloc A, FixedAlloc B) const
    {
        const auto &class_info = state.propagation.write_after_read[A.bucket_idx].class_info;
        // - if both A and B are views, return true. ignore
        if (A.is_view && B.is_view)
            return true;

        bool b_is_view_of_a = isViewOf(state, A.bucket_idx, A.cid, B.cid);
        bool a_is_view_of_b = isViewOf(state, A.bucket_idx, B.cid, A.cid);
        if (b_is_view_of_a || a_is_view_of_b)
            return true;

        if (A.start > B.start)
            std::swap(A, B);
        else if (A.start == B.start)
            return false;

        // - if A is INPUT/CACHE/ROOT, or B is INPUT/CACHE/ROOT, return false.
        if (A.is_input_or_cache || A.is_root || B.is_input_or_cache || B.is_root)
            return false;

        int32_t a_max_reader = (A.cid.value < class_info.size() && class_info[A.cid.value].is_active)
                                   ? class_info[A.cid.value].max_reader_start_max
                                   : -1;

        // Fast temporal non-overlap check:
        // If B starts strictly after all readers of A, they cannot overlap in time.
        if (B.start > a_max_reader)
            return true;

        const auto &readers_A = (A.cid.value < class_info.size() && class_info[A.cid.value].is_active)
                                    ? class_info[A.cid.value].readers
                                    : empty_readers;
        bool b_is_reader = std::binary_search(readers_A.begin(), readers_A.end(), B.cid,
                                              [](EClassId x, EClassId y) { return x.value < y.value; });

        if (b_is_reader)
        {
            // - if B is a reader (direct or through view(s)) of A, make sure A is in safe_inplace_idxs. if not, return false.
            const EClass &b_cls = state.bucket_egraphs[B.bucket_idx].getEClass(B.cid);
            const ENode &b_enode = state.bucket_egraphs[B.bucket_idx].getENode(b_cls.enodes[B.en_idx]);
            KernelId b_kid = b_enode.getKernelId();
            if (b_kid.value == 0 || !KernelRegistry::get().hasKernel(b_kid))
                return false;
            const auto &safe_inplace = KernelRegistry::get().getKernel(b_kid).safe_inplace_idxs;

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

            // - if B is a reader, but not (B.offset >= A.offset && B.offset + B.size <= A.offset + A.size) return false.
            if (state.domains[A.offset_var].isFixed())
            {
                const Domain &b_off_dom = state.domains[B.offset_var];
                if (b_off_dom.isFixed())
                {
                    if (!(B.offset >= A.offset && B.offset + B.size <= A.offset + A.size))
                        return false;
                }
                else
                {
                    if (b_off_dom.getMin() > static_cast<int32_t>(A.offset) ||
                        b_off_dom.getMin() + B.size > A.offset + A.size)
                        return false;
                }
            }
        }
        else
        {
            // B is not a reader of A, but B.start <= a_max_reader.
            // B overlaps in time with A's readers -> cannot share buffer.
            return false;
        }

        // - for every reader C = R(A)/B, we need B.start > C.start
        for (EClassId c_cid : readers_A)
        {
            if (c_cid == B.cid)
                continue;
            int32_t c_st_max = (c_cid.value < class_info.size())
                                   ? class_info[c_cid.value].start_max
                                   : -1;
            if (B.start <= c_st_max)
                return false;
        }
        return true;
    }

    bool checkPair(SearchState &state, FixedAlloc A, FixedAlloc B, std::vector<VarId> &worklist)
    {
        const auto &class_info = state.propagation.write_after_read[A.bucket_idx].class_info;
        if (!canShare(state, A, B))
            return false;

        if ((A.is_view && B.is_view) ||
            isViewOf(state, A.bucket_idx, A.cid, B.cid) ||
            isViewOf(state, A.bucket_idx, B.cid, A.cid))
            return true;

        if (A.start > B.start)
            std::swap(A, B);

        const auto &readers_A = (A.cid.value < class_info.size() && class_info[A.cid.value].is_active)
                                    ? class_info[A.cid.value].readers
                                    : empty_readers;
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
                    worklist.push_back(B.start_var);
                }
                if (c_st_dom.setMax(B.start - 1))
                {
                    if (c_st_dom.isEmpty())
                        return false;
                    state.setDomain(c_st_v, c_st_dom);
                    worklist.push_back(c_st_v);
                }
            }
        }
        return true;
    }

    bool enforceDisjoint(SearchState &state, const FixedAlloc &fixed_alloc, FixedAlloc &target,
                         std::vector<VarId> &worklist, bool &out_pushed)
    {
        out_pushed = false;
        Domain target_dom = state.domains[target.offset_var];
        if (target_dom.isEmpty())
            return false;

        uint32_t O = fixed_alloc.offset;
        uint32_t S = fixed_alloc.size;
        uint32_t S_T = target.size;

        int64_t before_threshold = static_cast<int64_t>(O) - S_T;
        uint32_t after_threshold = O + S;

        if (target_dom.getMin() < static_cast<int32_t>(after_threshold) &&
            target_dom.getMax() > static_cast<int32_t>(before_threshold))
        {
            if (target_dom.getMin() > before_threshold)
            {
                // Target cannot be placed before fixed_alloc (even min offset overlaps or is past before_threshold).
                // Target MUST be placed at or after O + S.
                if (target_dom.setMin(static_cast<int32_t>(after_threshold)))
                {
                    if (target_dom.isEmpty())
                        return false;
                    state.setDomain(target.offset_var, target_dom);
                    target.offset = static_cast<uint32_t>(target_dom.getMin());
                    worklist.push_back(target.offset_var);
                    out_pushed = true;
                }
            }
            else if (target_dom.getMax() < static_cast<int32_t>(after_threshold))
            {
                // Target cannot be placed at or after O + S (even max offset is < O + S).
                // Target MUST be placed before fixed_alloc (<= O - S_T).
                if (before_threshold < 0)
                {
                    target_dom = Domain::makeEmpty(target_dom.is_mask);
                    state.setDomain(target.offset_var, target_dom);
                    return false;
                }
                if (target_dom.setMax(static_cast<int32_t>(before_threshold)))
                {
                    if (target_dom.isEmpty())
                        return false;
                    state.setDomain(target.offset_var, target_dom);
                    worklist.push_back(target.offset_var);
                    out_pushed = true;
                }
            }
            else if (target_dom.isFixed())
            {
                return false;
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
        state.ensurePropagationState();
        if (changed != kInvalidVarId)
        {
            VarType type = state.var_infos[changed].type;
            if (type != VarType::OFFSET && type != VarType::START)
                return true;
            uint32_t b = state.var_infos[changed].bucket_idx;
            EClassId cid = state.var_infos[changed].eclass_id;
            FixedAlloc curr;
            if (!getAlloc(state, b, cid, curr, /*require_fixed_offset=*/false))
                return true;

            bool curr_offset_fixed = state.domains[curr.offset_var].isFixed();
            if (type == VarType::START && !curr_offset_fixed)
                return true;

            buildBucketInfo(state, b);
            const auto &temporal_overlaps = state.propagation.write_after_read[b].temporal_overlaps;

            bool pushed = true;
            while (pushed)
            {
                pushed = false;
                for (EClassId other_cid : temporal_overlaps[cid.value])
                {
                    if (other_cid == cid)
                        continue;
                    FixedAlloc other;
                    if (!getAlloc(state, b, other_cid, other, /*require_fixed_offset=*/true))
                        continue;
                    if (curr.mem_space != other.mem_space)
                        continue;

                    if (!canShare(state, curr, other))
                    {
                        if (!enforceDisjoint(state, other, curr, worklist, pushed))
                            return false;
                        if (pushed)
                            break;
                    }
                    else if (curr_offset_fixed)
                    {
                        if (std::max(curr.offset, other.offset) < std::min(curr.offset + curr.size, other.offset + other.size))
                        {
                            if (!checkPair(state, curr, other, worklist))
                                return false;
                        }
                    }
                }
            }

            if (curr_offset_fixed)
            {
                for (EClassId other_cid : temporal_overlaps[cid.value])
                {
                    if (other_cid == cid)
                        continue;
                    FixedAlloc other;
                    if (!getAlloc(state, b, other_cid, other, /*require_fixed_offset=*/false))
                        continue;
                    if (curr.mem_space != other.mem_space)
                        continue;

                    bool other_offset_fixed = state.domains[other.offset_var].isFixed();
                    if (!other_offset_fixed)
                    {
                        // Memory domain filter: if other is already placed after curr in memory
                        const Domain &other_dom = state.domains[other.offset_var];
                        if (other_dom.getMin() >= static_cast<int32_t>(curr.offset + curr.size))
                            continue;

                        if (!canShare(state, curr, other))
                        {
                            bool other_pushed = false;
                            if (!enforceDisjoint(state, curr, other, worklist, other_pushed))
                                return false;
                        }
                    }
                }
            }
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                buildBucketInfo(state, b);
                const auto &temporal_overlaps = state.propagation.write_after_read[b].temporal_overlaps;
                std::vector<FixedAlloc> fixed_allocs;
                std::vector<FixedAlloc> all_allocs;
                std::vector<int32_t> fixed_alloc_index(state.bucket_egraphs[b].classes.size(), -1);
                for (const auto &pair : state.selected_vars[b])
                {
                    FixedAlloc a;
                    if (getAlloc(state, b, pair.first, a, /*require_fixed_offset=*/false))
                    {
                        all_allocs.push_back(a);
                        if (state.domains[a.offset_var].isFixed())
                        {
                            fixed_alloc_index[a.cid.value] = static_cast<int32_t>(fixed_allocs.size());
                            fixed_allocs.push_back(a);
                        }
                    }
                }

                for (auto &target : all_allocs)
                {
                    bool pushed = true;
                    while (pushed)
                    {
                        pushed = false;
                        for (EClassId fixed_cid : temporal_overlaps[target.cid.value])
                        {
                            if (fixed_cid.value >= fixed_alloc_index.size() || fixed_alloc_index[fixed_cid.value] < 0)
                                continue;
                            const FixedAlloc &fixed = fixed_allocs[fixed_alloc_index[fixed_cid.value]];
                            if (target.cid == fixed.cid)
                                continue;
                            if (target.mem_space != fixed.mem_space)
                                continue;

                            if (!canShare(state, target, fixed))
                            {
                                if (!enforceDisjoint(state, fixed, target, worklist, pushed))
                                    return false;
                                if (pushed)
                                    break;
                            }
                            else if (state.domains[target.offset_var].isFixed())
                            {
                                if (std::max(target.offset, fixed.offset) < std::min(target.offset + target.size, fixed.offset + fixed.size))
                                {
                                    if (!checkPair(state, target, fixed, worklist))
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

class FixedOffsetStartPropagator : public Propagator
{
    using FixedAlloc = WriteAfterReadPropagator::FixedAlloc;

    bool resolvePair(SearchState &state, uint32_t b, FixedAlloc A, FixedAlloc B,
                     std::vector<VarId> &worklist)
    {
        // 1. Views check: if both are views, or either is view of the other, they can safely share
        if ((A.is_view && B.is_view) ||
            WriteAfterReadPropagator::isViewOf(state, b, A.cid, B.cid) ||
            WriteAfterReadPropagator::isViewOf(state, b, B.cid, A.cid))
        {
            return true;
        }

        // 2. Persistent check: persistent tensors (INPUT, CACHE, ROOT) can never share memory with non-views
        if (A.is_input_or_cache || A.is_root || B.is_input_or_cache || B.is_root)
        {
            LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [persistent overlap]: A(cid="
                       << A.cid.value << ", in_cache=" << A.is_input_or_cache << ", root=" << A.is_root
                       << ", st=" << state.domains[A.start_var].toString() << ", off=" << A.offset << ", sz=" << A.size
                       << ") and B(cid=" << B.cid.value << ", in_cache=" << B.is_input_or_cache << ", root=" << B.is_root
                       << ", st=" << state.domains[B.start_var].toString() << ", off=" << B.offset << ", sz=" << B.size << ")";
            return false;
        }

        const auto &class_info = state.propagation.write_after_read[b].class_info;
        auto reads = [&](EClassId producer, EClassId consumer) {
            if (producer.value >= class_info.size() || !class_info[producer.value].is_active)
                return false;
            const auto &readers = class_info[producer.value].readers;
            return std::binary_search(readers.begin(), readers.end(), consumer,
                                      [](EClassId x, EClassId y) { return x.value < y.value; });
        };

        auto canReadInPlace = [&](const FixedAlloc &producer, const FixedAlloc &consumer) {
            const EClass &consumer_cls = state.bucket_egraphs[b].getEClass(consumer.cid);
            const ENode &consumer_enode = state.bucket_egraphs[b].getENode(consumer_cls.enodes[consumer.en_idx]);
            KernelId consumer_kid = consumer_enode.getKernelId();
            if (consumer_kid.value == 0 || !KernelRegistry::get().hasKernel(consumer_kid))
                return false;

            const auto &safe_inplace = KernelRegistry::get().getKernel(consumer_kid).safe_inplace_idxs;
            bool found_safe = false;
            for (size_t in_idx = 0; in_idx < consumer_enode.getChildren().size(); ++in_idx)
            {
                EClassId child_cid = state.bucket_egraphs[b].findConst(consumer_enode.getChildren()[in_idx]);
                if (child_cid == producer.cid || WriteAfterReadPropagator::isViewOf(state, b, producer.cid, child_cid))
                {
                    if (std::find(safe_inplace.begin(), safe_inplace.end(), static_cast<uint32_t>(in_idx)) != safe_inplace.end())
                    {
                        found_safe = true;
                        break;
                    }
                }
            }
            const bool fits_in_producer = consumer.offset >= producer.offset &&
                                          consumer.offset + consumer.size <= producer.offset + producer.size;
            return found_safe && fits_in_producer;
        };

        // An unsafe producer/consumer overlap is impossible independent of
        // dispatch order. Detect it before branching either start variable.
        if (!state.domains[A.start_var].isFixed() && !state.domains[B.start_var].isFixed())
        {
            if ((reads(A.cid, B.cid) && !canReadInPlace(A, B)) ||
                (reads(B.cid, A.cid) && !canReadInPlace(B, A)))
            {
                LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [unsafe inplace before starts fixed] for A(cid="
                           << A.cid.value << ") and B(cid=" << B.cid.value << ")";
                return false;
            }
            return true;
        }

        if (!state.domains[A.start_var].isFixed() && state.domains[B.start_var].isFixed())
            std::swap(A, B);

        if (state.domains[A.start_var].isFixed() && state.domains[B.start_var].isFixed())
        {
            if (A.start > B.start)
                std::swap(A, B);
            else if (A.start == B.start)
            {
                LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [same start]: identical start=" << A.start
                           << " for non-views A(cid=" << A.cid.value << ", off=" << A.offset << ", sz=" << A.size
                           << ") and B(cid=" << B.cid.value << ", off=" << B.offset << ", sz=" << B.size << ")";
                return false;
            }
        }

        const auto &readers_A = (A.cid.value < class_info.size() && class_info[A.cid.value].is_active)
                                    ? class_info[A.cid.value].readers
                                    : WriteAfterReadPropagator::empty_readers;
        const auto &readers_B = (B.cid.value < class_info.size() && class_info[B.cid.value].is_active)
                                    ? class_info[B.cid.value].readers
                                    : WriteAfterReadPropagator::empty_readers;
        const bool b_reads_a = reads(A.cid, B.cid);
        const bool a_reads_b = reads(B.cid, A.cid);

        // 3. B reads A
        if (b_reads_a)
        {
            if (!canReadInPlace(A, B))
            {
                LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [B reads A: unsafe inplace] for A(cid="
                           << A.cid.value << ") and B(cid=" << B.cid.value << ")";
                return false;
            }

            int32_t min_b_start = A.start + 1;
            for (EClassId c_cid : readers_A)
            {
                if (c_cid == B.cid)
                    continue;
                auto c_sel_it = state.selected_vars[b].find(c_cid);
                if (c_sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &c_sel_dom = state.domains[c_sel_it->second];
                if (c_sel_dom.isFixed() && c_sel_dom.fixedValue() == 0)
                    continue;

                auto c_st_it = state.start_vars[b].find(c_cid);
                if (c_st_it == state.start_vars[b].end())
                    continue;

                for (uint32_t c_en = 0; c_en < c_st_it->second.size(); ++c_en)
                {
                    if (!c_sel_dom.contains(c_en + 1))
                        continue;
                    VarId c_st_v = c_st_it->second[c_en];
                    const Domain &c_st_dom = state.domains[c_st_v];
                    min_b_start = std::max(min_b_start, c_st_dom.getMin() + 1);

                    Domain b_st_dom = state.domains[B.start_var];
                    if (b_st_dom.getMax() - 1 < c_st_dom.getMax())
                    {
                        Domain new_c_dom = c_st_dom;
                        if (new_c_dom.setMax(b_st_dom.getMax() - 1))
                        {
                            if (new_c_dom.isEmpty())
                            {
                                LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [B reads A: reader empty]: C(cid="
                                           << c_cid.value << ") domain empty after setMax(" << (b_st_dom.getMax() - 1) << ")";
                                return false;
                            }
                            state.setDomain(c_st_v, new_c_dom);
                            worklist.push_back(c_st_v);
                        }
                    }
                }
            }
            Domain b_st_dom = state.domains[B.start_var];
            if (b_st_dom.setMin(min_b_start))
            {
                if (b_st_dom.isEmpty())
                {
                    LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [B reads A: B start empty]: B(cid="
                               << B.cid.value << ") domain empty after setMin(" << min_b_start << ")";
                    return false;
                }
                state.setDomain(B.start_var, b_st_dom);
                worklist.push_back(B.start_var);
            }
            return true;
        }

        // 4. A reads B
        if (a_reads_b)
        {
            if (!canReadInPlace(B, A))
            {
                LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [A reads B: unsafe inplace] for A(cid="
                           << A.cid.value << ") and B(cid=" << B.cid.value << ")";
                return false;
            }

            Domain b_st_dom = state.domains[B.start_var];
            if (b_st_dom.setMax(A.start - 1))
            {
                if (b_st_dom.isEmpty())
                {
                    LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [A reads B: B start empty]: B(cid="
                               << B.cid.value << ") domain empty after setMax(" << (A.start - 1) << ")";
                    return false;
                }
                state.setDomain(B.start_var, b_st_dom);
                worklist.push_back(B.start_var);
            }

            for (EClassId c_cid : readers_B)
            {
                if (c_cid == A.cid)
                    continue;
                auto c_sel_it = state.selected_vars[b].find(c_cid);
                if (c_sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &c_sel_dom = state.domains[c_sel_it->second];
                if (c_sel_dom.isFixed() && c_sel_dom.fixedValue() == 0)
                    continue;

                auto c_st_it = state.start_vars[b].find(c_cid);
                if (c_st_it == state.start_vars[b].end())
                    continue;

                for (uint32_t c_en = 0; c_en < c_st_it->second.size(); ++c_en)
                {
                    if (!c_sel_dom.contains(c_en + 1))
                        continue;
                    VarId c_st_v = c_st_it->second[c_en];
                    Domain c_st_dom = state.domains[c_st_v];
                    if (c_st_dom.setMax(A.start - 1))
                    {
                        if (c_st_dom.isEmpty())
                        {
                            LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [A reads B: reader empty]: C(cid="
                                       << c_cid.value << ") domain empty after setMax(" << (A.start - 1) << ")";
                            return false;
                        }
                        state.setDomain(c_st_v, c_st_dom);
                        worklist.push_back(c_st_v);
                    }
                }
            }
            return true;
        }

        // 5. Neither reads the other: lifespans must be disjoint
        int32_t t_after_a = A.start + 1;
        for (EClassId c_cid : readers_A)
        {
            auto c_sel_it = state.selected_vars[b].find(c_cid);
            if (c_sel_it == state.selected_vars[b].end())
                continue;
            const Domain &c_sel_dom = state.domains[c_sel_it->second];
            if (c_sel_dom.isFixed() && c_sel_dom.fixedValue() == 0)
                continue;

            auto c_st_it = state.start_vars[b].find(c_cid);
            if (c_st_it == state.start_vars[b].end())
                continue;

            for (uint32_t c_en = 0; c_en < c_st_it->second.size(); ++c_en)
            {
                if (c_sel_dom.contains(c_en + 1))
                {
                    VarId c_st_v = c_st_it->second[c_en];
                    t_after_a = std::max(t_after_a, state.domains[c_st_v].getMin() + 1);
                }
            }
        }

        Domain b_st_dom = state.domains[B.start_var];
        bool b_cannot_be_before_a = (b_st_dom.getMin() >= A.start);
        if (!b_cannot_be_before_a)
        {
            for (EClassId d_cid : readers_B)
            {
                auto d_sel_it = state.selected_vars[b].find(d_cid);
                if (d_sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &d_sel_dom = state.domains[d_sel_it->second];
                if (d_sel_dom.isFixed() && d_sel_dom.fixedValue() == 0)
                    continue;

                // An active reader of B needs at least B_min + 1; if that is >= A.start, it can never finish before A
                if (b_st_dom.getMin() + 1 >= A.start)
                {
                    b_cannot_be_before_a = true;
                    break;
                }

                auto d_st_it = state.start_vars[b].find(d_cid);
                if (d_st_it == state.start_vars[b].end())
                    continue;

                for (uint32_t d_en = 0; d_en < d_st_it->second.size(); ++d_en)
                {
                    if (d_sel_dom.contains(d_en + 1))
                    {
                        if (state.domains[d_st_it->second[d_en]].getMin() >= A.start)
                        {
                            b_cannot_be_before_a = true;
                            break;
                        }
                    }
                }
                if (b_cannot_be_before_a)
                    break;
            }
        }

        bool b_cannot_be_after_a = (b_st_dom.getMax() < t_after_a);

        if (b_cannot_be_before_a && b_cannot_be_after_a)
        {
            LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [disjoint lifespans impossible]: A(cid="
                       << A.cid.value << ", st=" << A.start << ", t_after_a=" << t_after_a
                       << ") vs B(cid=" << B.cid.value << ", dom=" << b_st_dom.toString()
                       << "): b_cannot_be_before_a=1 && b_cannot_be_after_a=1";
            return false;
        }


        if (b_cannot_be_before_a)
        {
            if (b_st_dom.setMin(t_after_a))
            {
                if (b_st_dom.isEmpty())
                {
                    LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [disjoint: B setMin empty]: B(cid="
                               << B.cid.value << ") domain empty after setMin(" << t_after_a << ")";
                    return false;
                }
                state.setDomain(B.start_var, b_st_dom);
                worklist.push_back(B.start_var);
            }
            for (EClassId c_cid : readers_A)
            {
                auto c_sel_it = state.selected_vars[b].find(c_cid);
                if (c_sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &c_sel_dom = state.domains[c_sel_it->second];
                if (c_sel_dom.isFixed() && c_sel_dom.fixedValue() == 0)
                    continue;

                auto c_st_it = state.start_vars[b].find(c_cid);
                if (c_st_it == state.start_vars[b].end())
                    continue;

                for (uint32_t c_en = 0; c_en < c_st_it->second.size(); ++c_en)
                {
                    if (!c_sel_dom.contains(c_en + 1))
                        continue;
                    VarId c_st_v = c_st_it->second[c_en];
                    Domain c_st_dom = state.domains[c_st_v];
                    if (c_st_dom.setMax(b_st_dom.getMax() - 1))
                    {
                        if (c_st_dom.isEmpty())
                        {
                            LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [disjoint: reader C empty]: C(cid="
                                       << c_cid.value << ") domain empty after setMax(" << (b_st_dom.getMax() - 1) << ")";
                            return false;
                        }
                        state.setDomain(c_st_v, c_st_dom);
                        worklist.push_back(c_st_v);
                    }
                }
            }
        }
        else if (b_cannot_be_after_a)
        {
            if (b_st_dom.setMax(A.start - 1))
            {
                if (b_st_dom.isEmpty())
                {
                    LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [disjoint: B setMax empty]: B(cid="
                               << B.cid.value << ") domain empty after setMax(" << (A.start - 1) << ")";
                    return false;
                }
                state.setDomain(B.start_var, b_st_dom);
                worklist.push_back(B.start_var);
            }
            for (EClassId d_cid : readers_B)
            {
                auto d_sel_it = state.selected_vars[b].find(d_cid);
                if (d_sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &d_sel_dom = state.domains[d_sel_it->second];
                if (d_sel_dom.isFixed() && d_sel_dom.fixedValue() == 0)
                    continue;

                auto d_st_it = state.start_vars[b].find(d_cid);
                if (d_st_it == state.start_vars[b].end())
                    continue;

                for (uint32_t d_en = 0; d_en < d_st_it->second.size(); ++d_en)
                {
                    if (!d_sel_dom.contains(d_en + 1))
                        continue;
                    VarId d_st_v = d_st_it->second[d_en];
                    Domain d_st_dom = state.domains[d_st_v];
                    if (d_st_dom.setMax(A.start - 1))
                    {
                        if (d_st_dom.isEmpty())
                        {
                            LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [disjoint: reader D empty]: D(cid="
                                       << d_cid.value << ") domain empty after setMax(" << (A.start - 1) << ")";
                            return false;
                        }
                        state.setDomain(d_st_v, d_st_dom);
                        worklist.push_back(d_st_v);
                    }
                }
            }
        }
        else if (b_st_dom.is_mask)
        {
            bool changed_mask = false;
            for (int32_t t = std::max(0, A.start); t < std::min(32, t_after_a); ++t)
            {
                if (b_st_dom.remove(t))
                    changed_mask = true;
            }
            if (changed_mask)
            {
                if (b_st_dom.isEmpty())
                {
                    LOG(DEBUG) << "[FixedOffsetStartPropagator] Conflict [disjoint: mask empty]: B(cid="
                               << B.cid.value << ") domain empty";
                    return false;
                }
                state.setDomain(B.start_var, b_st_dom);
                worklist.push_back(B.start_var);
            }
        }

        return true;
    }

  public:
    std::string name() const override
    {
        return "FixedOffsetStartPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        state.ensurePropagationState();
        if (changed != kInvalidVarId)
        {
            const VarInfo &info = state.var_infos[changed];
            if (info.type != VarType::OFFSET && info.type != VarType::START)
                return true;

            // If offset changed, it must be fixed to establish memory footprint
            if (info.type == VarType::OFFSET && !state.domains[changed].isFixed())
                return true;

            uint32_t b = info.bucket_idx;
            EClassId cid = info.eclass_id;

            // Eclass must have fixed offset to establish memory footprint
            auto off_it = state.offset_vars[b].find(cid);
            if (off_it == state.offset_vars[b].end() || !state.domains[off_it->second].isFixed())
                return true;

            FixedAlloc curr;
            if (!WriteAfterReadPropagator::getAlloc(state, b, cid, curr, /*require_fixed_offset=*/true, /*require_fixed_start=*/false))
                return true;

            // If START changed, ensure it matches the active/selected enode
            if (info.type == VarType::START && info.enode_idx != curr.en_idx)
                return true;

            WriteAfterReadPropagator::buildBucketInfo(state, b);
            WriteAfterReadPropagator::buildFixedOffsetIndex(state, b);
            auto &schedule = state.propagation.write_after_read[b];
            auto space_it = schedule.fixed_offset_allocations.find(curr.mem_space);
            if (space_it == schedule.fixed_offset_allocations.end())
                return true;
            const auto &candidates = space_it->second;
            const uint32_t max_size = schedule.max_fixed_allocation_size[curr.mem_space];
            const uint64_t first_possible_offset = curr.offset >= max_size
                                                       ? static_cast<uint64_t>(curr.offset) - max_size + 1
                                                       : 0;
            const uint64_t curr_end = static_cast<uint64_t>(curr.offset) + curr.size;
            auto candidate_it = std::lower_bound(candidates.begin(), candidates.end(), first_possible_offset,
                                                 [](const auto &candidate, uint64_t offset) {
                                                     return candidate.offset < offset;
                                                 });

            bool curr_start_fixed = state.domains[curr.start_var].isFixed();
            for (; candidate_it != candidates.end() && candidate_it->offset < curr_end; ++candidate_it)
            {
                const auto &candidate = *candidate_it;
                if (candidate.cid == cid || candidate.end <= curr.offset)
                    continue;
                if (!WriteAfterReadPropagator::mayOverlapInTime(state, b, cid, candidate.cid))
                    continue;

                FixedAlloc other;
                other.cid = candidate.cid;
                other.bucket_idx = b;
                other.mem_space = candidate.mem_space;
                other.offset = candidate.offset;
                other.size = candidate.size;
                other.start = state.domains[candidate.start_var].getMin();
                other.en_idx = candidate.en_idx;
                other.en_id = candidate.en_id;
                other.start_var = candidate.start_var;
                other.offset_var = candidate.offset_var;
                other.is_view = candidate.is_view;
                other.is_input_or_cache = candidate.is_input_or_cache;
                other.is_root = candidate.is_root;

                bool resolved = true;
                if (curr_start_fixed)
                {
                    resolved = resolvePair(state, b, curr, other, worklist);
                }
                else if (!state.domains[other.start_var].isFixed())
                {
                    // resolvePair can reject unsafe producer/consumer sharing
                    // without fixed starts; other temporal checks still defer.
                    resolved = resolvePair(state, b, curr, other, worklist);
                }
                else
                {
                    resolved = resolvePair(state, b, other, curr, worklist);
                }

                if (!resolved)
                {
                    LOG(DEBUG) << "[FixedOffsetStartPropagator] Pruned left branch cid=" << cid.value
                               << " (start=" << curr.start << ") due to conflict with other_cid=" << candidate.cid.value
                               << " (other_start=" << state.domains[other.start_var].toString()
                               << ", other_off=" << other.offset << ", other_sz=" << other.size << ")";
                    return false;
                }
            }
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                WriteAfterReadPropagator::buildBucketInfo(state, b);
                const auto &active_cids = state.propagation.write_after_read[b].active_cids;

                std::vector<FixedAlloc> fixed_start_allocs;
                std::vector<FixedAlloc> unfixed_start_allocs;

                for (EClassId cid : active_cids)
                {
                    FixedAlloc a;
                    if (WriteAfterReadPropagator::getAlloc(state, b, cid, a, /*require_fixed_offset=*/true, /*require_fixed_start=*/true))
                    {
                        fixed_start_allocs.push_back(a);
                    }
                    else if (WriteAfterReadPropagator::getAlloc(state, b, cid, a, /*require_fixed_offset=*/true, /*require_fixed_start=*/false))
                    {
                        unfixed_start_allocs.push_back(a);
                    }
                }

                // Check fixed with fixed
                for (size_t i = 0; i < fixed_start_allocs.size(); ++i)
                {
                    for (size_t j = i + 1; j < fixed_start_allocs.size(); ++j)
                    {
                        const auto &a = fixed_start_allocs[i];
                        const auto &b_alloc = fixed_start_allocs[j];
                        if (a.mem_space != b_alloc.mem_space)
                            continue;
                        if (std::max(a.offset, b_alloc.offset) >= std::min(a.offset + a.size, b_alloc.offset + b_alloc.size))
                            continue;
                        if (!resolvePair(state, b, a, b_alloc, worklist))
                            return false;
                    }
                }

                // Check fixed with unfixed
                for (const auto &a : fixed_start_allocs)
                {
                    for (const auto &b_alloc : unfixed_start_allocs)
                    {
                        if (a.mem_space != b_alloc.mem_space)
                            continue;
                        if (std::max(a.offset, b_alloc.offset) >= std::min(a.offset + a.size, b_alloc.offset + b_alloc.size))
                            continue;
                        if (!resolvePair(state, b, a, b_alloc, worklist))
                            return false;
                    }
                }

                // Check unfixed with unfixed if either is persistent
                for (size_t i = 0; i < unfixed_start_allocs.size(); ++i)
                {
                    for (size_t j = i + 1; j < unfixed_start_allocs.size(); ++j)
                    {
                        const auto &a = unfixed_start_allocs[i];
                        const auto &b_alloc = unfixed_start_allocs[j];
                        if (a.mem_space != b_alloc.mem_space)
                            continue;
                        if (std::max(a.offset, b_alloc.offset) >= std::min(a.offset + a.size, b_alloc.offset + b_alloc.size))
                            continue;
                        if (a.is_input_or_cache || a.is_root || b_alloc.is_input_or_cache || b_alloc.is_root)
                        {
                            if (!resolvePair(state, b, a, b_alloc, worklist))
                                return false;
                        }
                    }
                }
            }
        }
        return true;
    }
};

// If two active values have overlapping fixed addresses and their start
// domains already establish an order, push the later value past the earlier
// value's readers before branching on each individual start.
class WriteAfterReadStartPropagator : public Propagator
{
    struct ActiveAllocation
    {
        EClassId cid;
        BaseEClassId base_id;
        MemSpace mem_space;
        int32_t offset;
        uint32_t size;
        VarId start_var;
        Domain start_domain;
        bool is_view;
        bool is_persistent;
    };

    bool addReaderAncestors(const SearchState &state, uint32_t b, EClassId child,
                            EClassId reader, std::vector<std::vector<EClassId>> &readers_by_value) const
    {
        for (uint32_t step = 0; step < 32; ++step)
        {
            child = state.bucket_egraphs[b].findConst(child);
            if (child.value >= readers_by_value.size())
                return true;
            readers_by_value[child.value].push_back(reader);

            auto sel_it = state.selected_vars[b].find(child);
            if (sel_it == state.selected_vars[b].end())
                return true;
            const Domain &selection = state.domains[sel_it->second];
            if (!selection.isFixed() || selection.fixedValue() <= 0)
                return true;
            const EClass &cls = state.bucket_egraphs[b].getEClass(child);
            const uint32_t en_idx = static_cast<uint32_t>(selection.fixedValue() - 1);
            if (en_idx >= cls.enodes.size())
                return true;
            const ENodeId en_id = cls.enodes[en_idx];
            if (en_id.value >= state.bucket_enode_infos[b].size() ||
                !state.bucket_enode_infos[b][en_id.value].is_view)
                return true;
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
            if (enode.getChildren().empty())
                return true;
            child = enode.getChildren()[0];
        }
        return true;
    }

    bool propagateBucket(SearchState &state, uint32_t b, std::vector<VarId> &worklist) const
    {
        const size_t num_classes = state.bucket_egraphs[b].classes.size();
        std::vector<ActiveAllocation> active;
        active.reserve(state.selected_vars[b].size());

        for (const auto &selected_pair : state.selected_vars[b])
        {
            const EClassId cid = selected_pair.first;
            const Domain &selection = state.domains[selected_pair.second];
            if (!selection.isFixed() || selection.fixedValue() <= 0)
                continue;
            const uint32_t en_idx = static_cast<uint32_t>(selection.fixedValue() - 1);
            const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
            if (en_idx >= cls.enodes.size())
                continue;

            const auto offset_it = state.offset_vars[b].find(cid);
            const auto start_it = state.start_vars[b].find(cid);
            if (offset_it == state.offset_vars[b].end() || start_it == state.start_vars[b].end() ||
                en_idx >= start_it->second.size())
                continue;
            const Domain &offset_domain = state.domains[offset_it->second];
            if (!offset_domain.isFixed())
                continue;

            const ENodeId en_id = cls.enodes[en_idx];
            const ENode &enode = state.bucket_egraphs[b].getENode(en_id);
            const bool is_view = en_id.value < state.bucket_enode_infos[b].size() &&
                                 state.bucket_enode_infos[b][en_id.value].is_view;
            const bool is_root = b < state.bucket_root_ids.size() &&
                                 state.bucket_egraphs[b].findConst(cid) ==
                                     state.bucket_egraphs[b].findConst(state.bucket_root_ids[b]);
            const bool is_persistent = is_root || enode.getOpType() == OpType::INPUT ||
                                       enode.getOpType() == OpType::CACHE ||
                                       state.preallocated_buffers.count(cls.base_eclass_id) != 0;
            const uint32_t size_pages = std::max<uint32_t>(
                1, state.bytesToPages(getSizeBytes(cls.shape, cls.dtype), cls.mem_space));

            active.push_back({cid, cls.base_eclass_id, cls.mem_space, offset_domain.fixedValue(),
                              size_pages, start_it->second[en_idx],
                              state.domains[start_it->second[en_idx]], is_view, is_persistent});
        }

        if (active.size() < 2)
            return true;

        std::vector<std::vector<EClassId>> readers_by_value(num_classes);
        for (const auto &selected_pair : state.selected_vars[b])
        {
            const EClassId reader = selected_pair.first;
            const Domain &selection = state.domains[selected_pair.second];
            if (!selection.isFixed() || selection.fixedValue() <= 0)
                continue;
            const EClass &cls = state.bucket_egraphs[b].getEClass(reader);
            const uint32_t en_idx = static_cast<uint32_t>(selection.fixedValue() - 1);
            if (en_idx >= cls.enodes.size())
                continue;
            const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
            for (EClassId child : enode.getChildren())
                addReaderAncestors(state, b, child, reader, readers_by_value);
        }

        std::unordered_map<MemSpace, std::vector<size_t>> allocations_by_space;
        for (size_t i = 0; i < active.size(); ++i)
            allocations_by_space[active[i].mem_space].push_back(i);

        for (auto &space_pair : allocations_by_space)
        {
            auto &indices = space_pair.second;
            std::sort(indices.begin(), indices.end(), [&](size_t lhs, size_t rhs) {
                return active[lhs].offset < active[rhs].offset;
            });

            for (size_t i = 0; i < indices.size(); ++i)
            {
                const ActiveAllocation &first = active[indices[i]];
                const int64_t first_end = static_cast<int64_t>(first.offset) + first.size;
                for (size_t j = i + 1; j < indices.size() && active[indices[j]].offset < first_end; ++j)
                {
                    const ActiveAllocation &second = active[indices[j]];
                    if (first.cid == second.cid || first.base_id == second.base_id || first.is_view ||
                        second.is_view || first.is_persistent || second.is_persistent)
                        continue;
                    if (static_cast<int64_t>(second.offset) + second.size <= first.offset)
                        continue;

                    const bool first_before_second = first.start_domain.getMax() < second.start_domain.getMin();
                    const bool second_before_first = second.start_domain.getMax() < first.start_domain.getMin();
                    if (!first_before_second && !second_before_first)
                        continue;

                    const ActiveAllocation &earlier = first_before_second ? first : second;
                    const ActiveAllocation &later = first_before_second ? second : first;
                    if (earlier.cid.value >= readers_by_value.size())
                        continue;

                    const auto &readers = readers_by_value[earlier.cid.value];
                    int32_t earliest_last_reader = -1;
                    bool later_reads_earlier = false;
                    for (EClassId reader_cid : readers)
                    {
                        if (reader_cid == later.cid)
                        {
                            later_reads_earlier = true;
                            break;
                        }
                        const auto reader_start_it = state.start_vars[b].find(reader_cid);
                        const auto reader_sel_it = state.selected_vars[b].find(reader_cid);
                        if (reader_start_it == state.start_vars[b].end() ||
                            reader_sel_it == state.selected_vars[b].end())
                            continue;
                        const Domain &reader_selection = state.domains[reader_sel_it->second];
                        if (!reader_selection.isFixed() || reader_selection.fixedValue() <= 0)
                            continue;
                        const uint32_t reader_en_idx = static_cast<uint32_t>(reader_selection.fixedValue() - 1);
                        if (reader_en_idx >= reader_start_it->second.size())
                            continue;
                        earliest_last_reader = std::max(
                            earliest_last_reader,
                            state.domains[reader_start_it->second[reader_en_idx]].getMin());
                    }
                    if (later_reads_earlier || earliest_last_reader < 0)
                        continue;

                    Domain later_start = state.domains[later.start_var];
                    const int32_t required_start = earliest_last_reader + 1;
                    if (later_start.setMin(required_start))
                    {
                        if (later_start.isEmpty())
                            return false;
                        state.setDomain(later.start_var, later_start);
                        worklist.push_back(later.start_var);
                    }
                }
            }
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "WriteAfterReadStartPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed == kInvalidVarId)
            return true;
        const VarInfo &info = state.var_infos[changed];
        if (info.type != VarType::START || !state.domains[changed].isFixed())
            return true;

        const auto selected_it = state.selected_vars[info.bucket_idx].find(info.eclass_id);
        if (selected_it == state.selected_vars[info.bucket_idx].end())
            return true;
        const Domain &selection = state.domains[selected_it->second];
        if (!selection.isFixed() || selection.fixedValue() != static_cast<int32_t>(info.enode_idx + 1))
            return true;

        return propagateBucket(state, info.bucket_idx, worklist);
    }
};

// 8. if VarType::OFFSET || (VarType::SELECTED && dom.isFixed() && dom.fixedValue() > 0),
// whenever an eclass is confirmed to be a view of a base tensor, bidirectionally intersect
// offset domains [max(base.min, view.min)..min(base.max, view.max)]. If disjoint, return false.
// Also remove unviable view enodes from selection if offset domains cannot overlap.
class ViewOffsetPropagator : public Propagator
{
    std::vector<std::unordered_map<EClassId, std::vector<EClassId>>> bucket_view_users;
    bool initialized = false;

    void ensureInitialized(const SearchState &state)
    {
        if (initialized && bucket_view_users.size() == state.buckets.size())
            return;
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
        initialized = true;
    }

    bool getConfirmedViewBase(const SearchState &state, uint32_t b, EClassId cid, EClassId &out_base_cid) const
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

    bool intersectViewAndBase(SearchState &state, uint32_t b, EClassId view_cid, EClassId base_cid,
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

    bool filterUnviableViewEnodes(SearchState &state, uint32_t b, EClassId v_cid, EClassId base_cid,
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

    bool propagateEClass(SearchState &state, uint32_t b, EClassId cid, std::vector<VarId> &worklist)
    {
        // 1. Match views with base: follow confirmed view chain upwards from cid
        EClassId curr = cid;
        bool is_confirmed_view = false;
        for (uint32_t step = 0; step < 32; ++step)
        {
            EClassId next_base;
            if (!getConfirmedViewBase(state, b, curr, next_base))
                break;
            is_confirmed_view = true;
            if (!intersectViewAndBase(state, b, curr, next_base, worklist))
                return false;
            curr = next_base;
        }

        // If cid itself is not a confirmed view, filter any unviable view enodes in cid
        if (!is_confirmed_view)
        {
            const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
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
                    EClassId cand_base = state.bucket_egraphs[b].findConst(enode.getChildren()[0]);
                    if (!filterUnviableViewEnodes(state, b, cid, cand_base, worklist))
                        return false;
                }
            }
        }

        // 2. Match a base with views: propagate cid with all its view users
        if (b < bucket_view_users.size())
        {
            auto it = bucket_view_users[b].find(cid);
            if (it != bucket_view_users[b].end())
            {
                for (EClassId v_cid : it->second)
                {
                    EClassId v_base_cid;
                    if (getConfirmedViewBase(state, b, v_cid, v_base_cid) && v_base_cid == cid)
                    {
                        if (!intersectViewAndBase(state, b, v_cid, cid, worklist))
                            return false;
                    }
                    else
                    {
                        if (!filterUnviableViewEnodes(state, b, v_cid, cid, worklist))
                            return false;
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
        ensureInitialized(state);

        if (changed == kInvalidVarId)
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                for (const auto &pair : state.selected_vars[b])
                {
                    if (!propagateEClass(state, b, pair.first, worklist))
                        return false;
                }
            }
            return true;
        }

        const auto &info = state.var_infos[changed];
        if (info.type != VarType::OFFSET && info.type != VarType::SELECTED)
            return true;

        return propagateEClass(state, info.bucket_idx, info.eclass_id, worklist);
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
            const VarInfo &changed_info = state.var_infos[changed];
            if (changed_info.type != VarType::SELECTED)
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

// Offsets of a globally cached eclass must match across bucket arenas so a
// CACHE enode aliases the buffer populated by the corresponding computation.
class CachedOffsetPropagator : public Propagator
{
    bool offset_index_initialized_ = false;
    std::unordered_map<BaseEClassId, std::vector<VarId>> offsets_by_base_;

    void ensureOffsetIndex(const SearchState &state)
    {
        if (offset_index_initialized_)
            return;
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &offset_pair : state.offset_vars[b])
            {
                const BaseEClassId base_id = state.bucket_egraphs[b].getEClass(offset_pair.first).base_eclass_id;
                offsets_by_base_[base_id].push_back(offset_pair.second);
            }
        }
        offset_index_initialized_ = true;
    }

    bool synchronizeOffset(SearchState &state, BaseEClassId base_id, VarId source_var)
    {
        const Domain &source_domain = state.domains[source_var];
        if (!source_domain.isFixed())
            return true;
        const int32_t shared_offset = source_domain.fixedValue();

        auto offsets_it = offsets_by_base_.find(base_id);
        if (offsets_it == offsets_by_base_.end())
            return true;
        for (VarId other_var : offsets_it->second)
        {
            if (other_var == source_var)
                continue;

            const Domain &other_domain = state.domains[other_var];
            if (!other_domain.contains(shared_offset))
                return false;
            state.setDomain(other_var, Domain::makeFixed(shared_offset));
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "CachedOffsetPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed == kInvalidVarId)
            return true;

        ensureOffsetIndex(state);
        const VarInfo &info = state.var_infos[changed];
        BaseEClassId base_id;
        if (info.type == VarType::OFFSET)
        {
            base_id = state.bucket_egraphs[info.bucket_idx].getEClass(info.eclass_id).base_eclass_id;
            const auto cache_it = state.cached_vars.find(base_id);
            if (cache_it == state.cached_vars.end())
                return true;
            const Domain &cache_dom = state.domains[cache_it->second];
            if (!cache_dom.isFixed() || cache_dom.fixedValue() != 1)
                return true;
            return synchronizeOffset(state, base_id, changed);
        }

        if (info.type != VarType::CACHED)
            return true;
        const Domain &cache_dom = state.domains[changed];
        if (!cache_dom.isFixed() || cache_dom.fixedValue() != 1)
            return true;
        base_id = info.base_eclass_id;
        auto offsets_it = offsets_by_base_.find(base_id);
        if (offsets_it != offsets_by_base_.end())
        {
            for (VarId offset_var : offsets_it->second)
            {
                if (state.domains[offset_var].isFixed())
                    return synchronizeOffset(state, base_id, offset_var);
            }
        }
        return true;
    }
};

// Allocate a stable arena range as soon as a cache candidate is selected.
// Cached values live across buckets, so their offsets must agree everywhere
// and must not overlap any other persistent cached value.
class CachedOffsetAllocationPropagator : public Propagator
{
    bool offset_index_initialized_ = false;
    std::unordered_map<BaseEClassId, std::vector<VarId>> offsets_by_base_;

    void ensureOffsetIndex(const SearchState &state)
    {
        if (offset_index_initialized_)
            return;
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            for (const auto &offset_pair : state.offset_vars[b])
            {
                const BaseEClassId base_id = state.bucket_egraphs[b].getEClass(offset_pair.first).base_eclass_id;
                offsets_by_base_[base_id].push_back(offset_pair.second);
            }
        }
        offset_index_initialized_ = true;
    }

    uint32_t getMaxSizePages(const SearchState &state, BaseEClassId base_id) const
    {
        uint32_t max_pages = 1;
        const auto cache_it = state.cached_vars.find(base_id);
        if (cache_it != state.cached_vars.end())
        {
            const VarInfo &cinfo = state.var_infos[cache_it->second];
            max_pages = std::max(max_pages, state.bytesToPages(cinfo.size_bytes, cinfo.mem_space));
        }
        const auto off_it = offsets_by_base_.find(base_id);
        if (off_it != offsets_by_base_.end())
        {
            for (VarId off_var : off_it->second)
            {
                max_pages = std::max(max_pages, state.var_infos[off_var].size_pages);
            }
        }
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            EClassId cid = state.bucket_egraphs[b].findEClassByBaseId(base_id);
            if (cid != EClassId{})
            {
                const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
                max_pages = std::max(max_pages, state.bytesToPages(getSizeBytes(cls.shape, cls.dtype), cls.mem_space));
            }
        }
        return max_pages;
    }

    bool allocateCachedOffset(SearchState &state, BaseEClassId base_id, std::vector<VarId> &worklist)
    {
        const auto cache_it = state.cached_vars.find(base_id);
        const auto offsets_it = offsets_by_base_.find(base_id);
        if (cache_it == state.cached_vars.end() || offsets_it == offsets_by_base_.end() || offsets_it->second.empty())
            return true;

        const VarInfo &cache_info = state.var_infos[cache_it->second];
        const uint32_t size_pages = getMaxSizePages(state, base_id);

        int32_t min_offset = 0;
        int32_t max_offset = std::numeric_limits<int32_t>::max();
        int32_t fixed_offset = -1;
        for (VarId offset_var : offsets_it->second)
        {
            const Domain &domain = state.domains[offset_var];
            if (domain.isEmpty())
                return false;
            min_offset = std::max(min_offset, domain.getMin());
            max_offset = std::min(max_offset, domain.getMax());
            if (domain.isFixed())
            {
                if (fixed_offset >= 0 && fixed_offset != domain.fixedValue())
                    return false;
                fixed_offset = domain.fixedValue();
            }
        }

        if (fixed_offset >= 0)
        {
            for (VarId offset_var : offsets_it->second)
            {
                const Domain &domain = state.domains[offset_var];
                if (!domain.contains(fixed_offset))
                    return false;
                if (!domain.isFixed())
                {
                    if (state.setDomain(offset_var, Domain::makeFixed(fixed_offset)))
                        worklist.push_back(offset_var);
                }
            }
            return true;
        }

        // Approach A: Arrival-order sequential packing for cached nodes.
        // Pack cached variables sequentially in the dynamic arena starting at min_p
        // (after preallocated buffers).
        const auto prealloc_it = state.preallocated_pages.find(cache_info.mem_space);
        int64_t candidate = (prealloc_it != state.preallocated_pages.end()) ? prealloc_it->second : 0;
        candidate = std::max<int64_t>(candidate, min_offset);

        for (const auto &other_cache : state.cached_vars)
        {
            if (other_cache.first == base_id)
                continue;
            const Domain &cached_domain = state.domains[other_cache.second];
            if (!cached_domain.isFixed() || cached_domain.fixedValue() != 1)
                continue;

            const auto other_offsets_it = offsets_by_base_.find(other_cache.first);
            if (other_offsets_it == offsets_by_base_.end())
                continue;

            int32_t other_offset = -1;
            for (VarId offset_var : other_offsets_it->second)
            {
                const Domain &domain = state.domains[offset_var];
                if (domain.isFixed())
                {
                    other_offset = domain.fixedValue();
                    break;
                }
            }
            if (other_offset < 0)
                continue;

            const VarInfo &other_info = state.var_infos[other_cache.second];
            if (other_info.mem_space != cache_info.mem_space)
                continue;

            const uint32_t other_size = getMaxSizePages(state, other_cache.first);
            candidate = std::max<int64_t>(candidate, static_cast<int64_t>(other_offset) + other_size);
        }

        if (candidate > max_offset || candidate + size_pages - 1 > std::numeric_limits<int32_t>::max())
            return false;

        uint64_t mem_cap = state.getMemoryCap(cache_info.mem_space);
        uint32_t align = state.getPageAlignment(cache_info.mem_space);
        if (mem_cap > 0 && (static_cast<uint64_t>(candidate + size_pages) * align > mem_cap))
            return false;

        for (VarId offset_var : offsets_it->second)
        {
            const Domain &domain = state.domains[offset_var];
            if (!domain.contains(static_cast<int32_t>(candidate)))
                return false;
        }

        for (VarId offset_var : offsets_it->second)
        {
            if (state.setDomain(offset_var, Domain::makeFixed(static_cast<int32_t>(candidate))))
                worklist.push_back(offset_var);
        }
        return true;
    }

  public:
    std::string name() const override
    {
        return "CachedOffsetAllocationPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        ensureOffsetIndex(state);
        if (changed == kInvalidVarId)
        {
            std::vector<BaseEClassId> active_caches;
            for (const auto &cache_pair : state.cached_vars)
            {
                const Domain &domain = state.domains[cache_pair.second];
                if (domain.isFixed() && domain.fixedValue() == 1)
                    active_caches.push_back(cache_pair.first);
            }
            std::sort(active_caches.begin(), active_caches.end(), [](BaseEClassId a, BaseEClassId b) {
                return a.value < b.value;
            });
            for (BaseEClassId bid : active_caches)
            {
                if (!allocateCachedOffset(state, bid, worklist))
                    return false;
            }
            return true;
        }

        const VarInfo &info = state.var_infos[changed];
        if (info.type == VarType::CACHED)
        {
            const Domain &domain = state.domains[changed];
            return !domain.isFixed() || domain.fixedValue() != 1 ||
                   allocateCachedOffset(state, info.base_eclass_id, worklist);
        }
        if (info.type == VarType::OFFSET)
        {
            const BaseEClassId base_id = state.bucket_egraphs[info.bucket_idx].getEClass(info.eclass_id).base_eclass_id;
            const auto cache_it = state.cached_vars.find(base_id);
            if (cache_it == state.cached_vars.end())
                return true;
            const Domain &cache_domain = state.domains[cache_it->second];
            if (!cache_domain.isFixed() || cache_domain.fixedValue() != 1)
                return true;
            return allocateCachedOffset(state, base_id, worklist);
        }
        return true;
    }
};

// ============================================================================
// EXTRA PROPAGATORS (12 - 15)
// ============================================================================

// 12. Maintain the per-bucket critical path lower bound after selection changes.
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

    void initialize(SearchState &state) const
    {
        state.bucket_critical_path_lower_bounds.assign(state.buckets.size(), 0.0f);
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
            state.bucket_critical_path_lower_bounds[b] = computeBucketCriticalPath(state, b);
        state.critical_path_lower_bound_initialized = true;
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
            state.updateLowerBoundBucket(b);
    }

  public:
    std::string name() const override
    {
        return "CriticalPathPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (state.best_cost == TGConstants::INF)
            return true;
        if (!state.critical_path_lower_bound_initialized)
            initialize(state);
        if (changed != kInvalidVarId && state.var_infos[changed].type != VarType::SELECTED)
            return true;
        if (changed != kInvalidVarId)
        {
            const uint32_t bucket_idx = state.var_infos[changed].bucket_idx;
            if (bucket_idx < state.bucket_critical_path_lower_bounds.size())
            {
                state.bucket_critical_path_lower_bounds[bucket_idx] = computeBucketCriticalPath(state, bucket_idx);
                state.updateLowerBoundBucket(bucket_idx);
            }
        }
        return true;
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

// 15. Maintain cached per-engine workloads and their per-bucket lower bound.
class EngineWorkloadPropagator : public Propagator
{
  public:
    std::string name() const override
    {
        return "EngineWorkloadPropagator";
    }

  private:
    void initialize(SearchState &state) const
    {
        state.engine_work.assign(state.buckets.size(), {});
        state.selected_engine_work.assign(state.numVars(), {});
        state.bucket_engine_work_lower_bounds.assign(state.buckets.size(), 0.0f);
        state.engine_work_initialized = true;
        for (VarId var_id = 0; var_id < state.numVars(); ++var_id)
        {
            if (state.var_infos[var_id].type == VarType::SELECTED)
                updateSelectedWork(state, var_id);
        }
        for (uint32_t b = 0; b < state.buckets.size(); ++b)
        {
            state.bucket_engine_work_lower_bounds[b] = bucketEngineWorkLowerBound(state.engine_work[b]);
            state.updateLowerBoundBucket(b);
        }
    }

  public:
    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (state.best_cost == TGConstants::INF)
            return true;
        if (!state.engine_work_initialized)
            initialize(state);
        if (changed != kInvalidVarId && state.var_infos[changed].type != VarType::SELECTED)
            return true;
        if (changed != kInvalidVarId && state.engine_work_initialized)
        {
            const uint32_t bucket_idx = state.var_infos[changed].bucket_idx;
            updateSelectedWork(state, changed);
            state.bucket_engine_work_lower_bounds[bucket_idx] = bucketEngineWorkLowerBound(state.engine_work[bucket_idx]);
            state.updateLowerBoundBucket(bucket_idx);
        }
        return true;
    }

  private:
    static float bucketEngineWorkLowerBound(const std::unordered_map<Engine, float> &engine_work)
    {
        float bound = 0.0f;
        for (const auto &entry : engine_work)
            bound = std::max(bound, entry.second);
        return bound;
    }

    static void updateSelectedWork(SearchState &state, VarId var_id)
    {
        const uint32_t bucket_idx = state.var_infos[var_id].bucket_idx;
        auto &bucket_work = state.engine_work[bucket_idx];
        auto &cached_work = state.selected_engine_work[var_id];
        for (const auto &entry : cached_work)
        {
            bucket_work[entry.first] -= entry.second;
            if (bucket_work[entry.first] <= 1.0e-6f)
                bucket_work.erase(entry.first);
        }
        cached_work.clear();

        const Domain &sel_dom = state.domains[var_id];
        if (!sel_dom.isFixed() || sel_dom.fixedValue() <= 0)
            return;
        auto selected = state.selected_vars[bucket_idx].find(state.var_infos[var_id].eclass_id);
        if (selected == state.selected_vars[bucket_idx].end())
            return;
        const EClass &cls = state.bucket_egraphs[bucket_idx].getEClass(selected->first);
        const uint32_t en_idx = static_cast<uint32_t>(sel_dom.fixedValue() - 1);
        if (en_idx >= cls.enodes.size())
            return;
        ENodeId en_id = cls.enodes[en_idx];
        const ENode &enode = state.bucket_egraphs[bucket_idx].getENode(en_id);
        const float cost = en_id.value < state.bucket_enode_infos[bucket_idx].size()
                               ? state.bucket_enode_infos[bucket_idx][en_id.value].cost
                               : 0.0f;
        if (cost <= 0.0f || cost >= TGConstants::INF || enode.getOpType() == OpType::INPUT ||
            enode.getOpType() == OpType::CACHE)
            return;
        for (const Engine &engine : enode.getEngines())
        {
            bucket_work[engine] += cost;
            cached_work[engine] += cost;
        }
    }
};

// ============================================================================
// Registration Helpers
// ============================================================================

template <typename EngineT>
inline void addBasePropagators(EngineT &engine, bool fixed_starts_only = false)
{
    engine.addPropagator(std::make_unique<SelectionReachabilityPropagator>());
    engine.addPropagator(std::make_unique<SelectionChildrenPropagator>());
    engine.addPropagator(std::make_unique<UnselectedStartOffsetPropagator>());
    engine.addPropagator(std::make_unique<CacheExclusionPropagator>());
    engine.addPropagator(std::make_unique<StartPrecedencePropagator>(fixed_starts_only));
    engine.addPropagator(std::make_unique<StartUniquePropagator>());
    engine.addPropagator(std::make_unique<FixedOffsetStartPropagator>());
    engine.addPropagator(std::make_unique<WriteAfterReadPropagator>());
    engine.addPropagator(std::make_unique<ViewOffsetPropagator>());
    engine.addPropagator(std::make_unique<PearceKellyCyclePropagator>());
    engine.addPropagator(std::make_unique<ParentRemovalPropagator>());
    engine.addPropagator(std::make_unique<CacheRequirementPropagator>());
    engine.addPropagator(std::make_unique<CachedOffsetPropagator>());
}

template <typename EngineT>
inline void addExtraPropagators(EngineT &engine)
{
    engine.addPropagator(std::make_unique<CriticalPathPropagator>());
    engine.addPropagator(std::make_unique<CacheBudgetPropagator>());
    engine.addPropagator(std::make_unique<EngineWorkloadPropagator>());
    engine.addPropagator(std::make_unique<WriteAfterReadStartPropagator>());
    engine.addPropagator(std::make_unique<CachedOffsetAllocationPropagator>());
}

template <typename EngineT>
inline void addAllPropagators(EngineT &engine, bool fixed_starts_only = false)
{
    addBasePropagators(engine, fixed_starts_only);
    addExtraPropagators(engine);
}

} // namespace plan
