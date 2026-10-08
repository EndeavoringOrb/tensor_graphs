// tensor_graphs_cpp/core/plan/propagators/memory_no_overlap.hpp
#pragma once

#include <chrono>

#include "core/plan/propagators/base.hpp"

namespace plan
{

class WriteAfterReadPropagator : public Propagator
{
#ifdef TG_PROFILE
    uint64_t fixed_offset_profile_ns_ = 0;
    uint64_t ranged_offset_profile_ns_ = 0;
    uint64_t start_profile_ns_ = 0;
    uint64_t initial_profile_ns_ = 0;
#endif

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
        if (!sel_dom.isFixed() && sel_dom.contains(0))
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
        if (st_it == state.start_vars[b].end())
            return false;
        VarId st_v = st_it->second;
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
                    info.first_view_child = EClassId{UINT32_MAX};
                    info.next_view_sibling = EClassId{UINT32_MAX};
                    info.view_tin = UINT32_MAX;
                    info.view_tout = UINT32_MAX;
                    info.base_cid = EClassId{UINT32_MAX};
                    info.start_min = -1;
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

            info.start_min = -1;
            info.start_max = -1;
            auto st_it = state.start_vars[b].find(cid);
            if (st_it != state.start_vars[b].end())
            {
                info.start_min = state.domains[st_it->second].getMin();
                info.start_max = state.domains[st_it->second].getMax();
            }

            info.max_reader_start_max = -1;
            info.readers.clear();
            active_cids.push_back(cid);
            touched_cids.push_back(cid);
            }

            // Active view relations form a forest. Euler intervals make the
            // repeated ancestor checks in overlap propagation constant-time.
            for (EClassId cid : active_cids)
            {
                auto &info = class_info[cid.value];
                const EClassId parent = info.view_parent;
                if (info.is_view && parent.value < class_info.size() &&
                    class_info[parent.value].is_active)
                {
                    info.next_view_sibling = class_info[parent.value].first_view_child;
                    class_info[parent.value].first_view_child = cid;
                }
            }
            std::vector<std::pair<EClassId, bool>> view_stack;
            view_stack.reserve(active_cids.size() * 2);
            uint32_t view_clock = 0;
            for (EClassId root : active_cids)
            {
                const EClassId parent = class_info[root.value].view_parent;
                if (parent.value < class_info.size() && class_info[parent.value].is_active)
                    continue;
                view_stack.emplace_back(root, false);
                while (!view_stack.empty())
                {
                    const auto [cid, exiting] = view_stack.back();
                    view_stack.pop_back();
                    auto &info = class_info[cid.value];
                    if (exiting)
                    {
                        info.view_tout = view_clock++;
                        continue;
                    }
                    info.view_tin = view_clock++;
                    view_stack.emplace_back(cid, true);
                    for (EClassId child = info.first_view_child;
                         child.value < class_info.size();
                         child = class_info[child.value].next_view_sibling)
                        view_stack.emplace_back(child, false);
                }
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
                cand_st_max = state.domains[st_it->second].getMax();

            if (cand.value < class_info.size())
            {
                if (!class_info[cand.value].is_active && class_info[cand.value].start_max == -1)
                {
                    touched_cids.push_back(cand);
                }
                class_info[cand.value].start_min = -1;
                class_info[cand.value].start_max = cand_st_max;
                if (st_it != state.start_vars[b].end())
                    class_info[cand.value].start_min = state.domains[st_it->second].getMin();
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
                    start_max = state.domains[st_it->second].getMax();
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
                                         state.domains[st_it->second].isFixed();
                const bool broad = !start_fixed || info.is_input_or_cache || info.is_root;
                int32_t start = 0;
                if (start_fixed)
                    start = state.domains[st_it->second].fixedValue();
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

    using SpatialIntervalIndex = PropagationState::WriteAfterReadBucket::SpatialIntervalIndex;
    using SpatialIntervalNode = PropagationState::WriteAfterReadBucket::SpatialIntervalNode;
    using FixedOffsetAllocation = PropagationState::WriteAfterReadBucket::FixedOffsetAllocation;

    static bool spatialKeyLess(const FixedOffsetAllocation &a, const FixedOffsetAllocation &b)
    {
        return a.offset != b.offset ? a.offset < b.offset : a.cid.value < b.cid.value;
    }

    static void refreshSpatialNode(SpatialIntervalIndex &index, int32_t node_id)
    {
        auto &node = index.nodes[node_id];
        node.max_end = node.allocation.end;
        if (node.left >= 0)
            node.max_end = std::max(node.max_end, index.nodes[node.left].max_end);
        if (node.right >= 0)
            node.max_end = std::max(node.max_end, index.nodes[node.right].max_end);
    }

    static uint64_t spatialPriority(const FixedOffsetAllocation &allocation)
    {
        uint64_t value = (static_cast<uint64_t>(allocation.offset) << 32) | allocation.cid.value;
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    static int32_t rotateSpatialRight(SpatialIntervalIndex &index, int32_t root_id)
    {
        const int32_t new_root = index.nodes[root_id].left;
        index.nodes[root_id].left = index.nodes[new_root].right;
        index.nodes[new_root].right = root_id;
        refreshSpatialNode(index, root_id);
        refreshSpatialNode(index, new_root);
        return new_root;
    }

    static int32_t rotateSpatialLeft(SpatialIntervalIndex &index, int32_t root_id)
    {
        const int32_t new_root = index.nodes[root_id].right;
        index.nodes[root_id].right = index.nodes[new_root].left;
        index.nodes[new_root].left = root_id;
        refreshSpatialNode(index, root_id);
        refreshSpatialNode(index, new_root);
        return new_root;
    }

    static int32_t mergeSpatialNodes(SpatialIntervalIndex &index, int32_t left_id, int32_t right_id)
    {
        if (left_id < 0)
            return right_id;
        if (right_id < 0)
            return left_id;
        if (index.nodes[left_id].priority > index.nodes[right_id].priority)
        {
            index.nodes[left_id].right = mergeSpatialNodes(index, index.nodes[left_id].right, right_id);
            refreshSpatialNode(index, left_id);
            return left_id;
        }
        index.nodes[right_id].left = mergeSpatialNodes(index, left_id, index.nodes[right_id].left);
        refreshSpatialNode(index, right_id);
        return right_id;
    }

    static int32_t insertSpatialNode(SpatialIntervalIndex &index, int32_t root_id, int32_t node_id)
    {
        if (root_id < 0)
            return node_id;
        if (spatialKeyLess(index.nodes[node_id].allocation, index.nodes[root_id].allocation))
        {
            index.nodes[root_id].left = insertSpatialNode(index, index.nodes[root_id].left, node_id);
            if (index.nodes[index.nodes[root_id].left].priority > index.nodes[root_id].priority)
                root_id = rotateSpatialRight(index, root_id);
        }
        else
        {
            index.nodes[root_id].right = insertSpatialNode(index, index.nodes[root_id].right, node_id);
            if (index.nodes[index.nodes[root_id].right].priority > index.nodes[root_id].priority)
                root_id = rotateSpatialLeft(index, root_id);
        }
        refreshSpatialNode(index, root_id);
        return root_id;
    }

    static int32_t eraseSpatialNode(SpatialIntervalIndex &index, int32_t root_id,
                                    const FixedOffsetAllocation &allocation, bool &erased)
    {
        if (root_id < 0)
            return -1;
        const auto &root_allocation = index.nodes[root_id].allocation;
        if (!spatialKeyLess(allocation, root_allocation) && !spatialKeyLess(root_allocation, allocation))
        {
            const int32_t merged = mergeSpatialNodes(index, index.nodes[root_id].left,
                                                      index.nodes[root_id].right);
            index.free_nodes.push_back(root_id);
            erased = true;
            return merged;
        }
        if (spatialKeyLess(allocation, root_allocation))
            index.nodes[root_id].left = eraseSpatialNode(index, index.nodes[root_id].left, allocation, erased);
        else
            index.nodes[root_id].right = eraseSpatialNode(index, index.nodes[root_id].right, allocation, erased);
        refreshSpatialNode(index, root_id);
        return root_id;
    }

    static void addSpatialAllocation(SpatialIntervalIndex &index, const FixedOffsetAllocation &allocation)
    {
        int32_t node_id;
        if (index.free_nodes.empty())
        {
            node_id = static_cast<int32_t>(index.nodes.size());
            index.nodes.emplace_back();
        }
        else
        {
            node_id = index.free_nodes.back();
            index.free_nodes.pop_back();
        }
        auto &node = index.nodes[node_id];
        node.allocation = allocation;
        node.max_end = allocation.end;
        node.priority = spatialPriority(allocation);
        node.left = -1;
        node.right = -1;
        index.root = insertSpatialNode(index, index.root, node_id);
    }

    static void removeSpatialAllocation(SpatialIntervalIndex &index,
                                        const FixedOffsetAllocation &allocation)
    {
        bool erased = false;
        index.root = eraseSpatialNode(index, index.root, allocation, erased);
    }

    template <typename Callback>
    static bool visitSpatialOverlaps(const SpatialIntervalIndex &index, int32_t node_id,
                                     uint64_t low, uint64_t high, Callback &callback)
    {
        if (node_id < 0 || index.nodes[node_id].max_end <= low)
            return true;
        const auto &node = index.nodes[node_id];
        if (node.left >= 0 && index.nodes[node.left].max_end > low &&
            !visitSpatialOverlaps(index, node.left, low, high, callback))
            return false;
        if (node.allocation.offset < high && node.allocation.end > low && !callback(node.allocation))
            return false;
        if (node.allocation.offset < high && node.right >= 0 &&
            !visitSpatialOverlaps(index, node.right, low, high, callback))
            return false;
        return true;
    }

    static void buildFixedOffsetIndex(const SearchState &state, uint32_t b, VarId changed)
    {
        auto &schedule = state.propagation.write_after_read[b];
        const bool rebuild = !schedule.fixed_offset_index_initialized ||
                             schedule.fixed_offset_structure_dirty;
        if (!rebuild && changed < state.var_infos.size() &&
            state.var_infos[changed].type == VarType::OFFSET)
        {
            auto old_key = schedule.fixed_offset_keys_by_var.find(changed);
            if (old_key != schedule.fixed_offset_keys_by_var.end())
            {
                auto old_space = schedule.fixed_offset_allocations.find(old_key->second.first);
                if (old_space != schedule.fixed_offset_allocations.end())
                {
                    const auto old_allocation = old_space->second.find(old_key->second.second);
                    if (old_allocation != old_space->second.end())
                    {
                        auto spatial_it = schedule.fixed_offset_spatial_index.find(old_key->second.first);
                        if (spatial_it != schedule.fixed_offset_spatial_index.end())
                            removeSpatialAllocation(spatial_it->second, old_allocation->second);
                    }
                    old_space->second.erase(old_key->second.second);
                }
                schedule.fixed_offset_keys_by_var.erase(old_key);
            }

            const VarInfo &info = state.var_infos[changed];
            FixedAlloc alloc;
            if (getAlloc(state, info.bucket_idx, info.eclass_id, alloc,
                         /*require_fixed_offset=*/true, /*require_fixed_start=*/false))
            {
                const std::pair<uint32_t, uint32_t> key{alloc.offset, alloc.cid.value};
                auto &entries = schedule.fixed_offset_allocations[alloc.mem_space];
                entries.emplace(key, FixedOffsetAllocation{
                                         alloc.cid, alloc.mem_space, alloc.offset,
                                         static_cast<uint64_t>(alloc.offset) + alloc.size,
                                         alloc.size, alloc.en_idx, alloc.en_id, alloc.start_var,
                                         alloc.offset_var, alloc.is_view,
                                         alloc.is_input_or_cache, alloc.is_root});
                schedule.fixed_offset_keys_by_var[changed] = {alloc.mem_space, key};
                auto &max_size = schedule.max_fixed_allocation_size[alloc.mem_space];
                max_size = std::max(max_size, alloc.size);
                const auto &fixed_allocation = entries.at(key);
                addSpatialAllocation(schedule.fixed_offset_spatial_index[alloc.mem_space], fixed_allocation);
            }
            schedule.spatial_dirty = false;
            return;
        }

        for (auto &space_entries : schedule.fixed_offset_allocations)
            space_entries.second.clear();
        schedule.fixed_offset_keys_by_var.clear();
        schedule.max_fixed_allocation_size.clear();
        schedule.fixed_offset_spatial_index.clear();
        for (const auto &pair : state.selected_vars[b])
        {
            FixedAlloc alloc;
            if (!getAlloc(state, b, pair.first, alloc,
                          /*require_fixed_offset=*/true, /*require_fixed_start=*/false))
                continue;

            const std::pair<uint32_t, uint32_t> key{alloc.offset, alloc.cid.value};
            auto &entries = schedule.fixed_offset_allocations[alloc.mem_space];
            entries.emplace(key, FixedOffsetAllocation{
                                     alloc.cid, alloc.mem_space, alloc.offset,
                                     static_cast<uint64_t>(alloc.offset) + alloc.size,
                                     alloc.size, alloc.en_idx, alloc.en_id, alloc.start_var,
                                     alloc.offset_var, alloc.is_view,
                                     alloc.is_input_or_cache, alloc.is_root});
            schedule.fixed_offset_keys_by_var[alloc.offset_var] = {alloc.mem_space, key};
            auto &max_size = schedule.max_fixed_allocation_size[alloc.mem_space];
            max_size = std::max(max_size, alloc.size);
            addSpatialAllocation(schedule.fixed_offset_spatial_index[alloc.mem_space], entries.at(key));
        }

        schedule.fixed_offset_index_initialized = true;
        schedule.fixed_offset_structure_dirty = false;
        schedule.spatial_dirty = false;
    }

    static bool mayOverlapInTime(const SearchState &state, uint32_t b, EClassId a_cid, EClassId c_cid)
    {
        const auto &class_info = state.propagation.write_after_read[b].class_info;
        if (a_cid.value >= class_info.size() || c_cid.value >= class_info.size())
            return true;
        const auto &a_info = class_info[a_cid.value];
        const auto &c_info = class_info[c_cid.value];
        const bool a_fixed = a_info.start_min >= 0 && a_info.start_min == a_info.start_max;
        const bool c_fixed = c_info.start_min >= 0 && c_info.start_min == c_info.start_max;
        if (!a_fixed || !c_fixed || a_info.is_input_or_cache || a_info.is_root ||
            c_info.is_input_or_cache || c_info.is_root)
            return true;

        const int32_t a_start = a_info.start_min;
        const int32_t c_start = c_info.start_min;
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

    static bool isActiveViewOf(const SearchState &state, uint32_t b,
                               EClassId base, EClassId view_cand)
    {
        if (base == view_cand)
            return true;
        const auto &class_info = state.propagation.write_after_read[b].class_info;
        if (base.value < class_info.size() && view_cand.value < class_info.size() &&
            class_info[base.value].is_active && class_info[view_cand.value].is_active)
        {
            const auto &base_info = class_info[base.value];
            const auto &view_info = class_info[view_cand.value];
            return base_info.view_tin <= view_info.view_tin &&
                   view_info.view_tout <= base_info.view_tout;
        }
        return isViewOf(state, b, base, view_cand);
    }

    static bool allActiveAlternativesRead(const SearchState &state, uint32_t b, EClassId producer,
                                          EClassId consumer)
    {
        const auto selection_it = state.selected_vars[b].find(consumer);
        if (selection_it == state.selected_vars[b].end())
            return false;
        const Domain &selection = state.domains[selection_it->second];
        if (selection.contains(0))
            return false;

        const EClass &cls = state.bucket_egraphs[b].getEClass(consumer);
        bool found_active_alternative = false;
        for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
        {
            if (!selection.contains(static_cast<int32_t>(en_idx + 1)))
                continue;
            found_active_alternative = true;
            const ENode &enode = state.bucket_egraphs[b].getENode(cls.enodes[en_idx]);
            bool reads_producer = false;
            for (EClassId child : enode.getChildren())
            {
                const EClassId child_cid = state.bucket_egraphs[b].findConst(child);
                if (child_cid == producer || isViewOf(state, b, producer, child_cid))
                {
                    reads_producer = true;
                    break;
                }
            }
            if (!reads_producer)
                return false;
        }
        return found_active_alternative;
    }

  private:
    bool canShare(const SearchState &state, FixedAlloc A, FixedAlloc B) const
    {
        const auto &class_info = state.propagation.write_after_read[A.bucket_idx].class_info;
        // - if both A and B are views, return true. ignore
        if (A.is_view && B.is_view)
            return true;

        bool b_is_view_of_a = isActiveViewOf(state, A.bucket_idx, A.cid, B.cid);
        bool a_is_view_of_b = isActiveViewOf(state, A.bucket_idx, B.cid, A.cid);
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
            if (!allActiveAlternativesRead(state, A.bucket_idx, A.cid, c_cid))
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
            if (!allActiveAlternativesRead(state, A.bucket_idx, A.cid, c_cid))
                continue;
            auto c_sel_it = state.selected_vars[A.bucket_idx].find(c_cid);
            if (c_sel_it == state.selected_vars[A.bucket_idx].end())
                continue;
            const Domain &c_sel_dom = state.domains[c_sel_it->second];
            if (c_sel_dom.contains(0))
                continue;

            auto c_st_it = state.start_vars[A.bucket_idx].find(c_cid);
            if (c_st_it == state.start_vars[A.bucket_idx].end())
                continue;

            for (uint32_t c_en = 0; c_en < state.bucket_egraphs[A.bucket_idx].getEClass(c_cid).enodes.size(); ++c_en)
            {
                if (!c_sel_dom.contains(c_en + 1))
                    continue;
                VarId c_st_v = c_st_it->second;
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
#ifdef TG_PROFILE
        uint64_t *profile_ns = &initial_profile_ns_;
        if (changed != kInvalidVarId && changed < state.var_infos.size())
        {
            const VarInfo &info = state.var_infos[changed];
            if (info.type == VarType::OFFSET)
                profile_ns = state.domains[changed].isFixed()
                                 ? &fixed_offset_profile_ns_
                                 : &ranged_offset_profile_ns_;
            else if (info.type == VarType::START)
                profile_ns = &start_profile_ns_;
        }
        struct ProfileScope
        {
            uint64_t &total_ns;
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            ~ProfileScope()
            {
                total_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count());
            }
        } profile_scope{*profile_ns};
#endif
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

            if (curr_offset_fixed)
            {
                auto spatial_it = state.propagation.write_after_read[b].fixed_offset_spatial_index.find(curr.mem_space);
                if (spatial_it != state.propagation.write_after_read[b].fixed_offset_spatial_index.end())
                {
                    const uint64_t curr_end = static_cast<uint64_t>(curr.offset) + curr.size;
                    bool valid = true;
                    auto processCandidate = [&](const FixedOffsetAllocation &indexed_other) {
                        const EClassId other_cid = indexed_other.cid;
                        if (other_cid == cid || !mayOverlapInTime(state, b, cid, other_cid))
                            return true;
                        FixedAlloc other;
                        other.cid = indexed_other.cid;
                        other.bucket_idx = b;
                        other.mem_space = indexed_other.mem_space;
                        other.offset = indexed_other.offset;
                        other.size = indexed_other.size;
                        other.start = state.domains[indexed_other.start_var].getMin();
                        other.en_idx = indexed_other.en_idx;
                        other.en_id = indexed_other.en_id;
                        other.start_var = indexed_other.start_var;
                        other.offset_var = indexed_other.offset_var;
                        other.is_view = indexed_other.is_view;
                        other.is_input_or_cache = indexed_other.is_input_or_cache;
                        other.is_root = indexed_other.is_root;

                        bool pushed = false;
                        if (!canShare(state, curr, other))
                        {
                            if (!enforceDisjoint(state, other, curr, worklist, pushed))
                            {
                                valid = false;
                                return false;
                            }
                        }
                        else if (!checkPair(state, curr, other, worklist))
                        {
                            valid = false;
                            return false;
                        }
                        return true;
                    };
                    visitSpatialOverlaps(spatial_it->second, spatial_it->second.root,
                                         curr.offset, curr_end, processCandidate);
                    if (!valid)
                        return false;
                }
            }
            else
            {
                bool pushed = true;
                while (pushed)
                {
                    pushed = false;
                    auto spatial_it = state.propagation.write_after_read[b].fixed_offset_spatial_index.find(curr.mem_space);
                    if (spatial_it == state.propagation.write_after_read[b].fixed_offset_spatial_index.end())
                        break;
                    const Domain &curr_offset_domain = state.domains[curr.offset_var];
                    const uint64_t domain_min = static_cast<uint64_t>(curr_offset_domain.getMin());
                    const uint64_t domain_end = static_cast<uint64_t>(curr_offset_domain.getMax()) + curr.size;
                    bool valid = true;
                    auto processCandidate = [&](const FixedOffsetAllocation &indexed_other) {
                        const EClassId other_cid = indexed_other.cid;
                        if (other_cid == cid || !mayOverlapInTime(state, b, cid, other_cid))
                            return true;
                        FixedAlloc other;
                        other.cid = indexed_other.cid;
                        other.bucket_idx = b;
                        other.mem_space = indexed_other.mem_space;
                        other.offset = indexed_other.offset;
                        other.size = indexed_other.size;
                        other.start = state.domains[indexed_other.start_var].getMin();
                        other.en_idx = indexed_other.en_idx;
                        other.en_id = indexed_other.en_id;
                        other.start_var = indexed_other.start_var;
                        other.offset_var = indexed_other.offset_var;
                        other.is_view = indexed_other.is_view;
                        other.is_input_or_cache = indexed_other.is_input_or_cache;
                        other.is_root = indexed_other.is_root;

                        const int64_t before_threshold = static_cast<int64_t>(other.offset) - curr.size;
                        const uint64_t after_threshold = static_cast<uint64_t>(other.offset) + other.size;
                        if (domain_min >= after_threshold || curr_offset_domain.getMax() <= before_threshold)
                            return true;

                        if (!canShare(state, curr, other))
                        {
                            if (!enforceDisjoint(state, other, curr, worklist, pushed))
                            {
                                valid = false;
                                return false;
                            }
                            if (pushed)
                                return false;
                        }
                        return true;
                    };
                    visitSpatialOverlaps(spatial_it->second, spatial_it->second.root,
                                         domain_min, domain_end, processCandidate);
                    if (!valid)
                        return false;
                }
            }

            // Unfixed offsets are not forward-pruned here. The brancher chooses
            // each later offset against the fixed allocations, and this
            // propagator checks the pair when that offset becomes fixed.
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

#ifdef TG_PROFILE
    uint64_t fixedOffsetProfileNs() const { return fixed_offset_profile_ns_; }
    uint64_t rangedOffsetProfileNs() const { return ranged_offset_profile_ns_; }
    uint64_t startProfileNs() const { return start_profile_ns_; }
    uint64_t initialProfileNs() const { return initial_profile_ns_; }
#endif
};

// FixedOffsetStartPropagator: see docs/core/propagators.md.
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
            LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [persistent overlap]: A(cid="
                       << A.cid.value << ", in_cache=" << A.is_input_or_cache << ", root=" << A.is_root
                       << ", st=" << state.domains[A.start_var].toString() << ", off=" << A.offset << ", sz=" << A.size
                       << ") and B(cid=" << B.cid.value << ", in_cache=" << B.is_input_or_cache << ", root=" << B.is_root
                       << ", st=" << state.domains[B.start_var].toString() << ", off=" << B.offset << ", sz=" << B.size << ")";
            return false;
        }

        const auto &class_info = state.propagation.write_after_read[b].class_info;
        auto reads = [&](EClassId producer, EClassId consumer) {
            return WriteAfterReadPropagator::allActiveAlternativesRead(state, b, producer, consumer);
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
                LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [unsafe inplace before starts fixed] for A(cid="
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
                LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [same start]: identical start=" << A.start
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
                LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [B reads A: unsafe inplace] for A(cid="
                           << A.cid.value << ") and B(cid=" << B.cid.value << ")";
                return false;
            }

            int32_t min_b_start = A.start + 1;
            for (EClassId c_cid : readers_A)
            {
                if (c_cid == B.cid)
                    continue;
                if (!WriteAfterReadPropagator::allActiveAlternativesRead(state, b, A.cid, c_cid))
                    continue;
                auto c_sel_it = state.selected_vars[b].find(c_cid);
                if (c_sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &c_sel_dom = state.domains[c_sel_it->second];
                if (c_sel_dom.contains(0))
                    continue;

                auto c_st_it = state.start_vars[b].find(c_cid);
                if (c_st_it == state.start_vars[b].end())
                    continue;

                for (uint32_t c_en = 0; c_en < state.bucket_egraphs[b].getEClass(c_cid).enodes.size(); ++c_en)
                {
                    if (!c_sel_dom.contains(c_en + 1))
                        continue;
                    VarId c_st_v = c_st_it->second;
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
                                LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [B reads A: reader empty]: C(cid="
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
                    LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [B reads A: B start empty]: B(cid="
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
                LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [A reads B: unsafe inplace] for A(cid="
                           << A.cid.value << ") and B(cid=" << B.cid.value << ")";
                return false;
            }

            Domain b_st_dom = state.domains[B.start_var];
            if (b_st_dom.setMax(A.start - 1))
            {
                if (b_st_dom.isEmpty())
                {
                    LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [A reads B: B start empty]: B(cid="
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
                if (!WriteAfterReadPropagator::allActiveAlternativesRead(state, b, B.cid, c_cid))
                    continue;
                auto c_sel_it = state.selected_vars[b].find(c_cid);
                if (c_sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &c_sel_dom = state.domains[c_sel_it->second];
                if (c_sel_dom.contains(0))
                    continue;

                auto c_st_it = state.start_vars[b].find(c_cid);
                if (c_st_it == state.start_vars[b].end())
                    continue;

                for (uint32_t c_en = 0; c_en < state.bucket_egraphs[b].getEClass(c_cid).enodes.size(); ++c_en)
                {
                    if (!c_sel_dom.contains(c_en + 1))
                        continue;
                    VarId c_st_v = c_st_it->second;
                    Domain c_st_dom = state.domains[c_st_v];
                    if (c_st_dom.setMax(A.start - 1))
                    {
                        if (c_st_dom.isEmpty())
                        {
                            LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [A reads B: reader empty]: C(cid="
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
            if (!WriteAfterReadPropagator::allActiveAlternativesRead(state, b, A.cid, c_cid))
                continue;
            auto c_sel_it = state.selected_vars[b].find(c_cid);
            if (c_sel_it == state.selected_vars[b].end())
                continue;
            const Domain &c_sel_dom = state.domains[c_sel_it->second];
            if (c_sel_dom.contains(0))
                continue;

            auto c_st_it = state.start_vars[b].find(c_cid);
            if (c_st_it == state.start_vars[b].end())
                continue;

            for (uint32_t c_en = 0; c_en < state.bucket_egraphs[b].getEClass(c_cid).enodes.size(); ++c_en)
            {
                if (c_sel_dom.contains(c_en + 1))
                {
                    VarId c_st_v = c_st_it->second;
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
                if (!WriteAfterReadPropagator::allActiveAlternativesRead(state, b, B.cid, d_cid))
                    continue;
                auto d_sel_it = state.selected_vars[b].find(d_cid);
                if (d_sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &d_sel_dom = state.domains[d_sel_it->second];
                if (d_sel_dom.contains(0))
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

                for (uint32_t d_en = 0; d_en < state.bucket_egraphs[b].getEClass(d_cid).enodes.size(); ++d_en)
                {
                    if (d_sel_dom.contains(d_en + 1))
                    {
                        if (state.domains[d_st_it->second].getMin() >= A.start)
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
            LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [disjoint lifespans impossible]: A(cid="
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
                    LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [disjoint: B setMin empty]: B(cid="
                               << B.cid.value << ") domain empty after setMin(" << t_after_a << ")";
                    return false;
                }
                state.setDomain(B.start_var, b_st_dom);
                worklist.push_back(B.start_var);
            }
            for (EClassId c_cid : readers_A)
            {
                if (!WriteAfterReadPropagator::allActiveAlternativesRead(state, b, A.cid, c_cid))
                    continue;
                auto c_sel_it = state.selected_vars[b].find(c_cid);
                if (c_sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &c_sel_dom = state.domains[c_sel_it->second];
                if (c_sel_dom.contains(0))
                    continue;

                auto c_st_it = state.start_vars[b].find(c_cid);
                if (c_st_it == state.start_vars[b].end())
                    continue;

                for (uint32_t c_en = 0; c_en < state.bucket_egraphs[b].getEClass(c_cid).enodes.size(); ++c_en)
                {
                    if (!c_sel_dom.contains(c_en + 1))
                        continue;
                    VarId c_st_v = c_st_it->second;
                    Domain c_st_dom = state.domains[c_st_v];
                    if (c_st_dom.setMax(b_st_dom.getMax() - 1))
                    {
                        if (c_st_dom.isEmpty())
                        {
                            LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [disjoint: reader C empty]: C(cid="
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
                    LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [disjoint: B setMax empty]: B(cid="
                               << B.cid.value << ") domain empty after setMax(" << (A.start - 1) << ")";
                    return false;
                }
                state.setDomain(B.start_var, b_st_dom);
                worklist.push_back(B.start_var);
            }
            for (EClassId d_cid : readers_B)
            {
                if (!WriteAfterReadPropagator::allActiveAlternativesRead(state, b, B.cid, d_cid))
                    continue;
                auto d_sel_it = state.selected_vars[b].find(d_cid);
                if (d_sel_it == state.selected_vars[b].end())
                    continue;
                const Domain &d_sel_dom = state.domains[d_sel_it->second];
                if (d_sel_dom.contains(0))
                    continue;

                auto d_st_it = state.start_vars[b].find(d_cid);
                if (d_st_it == state.start_vars[b].end())
                    continue;

                for (uint32_t d_en = 0; d_en < state.bucket_egraphs[b].getEClass(d_cid).enodes.size(); ++d_en)
                {
                    if (!d_sel_dom.contains(d_en + 1))
                        continue;
                    VarId d_st_v = d_st_it->second;
                    Domain d_st_dom = state.domains[d_st_v];
                    if (d_st_dom.setMax(A.start - 1))
                    {
                        if (d_st_dom.isEmpty())
                        {
                            LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [disjoint: reader D empty]: D(cid="
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
                    LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Conflict [disjoint: mask empty]: B(cid="
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

            uint32_t b = info.bucket_idx;
            EClassId cid = info.eclass_id;
            if (info.type == VarType::OFFSET ||
                !state.propagation.write_after_read[b].fixed_offset_index_initialized ||
                state.propagation.write_after_read[b].fixed_offset_structure_dirty)
                WriteAfterReadPropagator::buildFixedOffsetIndex(state, b, changed);

            // If offset changed, it must be fixed to establish memory footprint
            if (info.type == VarType::OFFSET && !state.domains[changed].isFixed())
                return true;

            // Eclass must have fixed offset to establish memory footprint
            auto off_it = state.offset_vars[b].find(cid);
            if (off_it == state.offset_vars[b].end() || !state.domains[off_it->second].isFixed())
                return true;

            FixedAlloc curr;
            if (!WriteAfterReadPropagator::getAlloc(state, b, cid, curr, /*require_fixed_offset=*/true, /*require_fixed_start=*/false))
                return true;

            WriteAfterReadPropagator::buildBucketInfo(state, b);
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
            auto candidate_it = candidates.lower_bound(
                {static_cast<uint32_t>(first_possible_offset), 0});

            bool curr_start_fixed = state.domains[curr.start_var].isFixed();
            for (; candidate_it != candidates.end() && candidate_it->first.first < curr_end; ++candidate_it)
            {
                const auto &candidate = candidate_it->second;
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
                    LOG_HOT_PATH(DEBUG) << "[FixedOffsetStartPropagator] Pruned left branch cid=" << cid.value
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


// MemoryNoOverlapPropagator: see docs/core/propagators.md.
class MemoryNoOverlapPropagator : public Propagator
{
    FixedOffsetStartPropagator fixed_offset_prop_;
    WriteAfterReadPropagator write_after_read_prop_;
#ifdef TG_PROFILE
    uint64_t fixed_offset_ns_ = 0;
    uint64_t write_after_read_ns_ = 0;
#endif

  public:
    uint8_t interestedVarTypes() const override
    {
        return varTypeMask(VarType::START) | varTypeMask(VarType::OFFSET);
    }

    StartSelectionGuard startSelectionGuard() const override
    {
        return StartSelectionGuard::FIXED_OFFSET_FOR_START;
    }

    std::string name() const override
    {
        return "MemoryNoOverlapPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            const VarInfo &info = state.var_infos[changed];
            if (info.type != VarType::OFFSET && info.type != VarType::START)
                return true;
            if (info.type == VarType::START)
            {
                auto offset_it = state.offset_vars[info.bucket_idx].find(info.eclass_id);
                if (offset_it == state.offset_vars[info.bucket_idx].end() ||
                    !state.domains[offset_it->second].isFixed())
                return true;
            }
        }
#ifdef TG_PROFILE
        auto fixed_start = std::chrono::steady_clock::now();
#endif
        if (!fixed_offset_prop_.propagate(state, changed, worklist))
        {
#ifdef TG_PROFILE
            fixed_offset_ns_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - fixed_start).count());
#endif
            return false;
        }
#ifdef TG_PROFILE
        fixed_offset_ns_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - fixed_start).count());
        auto write_after_read_start = std::chrono::steady_clock::now();
#endif
        if (!write_after_read_prop_.propagate(state, changed, worklist))
        {
#ifdef TG_PROFILE
            write_after_read_ns_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - write_after_read_start).count());
#endif
            return false;
        }
#ifdef TG_PROFILE
        write_after_read_ns_ += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - write_after_read_start).count());
#endif
        return true;
    }

#ifdef TG_PROFILE
    uint64_t fixedOffsetProfileNs() const { return fixed_offset_ns_; }
    uint64_t writeAfterReadProfileNs() const { return write_after_read_ns_; }
    uint64_t writeAfterReadFixedOffsetProfileNs() const { return write_after_read_prop_.fixedOffsetProfileNs(); }
    uint64_t writeAfterReadRangedOffsetProfileNs() const { return write_after_read_prop_.rangedOffsetProfileNs(); }
    uint64_t writeAfterReadStartProfileNs() const { return write_after_read_prop_.startProfileNs(); }
    uint64_t writeAfterReadInitialProfileNs() const { return write_after_read_prop_.initialProfileNs(); }
#endif
};

} // namespace plan
