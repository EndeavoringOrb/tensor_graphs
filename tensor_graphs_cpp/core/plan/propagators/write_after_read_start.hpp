// tensor_graphs_cpp/core/plan/propagators/write_after_read_start.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

class WriteAfterReadStartPropagator : public Propagator
{
    mutable std::string conflict_reason_;
    struct ReaderIndex
    {
        uint64_t selection_revision = UINT64_MAX;
        std::vector<std::vector<EClassId>> readers_by_value;
    };
    // Graph structure is immutable during search. Starts, offsets, and cache
    // decisions do not change readers; selection updates and undo both do.
    mutable const SearchState *reader_state_ = nullptr;
    mutable std::vector<ReaderIndex> reader_indexes_;
#ifdef TG_PROFILE
    mutable uint64_t allocation_ns_ = 0, reader_index_ns_ = 0, overlap_ns_ = 0;
    mutable uint64_t reader_rebuilds_ = 0, overlap_candidates_ = 0, ordered_pairs_ = 0;
    mutable uint64_t reader_cache_hits_ = 0, reader_index_skips_ = 0;
#endif
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
#ifdef TG_PROFILE
        auto phase_start = std::chrono::steady_clock::now();
        auto finish_phase = [&](uint64_t &total) {
            const auto now = std::chrono::steady_clock::now();
            total += std::chrono::duration_cast<std::chrono::nanoseconds>(now - phase_start).count();
            phase_start = now;
        };
#endif
        const auto &fixed_offset_vars = state.getFixedOffsetVars(b);
        if (fixed_offset_vars.size() < 2)
            return true;

        std::vector<ActiveAllocation> active;
        active.reserve(fixed_offset_vars.size());

        for (const auto &pair : fixed_offset_vars)
        {
            const VarId offset_var = pair.second;
            const VarInfo &off_info = state.var_infos[offset_var];
            const EClassId cid = off_info.eclass_id;

            const auto selected_it = state.selected_vars[b].find(cid);
            if (selected_it == state.selected_vars[b].end())
                continue;
            const Domain &selection = state.domains[selected_it->second];
            if (!selection.isFixed() || selection.fixedValue() <= 0)
                continue;
            const uint32_t en_idx = static_cast<uint32_t>(selection.fixedValue() - 1);
            const EClass &cls = state.bucket_egraphs[b].getEClass(cid);
            if (en_idx >= cls.enodes.size())
                continue;

            const auto start_it = state.start_vars[b].find(cid);
            if (start_it == state.start_vars[b].end())
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
                                       (cls.base_eclass_id != BaseEClassId{} &&
                                        (state.preallocated_buffers.count(cls.base_eclass_id) != 0 || state.isBaseFixedCached(cls.base_eclass_id)));
            const uint32_t size_pages = std::max<uint32_t>(
                1, state.bytesToPages(getStorageSpanBytes(enode.getShape(), enode.getStrides(), enode.getDType()), cls.mem_space));

            active.push_back({cid, cls.base_eclass_id, cls.mem_space, pair.first,
                              size_pages, start_it->second,
                              state.domains[start_it->second], is_view, is_persistent});
        }

        if (active.size() < 2)
            return true;

#ifdef TG_PROFILE
        finish_phase(allocation_ns_);
#endif
        const std::vector<std::vector<EClassId>> *readers_by_value = nullptr;
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
#ifdef TG_PROFILE
                    ++overlap_candidates_;
#endif
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

#ifdef TG_PROFILE
                    ++ordered_pairs_;
#endif
                    const ActiveAllocation &earlier = first_before_second ? first : second;
                    const ActiveAllocation &later = first_before_second ? second : first;
                    if (readers_by_value == nullptr)
                    {
#ifdef TG_PROFILE
                        finish_phase(overlap_ns_);
#endif
                        readers_by_value = &getReaderIndex(state, b);
#ifdef TG_PROFILE
                        finish_phase(reader_index_ns_);
#endif
                    }
                    const auto &readers = readers_by_value->at(earlier.cid.value);
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
                        earliest_last_reader = std::max(
                            earliest_last_reader,
                            state.domains[reader_start_it->second].getMin());
                    }
                    if (later_reads_earlier || earliest_last_reader < 0)
                        continue;

                    Domain later_start = state.domains[later.start_var];
                    const int32_t required_start = earliest_last_reader + 1;
                    if (later_start.setMin(required_start))
                    {
                        if (later_start.isEmpty())
                        {
                            conflict_reason_ = "bucket=" + std::to_string(b) + " earlier=" +
                                state.var_infos[earlier.start_var].name + state.domains[earlier.start_var].toString() +
                                " later=" + state.var_infos[later.start_var].name + state.domains[later.start_var].toString() +
                                " overlap-offsets=" + std::to_string(earlier.offset) + "/" + std::to_string(later.offset) +
                                " last-reader-min=" + std::to_string(earliest_last_reader) + " required-min=" + std::to_string(required_start);
#ifdef TG_PROFILE
                            finish_phase(overlap_ns_);
#endif
                            return false;
                        }
                        state.setDomain(later.start_var, later_start);
                        worklist.push_back(later.start_var);
                    }
                }
            }
        }
#ifdef TG_PROFILE
        finish_phase(overlap_ns_);
        reader_index_skips_ += readers_by_value == nullptr;
#endif
        return true;
    }

    const std::vector<std::vector<EClassId>> &getReaderIndex(const SearchState &state, uint32_t b) const
    {
        if (reader_state_ != &state)
        {
            reader_indexes_.clear();
            reader_state_ = &state;
        }
        reader_indexes_.resize(state.buckets.size());
        ReaderIndex &index = reader_indexes_.at(b);
        const uint64_t revision = state.getSelectionRevision();
        if (index.selection_revision == revision)
        {
#ifdef TG_PROFILE
            ++reader_cache_hits_;
#endif
            return index.readers_by_value;
        }

        index.readers_by_value.assign(state.bucket_egraphs[b].classes.size(), {});
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
                addReaderAncestors(state, b, child, reader, index.readers_by_value);
        }
        index.selection_revision = revision;
#ifdef TG_PROFILE
        ++reader_rebuilds_;
#endif
        return index.readers_by_value;
    }

  public:
    std::string conflictReason() const override { return conflict_reason_; }
    std::string diagnosticSummary() const override
    {
#ifdef TG_PROFILE
        return "allocation=" + std::to_string(allocation_ns_ / 1.0e9) + "s reader-index=" +
            std::to_string(reader_index_ns_ / 1.0e9) + "s overlap=" + std::to_string(overlap_ns_ / 1.0e9) +
            "s reader-rebuilds=" + std::to_string(reader_rebuilds_) + " reader-cache-hits=" +
            std::to_string(reader_cache_hits_) + " reader-index-skips=" + std::to_string(reader_index_skips_) + " overlap-candidates=" + std::to_string(overlap_candidates_) +
            " ordered-pairs=" + std::to_string(ordered_pairs_);
#else
        return {};
#endif
    }

    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::START); }
    StartSelectionGuard startSelectionGuard() const override
    {
        return StartSelectionGuard::FIXED_START_NON_OPTIONAL;
    }

    std::string name() const override
    {
        return "WriteAfterReadStartPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        conflict_reason_.clear();
        if (changed == kInvalidVarId)
            return true;
        const VarInfo &info = state.var_infos[changed];
        if (info.type != VarType::START || !state.domains[changed].isFixed())
            return true;

        const auto selected_it = state.selected_vars[info.bucket_idx].find(info.eclass_id);
        if (selected_it == state.selected_vars[info.bucket_idx].end())
            return true;
        const Domain &selection = state.domains[selected_it->second];
        if (selection.isFixed() && selection.fixedValue() == 0)
            return true;

        return propagateBucket(state, info.bucket_idx, worklist);
    }
};


} // namespace plan
