// tensor_graphs_cpp/core/plan/propagators/start_run_makespan.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/misc.hpp"
#include "core/plan/propagators/base.hpp"
#include "core/types.hpp"

namespace plan
{

class StartRunMakespanPropagator : public Propagator
{
  public:
    struct ScheduledOp
    {
        int32_t start_val = 0;
        EClassId cid;
        float cost = 0.0f;
        bool is_view = false;
        bool is_input_or_cache = false;
        std::vector<Engine> engines;
        std::vector<EClassId> children;
    };

    struct Run
    {
        size_t start_idx = 0; // index in scheduled_ops
        size_t end_idx = 0;   // exclusive index in scheduled_ops
        int32_t min_start = 0;
        int32_t max_start = 0;
    };

  private:
    struct CompactOp
    {
        int32_t start_val = 0;
        uint32_t node_idx = 0;
        uint32_t en_idx = 0;
    };

    struct StaticOpAlt
    {
        float duration = 0.0f;
        bool is_view = false;
        bool is_input_or_cache = false;
        std::vector<uint32_t> canon_children;
        std::vector<uint16_t> engine_ids;
    };

    struct StaticNode
    {
        EClassId cid;
        VarId st_vid = kInvalidVarId;
        VarId sel_vid = kInvalidVarId;
        std::vector<StaticOpAlt> alternatives;
    };

    struct BucketStaticData
    {
        std::vector<StaticNode> nodes;
        std::vector<Engine> engines;
        std::unordered_map<Engine, uint16_t> engine_to_id;
        uint32_t max_eclass_id = 0;
        bool initialized = false;
    };

    mutable std::vector<BucketStaticData> bucket_data_;
    mutable size_t cached_num_vars_ = 0;
    mutable bool initialized_ = false;

    // Scratch buffers reused across calls to avoid allocations
    mutable std::vector<CompactOp> scratch_compact_ops_;
    mutable std::vector<Run> scratch_runs_;
    mutable std::vector<float> scratch_eclass_finish_;
    mutable std::vector<uint32_t> scratch_eclass_epoch_;
    mutable std::vector<float> scratch_engine_finish_;
    mutable uint32_t current_epoch_ = 0;

    void initBucket(const SearchState &state, uint32_t b) const
    {
        auto &bdata = bucket_data_[b];
        bdata = BucketStaticData{};
        if (b >= state.start_vars.size() || b >= state.selected_vars.size() ||
            b >= state.bucket_egraphs.size())
            return;

        const auto &st_vars = state.start_vars[b];
        const auto &sel_vars = state.selected_vars[b];
        const auto &egraph = state.bucket_egraphs[b];

        const Engine cpu{0, EngineType::CPU};
        bdata.engines.push_back(cpu);
        bdata.engine_to_id[cpu] = 0;

        uint32_t max_cid = 0;

        for (const auto &[cid, st_vid] : st_vars)
        {
            auto sel_it = sel_vars.find(cid);
            if (sel_it == sel_vars.end())
                continue;
            const VarId sel_vid = sel_it->second;

            if (cid.value > max_cid && cid.value != UINT32_MAX)
                max_cid = cid.value;

            StaticNode node;
            node.cid = cid;
            node.st_vid = st_vid;
            node.sel_vid = sel_vid;

            const EClass &cls = egraph.getEClass(cid);
            node.alternatives.resize(cls.enodes.size());

            for (uint32_t en_idx = 0; en_idx < cls.enodes.size(); ++en_idx)
            {
                const ENodeId en_id = cls.enodes[en_idx];
                const ENode &enode = egraph.getENode(en_id);
                auto &alt = node.alternatives[en_idx];

                alt.is_input_or_cache = (enode.getOpType() == OpType::INPUT ||
                                         enode.getOpType() == OpType::CACHE);

                float cost = (b < state.bucket_enode_infos.size() && en_id.value < state.bucket_enode_infos[b].size())
                                 ? state.bucket_enode_infos[b][en_id.value].cost
                                 : 0.0f;
                if (cost == TGConstants::INF || std::isnan(cost) || alt.is_input_or_cache)
                    cost = 0.0f;

                alt.is_view = (b < state.bucket_enode_infos.size() && en_id.value < state.bucket_enode_infos[b].size()) &&
                              state.bucket_enode_infos[b][en_id.value].is_view;

                alt.duration = alt.is_view ? 0.0f : cost;

                alt.canon_children.reserve(enode.getChildren().size());
                for (EClassId ch : enode.getChildren())
                {
                    const EClassId canon_ch = egraph.findConst(ch);
                    alt.canon_children.push_back(canon_ch.value);
                    if (canon_ch.value > max_cid && canon_ch.value != UINT32_MAX)
                        max_cid = canon_ch.value;
                }

                if (!alt.is_view && !alt.is_input_or_cache)
                {
                    if (enode.getEngines().empty())
                    {
                        alt.engine_ids.push_back(0); // CPU is index 0
                    }
                    else
                    {
                        for (const Engine &eng : enode.getEngines())
                        {
                            auto it = bdata.engine_to_id.find(eng);
                            if (it == bdata.engine_to_id.end())
                            {
                                const uint16_t new_id = static_cast<uint16_t>(bdata.engines.size());
                                bdata.engine_to_id[eng] = new_id;
                                bdata.engines.push_back(eng);
                                alt.engine_ids.push_back(new_id);
                            }
                            else
                            {
                                alt.engine_ids.push_back(it->second);
                            }
                        }
                    }
                }
            }
            bdata.nodes.push_back(std::move(node));
        }

        bdata.max_eclass_id = max_cid;
        bdata.initialized = true;
    }

    void ensureInitialized(const SearchState &state) const
    {
        bool needs_init = !initialized_ || bucket_data_.size() != state.buckets.size() ||
                          cached_num_vars_ != state.numVars();
        if (!needs_init)
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                if (b >= state.start_vars.size() ||
                    bucket_data_[b].nodes.size() != state.start_vars[b].size())
                {
                    needs_init = true;
                    break;
                }
            }
        }

        if (needs_init)
        {
            bucket_data_.resize(state.buckets.size());
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                initBucket(state, b);
            }
            cached_num_vars_ = state.numVars();
            initialized_ = true;
        }
    }

  public:
    float computeGapLowerBound(const SearchState &state, uint32_t bucket_idx,
                               const Run &prev_run, const Run &next_run,
                               int32_t gap_size,
                               const std::vector<ScheduledOp> &ops = {}) const
    {
        if (gap_size <= 0)
            return 0.0f;

        // Safe baseline: return 0.0f.
        // Tighter bounds (e.g. minimum costs of required unassigned nodes, or critical
        // path distances through the gap) can be added here and validated by tests.
        return 0.0f;
    }

    float computeBucketStartRunLowerBound(const SearchState &state, uint32_t b) const
    {
        if (b >= state.buckets.size())
            Error::throw_err("bucket_idx " + std::to_string(b) + " out of bounds (" +
                             std::to_string(state.buckets.size()) + ") in StartRunMakespanPropagator");

        if (b >= state.start_vars.size() || b >= state.selected_vars.size() ||
            b >= state.bucket_egraphs.size())
            return 0.0f;

        ensureInitialized(state);
        if (b >= bucket_data_.size())
            return 0.0f;

        const auto &bdata = bucket_data_[b];
        if (!bdata.initialized || bdata.nodes.empty())
            return 0.0f;

        scratch_compact_ops_.clear();

        for (uint32_t n_idx = 0; n_idx < bdata.nodes.size(); ++n_idx)
        {
            const auto &node = bdata.nodes[n_idx];
            if (node.st_vid >= state.domains.size())
                continue;
            const Domain &st_dom = state.domains[node.st_vid];
            if (!st_dom.isFixed())
                continue;

            if (node.sel_vid >= state.domains.size())
                continue;
            const Domain &sel_dom = state.domains[node.sel_vid];
            // Sound lower bound: include operations known to be selected (>0).
            if (!sel_dom.isFixed() || sel_dom.fixedValue() <= 0)
                continue;

            const uint32_t en_idx = static_cast<uint32_t>(sel_dom.fixedValue() - 1);
            if (en_idx >= node.alternatives.size())
                continue;

            scratch_compact_ops_.push_back({st_dom.fixedValue(), n_idx, en_idx});
        }

        if (scratch_compact_ops_.empty())
            return 0.0f;

        std::sort(scratch_compact_ops_.begin(), scratch_compact_ops_.end(),
                  [](const CompactOp &p, const CompactOp &q) {
                      return p.start_val < q.start_val;
                  });

        // Partition into contiguous runs
        scratch_runs_.clear();
        size_t cur_start = 0;
        for (size_t i = 1; i <= scratch_compact_ops_.size(); ++i)
        {
            if (i == scratch_compact_ops_.size() ||
                scratch_compact_ops_[i].start_val != scratch_compact_ops_[i - 1].start_val + 1)
            {
                Run r;
                r.start_idx = cur_start;
                r.end_idx = i;
                r.min_start = scratch_compact_ops_[cur_start].start_val;
                r.max_start = scratch_compact_ops_[i - 1].start_val;
                scratch_runs_.push_back(r);
                cur_start = i;
            }
        }

        if (scratch_eclass_finish_.size() <= bdata.max_eclass_id)
        {
            scratch_eclass_finish_.resize(bdata.max_eclass_id + 1, 0.0f);
            scratch_eclass_epoch_.resize(bdata.max_eclass_id + 1, 0);
        }

        ++current_epoch_;
        if (current_epoch_ == 0)
        {
            std::fill(scratch_eclass_epoch_.begin(), scratch_eclass_epoch_.end(), 0);
            current_epoch_ = 1;
        }
        const uint32_t cur_epoch = current_epoch_;

        if (scratch_engine_finish_.size() < bdata.engines.size())
            scratch_engine_finish_.resize(bdata.engines.size(), 0.0f);
        else
            std::fill_n(scratch_engine_finish_.begin(), bdata.engines.size(), 0.0f);

        // Chained Timeline Simulation across runs with gap advancement
        for (size_t r = 0; r < scratch_runs_.size(); ++r)
        {
            const Run &run = scratch_runs_[r];

            if (r > 0)
            {
                const int32_t gap_size = run.min_start - scratch_runs_[r - 1].max_start - 1;
                if (gap_size > 0)
                {
                    const float gap_lb = computeGapLowerBound(state, b, scratch_runs_[r - 1], run, gap_size);
                    if (gap_lb > 0.0f)
                    {
                        for (size_t eng_idx = 0; eng_idx < bdata.engines.size(); ++eng_idx)
                        {
                            scratch_engine_finish_[eng_idx] += gap_lb;
                        }
                    }
                }
            }

            for (size_t i = run.start_idx; i < run.end_idx; ++i)
            {
                const auto &cop = scratch_compact_ops_[i];
                const auto &node = bdata.nodes[cop.node_idx];
                const auto &alt = node.alternatives[cop.en_idx];
                const uint32_t cid_val = node.cid.value;

                if (alt.is_input_or_cache)
                {
                    if (cid_val < scratch_eclass_finish_.size())
                    {
                        scratch_eclass_finish_[cid_val] = 0.0f;
                        scratch_eclass_epoch_[cid_val] = cur_epoch;
                    }
                    continue;
                }

                float children_finish = 0.0f;
                for (uint32_t ch_val : alt.canon_children)
                {
                    if (ch_val < scratch_eclass_finish_.size() &&
                        scratch_eclass_epoch_[ch_val] == cur_epoch)
                    {
                        children_finish = std::max(children_finish, scratch_eclass_finish_[ch_val]);
                    }
                }

                float engine_free = 0.0f;
                for (uint16_t eng_id : alt.engine_ids)
                {
                    if (eng_id < scratch_engine_finish_.size())
                        engine_free = std::max(engine_free, scratch_engine_finish_[eng_id]);
                }

                const float start_time = std::max(children_finish, engine_free);
                const float finish_time = start_time + alt.duration;

                if (cid_val < scratch_eclass_finish_.size())
                {
                    scratch_eclass_finish_[cid_val] = finish_time;
                    scratch_eclass_epoch_[cid_val] = cur_epoch;
                }

                for (uint16_t eng_id : alt.engine_ids)
                {
                    if (eng_id < scratch_engine_finish_.size())
                        scratch_engine_finish_[eng_id] = finish_time;
                }
            }
        }

        float total_makespan = 0.0f;
        for (size_t i = 0; i < bdata.engines.size(); ++i)
        {
            total_makespan = std::max(total_makespan, scratch_engine_finish_[i]);
        }
        return total_makespan;
    }

    uint8_t interestedVarTypes() const override
    {
        return varTypeMask(VarType::START) | varTypeMask(VarType::SELECTED);
    }

    StartSelectionGuard startSelectionGuard() const override
    {
        return StartSelectionGuard::FIXED_START_POSITIVE;
    }

    std::string name() const override
    {
        return "StartRunMakespanPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (changed != kInvalidVarId)
        {
            if (changed >= state.var_infos.size())
                Error::throw_err("changed VarId " + std::to_string(changed) + " out of bounds in StartRunMakespanPropagator");
            const auto &var_info = state.var_infos[changed];
            if (var_info.type != VarType::START &&
                var_info.type != VarType::SELECTED)
                return true;

            const uint32_t b = var_info.bucket_idx;
            if (b >= state.buckets.size())
                Error::throw_err("bucket_idx " + std::to_string(b) + " out of bounds in StartRunMakespanPropagator");

            // Fast prune: an unfixed START variable cannot affect scheduled operations
            if (var_info.type == VarType::START && !state.domains[changed].isFixed())
                return true;

            if (state.bucket_start_run_lower_bounds.size() < state.buckets.size())
                state.bucket_start_run_lower_bounds.resize(state.buckets.size(), 0.0f);

            state.bucket_start_run_lower_bounds[b] = computeBucketStartRunLowerBound(state, b);
            state.updateLowerBoundBucket(b);
        }
        else
        {
            if (state.bucket_start_run_lower_bounds.size() < state.buckets.size())
                state.bucket_start_run_lower_bounds.resize(state.buckets.size(), 0.0f);

            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                state.bucket_start_run_lower_bounds[b] = computeBucketStartRunLowerBound(state, b);
                state.updateLowerBoundBucket(b);
            }
        }

        if (state.best_cost < TGConstants::INF && state.lower_bound >= state.best_cost)
        {
            return false;
        }
        return true;
    }
};

} // namespace plan
