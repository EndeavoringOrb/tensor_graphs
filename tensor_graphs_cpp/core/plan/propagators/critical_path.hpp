// tensor_graphs_cpp/core/plan/propagators/critical_path.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

class CriticalPathPropagator : public Propagator
{
    struct StaticAlternative
    {
        float cost = 0.0f;
        std::vector<uint32_t> children;
    };

    struct StaticNode
    {
        VarId selection_var = kInvalidVarId;
        std::vector<StaticAlternative> alternatives;
    };

    struct BucketData
    {
        std::vector<StaticNode> nodes;
        uint32_t root_idx = UINT32_MAX;
        std::vector<float> memo;
        std::vector<uint32_t> memo_epochs;
        std::vector<uint32_t> visiting_epochs;
        uint32_t epoch = 0;
    };

    mutable std::vector<BucketData> bucket_data_;
    mutable size_t cached_num_vars_ = 0;
    mutable bool static_initialized_ = false;

    void ensureStaticData(const SearchState &state) const
    {
        if (static_initialized_ && cached_num_vars_ == state.numVars() && bucket_data_.size() == state.buckets.size())
            return;
#ifdef TG_PROFILE
        InitializationProfileScope initialization_scope(initialization_profile_ns_);
#endif
        bucket_data_.clear();
        bucket_data_.resize(state.buckets.size());
        // Buckets are independent: each fills only its own data.
        ThreadPool::get().parallel_for(static_cast<uint32_t>(state.buckets.size()), [&](uint32_t b) {
            auto &data = bucket_data_[b];
            const auto &graph = state.bucket_egraphs.at(b);
            const auto &selections = state.selected_vars.at(b);
            std::unordered_map<EClassId, uint32_t> node_indices;
            std::vector<EClassId> node_ids;
            node_ids.reserve(selections.size());
            for (const auto &[cid, var_id] : selections)
            {
                const auto canon_cid = graph.findConst(cid);
                if (canon_cid != cid)
                    Error::throw_err("Noncanonical selection in CriticalPathPropagator");
                node_indices.emplace(cid, static_cast<uint32_t>(node_ids.size()));
                node_ids.push_back(cid);
                data.nodes.push_back({var_id, {}});
            }
            const auto root_it = node_indices.find(graph.findConst(state.bucket_root_ids.at(b)));
            if (root_it != node_indices.end())
                data.root_idx = root_it->second;
            for (uint32_t idx = 0; idx < node_ids.size(); ++idx)
            {
                auto &node = data.nodes[idx];
                for (ENodeId en_id : graph.getEClass(node_ids[idx]).enodes)
                {
                    const auto &enode = graph.getENode(en_id);
                    StaticAlternative alt;
                    if (b < state.bucket_enode_infos.size() && en_id.value < state.bucket_enode_infos[b].size())
                    {
                        const auto &info = state.bucket_enode_infos[b][en_id.value];
                        alt.cost = info.cost;
                        if (info.is_view || alt.cost == TGConstants::INF || std::isnan(alt.cost))
                            alt.cost = 0.0f;
                    }
                    if (enode.getOpType() == OpType::INPUT || enode.getOpType() == OpType::CACHE)
                        alt.cost = 0.0f;
                    alt.children.reserve(enode.getChildren().size());
                    for (EClassId child : enode.getChildren())
                    {
                        const auto child_it = node_indices.find(graph.findConst(child));
                        alt.children.push_back(child_it == node_indices.end() ? UINT32_MAX : child_it->second);
                    }
                    node.alternatives.push_back(std::move(alt));
                }
            }
            data.memo.resize(data.nodes.size());
            data.memo_epochs.resize(data.nodes.size(), 0);
            data.visiting_epochs.resize(data.nodes.size(), 0);
        });
        cached_num_vars_ = state.numVars();
        static_initialized_ = true;
    }

    float computeNodeCriticalPath(const SearchState &state, BucketData &data, uint32_t idx) const
    {
        if (idx == UINT32_MAX)
            return 0.0f;
        if (data.memo_epochs[idx] == data.epoch)
            return data.memo[idx];
        // Preserve the existing conservative bound for cyclic alternatives.
        if (data.visiting_epochs[idx] == data.epoch)
            return 0.0f;
        data.visiting_epochs[idx] = data.epoch;
        const auto &node = data.nodes[idx];
        const Domain &domain = state.domains[node.selection_var];
        float cp_val = 0.0f;
        if (!domain.contains(0))
        {
            float min_cp = TGConstants::INF;
            const uint32_t first_alt = domain.isFixed() ? static_cast<uint32_t>(domain.fixedValue() - 1) : 0;
            const size_t last_alt = domain.isFixed() ? static_cast<size_t>(first_alt) + 1 : node.alternatives.size();
            if (last_alt > node.alternatives.size())
                Error::throw_err("Invalid selected alternative in CriticalPathPropagator");
            for (uint32_t en_idx = first_alt; en_idx < last_alt; ++en_idx)
            {
                if (!domain.contains(en_idx + 1))
                    continue;
                const auto &alt = node.alternatives[en_idx];
                float max_child = 0.0f;
                for (uint32_t child : alt.children)
                    max_child = std::max(max_child, computeNodeCriticalPath(state, data, child));
                min_cp = std::min(min_cp, alt.cost + max_child);
            }
            cp_val = min_cp < TGConstants::INF ? min_cp : 0.0f;
        }
        data.visiting_epochs[idx] = 0;
        data.memo_epochs[idx] = data.epoch;
        data.memo[idx] = cp_val;
        return cp_val;
    }

    float computeBucketCriticalPath(const SearchState &state, uint32_t b) const
    {
        ensureStaticData(state);
        auto &data = bucket_data_.at(b);
        if (++data.epoch == 0)
        {
            std::fill(data.memo_epochs.begin(), data.memo_epochs.end(), 0);
            std::fill(data.visiting_epochs.begin(), data.visiting_epochs.end(), 0);
            data.epoch = 1;
        }
        return computeNodeCriticalPath(state, data, data.root_idx);
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
    PropagationBatch propagationBatch(const SearchState &state, VarId changed) const override
    {
        // Bound only: a trial plan is accepted only if its leaf cost beats
        // the incumbent, so completing one needs no critical-path bound.
        if (state.completing_plan)
            return PropagationBatch::SKIP;
        return changed == kInvalidVarId ? PropagationBatch::GLOBAL : PropagationBatch::BUCKET;
    }
    bool changesDomains() const override { return false; }

    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }
    void prepare(const SearchState &state) override { ensureStaticData(state); }

    std::string name() const override
    {
        return "CriticalPathPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (!state.critical_path_lower_bound_initialized)
            initialize(state);
        if (changed != kInvalidVarId)
        {
            if (changed >= state.var_infos.size())
                Error::throw_err("changed VarId " + std::to_string(changed) + " out of bounds in CriticalPathPropagator");
            if (state.var_infos[changed].type != VarType::SELECTED)
                return true;
            const uint32_t bucket_idx = state.var_infos[changed].bucket_idx;
            if (bucket_idx >= state.buckets.size())
                Error::throw_err("bucket_idx " + std::to_string(bucket_idx) + " out of bounds in CriticalPathPropagator");
            state.bucket_critical_path_lower_bounds.at(bucket_idx) = computeBucketCriticalPath(state, bucket_idx);
            state.updateLowerBoundBucket(bucket_idx);
        }
        else
        {
            for (uint32_t b = 0; b < state.buckets.size(); ++b)
            {
                state.bucket_critical_path_lower_bounds[b] = computeBucketCriticalPath(state, b);
                state.updateLowerBoundBucket(b);
            }
        }
        if (!state.lower_bounds_pending && state.best_cost < TGConstants::INF && state.lower_bound >= state.best_cost)
            return false;
        return true;
    }
};


} // namespace plan
