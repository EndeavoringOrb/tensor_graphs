// tensor_graphs_cpp/core/plan/propagators/critical_path.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

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
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }

    std::string name() const override
    {
        return "CriticalPathPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (!state.critical_path_lower_bound_initialized)
            initialize(state);
        if (changed != kInvalidVarId && state.var_infos[changed].type != VarType::SELECTED)
            return true;
        if (changed != kInvalidVarId)
        {
            if (changed >= state.var_infos.size())
                Error::throw_err("changed VarId " + std::to_string(changed) + " out of bounds in CriticalPathPropagator");
            const uint32_t bucket_idx = state.var_infos[changed].bucket_idx;
            if (bucket_idx >= state.buckets.size())
                Error::throw_err("bucket_idx " + std::to_string(bucket_idx) + " out of bounds in CriticalPathPropagator");
            if (bucket_idx < state.bucket_critical_path_lower_bounds.size())
            {
                state.bucket_critical_path_lower_bounds[bucket_idx] = computeBucketCriticalPath(state, bucket_idx);
                state.updateLowerBoundBucket(bucket_idx);
            }
        }
        if (state.best_cost < TGConstants::INF && state.lower_bound >= state.best_cost)
            return false;
        return true;
    }
};


} // namespace plan
