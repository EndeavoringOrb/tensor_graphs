// tensor_graphs_cpp/core/plan/propagators/engine_workload.hpp
#pragma once

#include "core/plan/propagators/base.hpp"

namespace plan
{

class EngineWorkloadPropagator : public Propagator
{
  public:
    uint8_t interestedVarTypes() const override { return varTypeMask(VarType::SELECTED); }

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
        if (!state.engine_work_initialized)
            initialize(state);
        if (changed != kInvalidVarId && state.var_infos[changed].type != VarType::SELECTED)
            return true;
        if (changed != kInvalidVarId && state.engine_work_initialized)
        {
            if (changed >= state.var_infos.size())
                Error::throw_err("changed VarId " + std::to_string(changed) + " out of bounds in EngineWorkloadPropagator");
            const uint32_t bucket_idx = state.var_infos[changed].bucket_idx;
            if (bucket_idx >= state.buckets.size())
                Error::throw_err("bucket_idx " + std::to_string(bucket_idx) + " out of bounds in EngineWorkloadPropagator");
            updateSelectedWork(state, changed);
            state.bucket_engine_work_lower_bounds[bucket_idx] = bucketEngineWorkLowerBound(state.engine_work[bucket_idx]);
            state.updateLowerBoundBucket(bucket_idx);
        }
        if (state.best_cost < TGConstants::INF && state.lower_bound >= state.best_cost)
            return false;
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


} // namespace plan
