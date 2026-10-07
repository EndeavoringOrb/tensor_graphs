// tensor_graphs_cpp/core/plan/propagator.hpp
#pragma once

#include "core/plan/propagators/base.hpp"
#include "core/plan/propagators/view_offset_helpers.hpp"

#include "core/plan/propagators/selection_reachability.hpp"
#include "core/plan/propagators/selection_children.hpp"
#include "core/plan/propagators/unselected_start_offset.hpp"
#include "core/plan/propagators/cache_exclusion.hpp"
#include "core/plan/propagators/input_prune_start_precedence.hpp"
#include "core/plan/propagators/consumer_start_precedence.hpp"
#include "core/plan/propagators/start_precedence.hpp"
#include "core/plan/propagators/start_unique.hpp"
#include "core/plan/propagators/memory_no_overlap.hpp"
#include "core/plan/propagators/view_select_offset.hpp"
#include "core/plan/propagators/view_to_base_offset.hpp"
#include "core/plan/propagators/base_to_view_offset.hpp"
#include "core/plan/propagators/cycle_avoidance.hpp"
#include "core/plan/propagators/pearce_kelly_cycle.hpp"
#include "core/plan/propagators/parent_removal.hpp"
#include "core/plan/propagators/cache_requirement.hpp"
#include "core/plan/propagators/cached_offset.hpp"
#include "core/plan/propagators/cached_offset_allocation.hpp"
#include "core/plan/propagators/critical_path.hpp"
#include "core/plan/propagators/cache_budget.hpp"
#include "core/plan/propagators/early_cache_budget.hpp"
#include "core/plan/propagators/engine_workload.hpp"
#include "core/plan/propagators/write_after_read_start.hpp"

namespace plan
{

// Composite ViewOffsetPropagator for backward compatibility
class ViewOffsetPropagator : public Propagator
{
    ViewSelectOffsetPropagator view_select_;
    ViewToBaseOffsetPropagator view_to_base_;
    BaseToViewOffsetPropagator base_to_view_;

  public:
    std::string name() const override
    {
        return "ViewOffsetPropagator";
    }

    bool propagate(SearchState &state, VarId changed, std::vector<VarId> &worklist) override
    {
        if (!view_select_.propagate(state, changed, worklist))
            return false;
        if (!view_to_base_.propagate(state, changed, worklist))
            return false;
        if (!base_to_view_.propagate(state, changed, worklist))
            return false;
        return true;
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
    engine.addPropagator(std::make_unique<InputPruneStartPrecedencePropagator>());
    engine.addPropagator(std::make_unique<ConsumerStartPrecedencePropagator>(fixed_starts_only));
    engine.addPropagator(std::make_unique<StartPrecedencePropagator>());
    engine.addPropagator(std::make_unique<StartUniquePropagator>());
    engine.addPropagator(std::make_unique<MemoryNoOverlapPropagator>());
    engine.addPropagator(std::make_unique<ViewSelectOffsetPropagator>());
    engine.addPropagator(std::make_unique<ViewToBaseOffsetPropagator>());
    engine.addPropagator(std::make_unique<BaseToViewOffsetPropagator>());
    engine.addPropagator(std::make_unique<CycleAvoidancePropagator>());
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
    engine.addPropagator(std::make_unique<EarlyCacheBudgetPropagator>());
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
