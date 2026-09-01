#pragma once

#include "core/plan/extractor.hpp"      // AllDispatchRuleTypes + AllExtractRuleTypes
#include "core/plan/planner.hpp"        // AllCacheRuleTypes + AllENodeDominationRuleTypes
#include "core/plan/mem.hpp" // AllBufferizeRuleTypes + AllMallocRuleTypes

inline constexpr auto kAllRuleSpecsArray = prune::concat_arrays(
    prune::make_category_specs<AllDispatchRuleTypes>("dispatch"),
    prune::make_category_specs<AllBufferizeRuleTypes>("bufferize"),
    prune::make_category_specs<AllMallocRuleTypes>("malloc"), prune::make_category_specs<AllCacheRuleTypes>("cache"),
    prune::make_category_specs<AllExtractRuleTypes>("extract"),
    prune::make_category_specs<AllENodeDominationRuleTypes>("enode"));

inline constexpr const auto &kAllRuleSpecs = kAllRuleSpecsArray;

inline void enableAllDefaultRules(Settings &settings, bool enabled = true)
{
    for (const auto &spec : kAllRuleSpecs)
    {
        settings.set_rule_enabled(spec.category, spec.rule_name, enabled);
    }
    for (const std::string &cat : {"dispatch", "bufferize", "malloc", "cache", "extract", "enode"})
    {
        settings.category_defined[cat] = true;
    }
}