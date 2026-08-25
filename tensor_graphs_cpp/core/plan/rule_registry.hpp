// tensor_graphs_cpp/core/plan/rule_registry.hpp
//
// Central rule registry: lists every pruning rule the framework knows about,
// grouped by iterator category. The pruning test framework (tests/pruning/)
// uses this registry to AUTO-DISCOVER rules so that adding a new rule here is
// the only change required to also test + benchmark it -- no manual edits to
// tests/pruning/test.hpp or tests/pruning/benchmark.hpp are needed.
//
// Categories currently registered:
//
//   dispatch   -- DispatchIterator (ENode-level dispatch order pruning)
//   bufferize  -- BufferizeIterator (in-place vs. new-buffer choice pruning)
//   malloc     -- MallocIterator (offset + mem_cap pruning during packing)
//   cache      -- CacheIterator (per-logical-id caching choice pruning)
//   extract    -- Extractor (ENode selection pruning during topological walk)
//
// Adding a new rule:
//
//   1. Define the rule struct in the appropriate iterator header
//      (it must implement `bool check(Cand, size_t, const Ctx&) const`
//      plus optionally `void init(const Ctx&)`, `const char* name() const`).
//   2. Add a `using <RuleName> = <RuleClass>;` alias and append
//      `<RuleName>` to the relevant `All<...>Rules` alias below.
//   3. The test/benchmark framework picks it up automatically on next build.
//
// No code outside this file needs to change when adding a rule, EXCEPT the
// rule's own iterator factory (makeConfigured*Iterator) which decides whether
// the rule is enabled by default in production. Tests will still exercise the
// rule independently regardless of that default.
#pragma once

#include "core/plan/extractor.hpp"      // DispatchIterator + Extractor rules
#include "core/plan/planner.hpp"        // CacheIterator + ENode domination rules
#include "core/plan/validators/mem.hpp" // BufferizeIterator + MallocIterator rules

// =============================================================================
// DispatchIterator rules
// =============================================================================
namespace dispatch_rules
{
} // namespace dispatch_rules

using AllDispatchRuleTypes = std::tuple<>;

// =============================================================================
// BufferizeIterator rules
// =============================================================================
namespace bufferizeRules
{
using MemSpaceMismatchInplaceRuleT = MemSpaceMismatchInplaceRule;
using LinearChainInplaceDominationRuleT = LinearChainInplaceDominationRule;
using IntervalSubsetDominationRuleT = IntervalSubsetDominationRule;
using CommutativeInplaceSymmetryRuleT = CommutativeInplaceSymmetryRule;
using DeadBufferReuseDominationRuleT = DeadBufferReuseDominationRule;
} // namespace bufferizeRules

using AllBufferizeRuleTypes =
    std::tuple<MemSpaceMismatchInplaceRule, LinearChainInplaceDominationRule, IntervalSubsetDominationRule,
               CommutativeInplaceSymmetryRule, DeadBufferReuseDominationRule>;

// =============================================================================
// MallocIterator rules
// =============================================================================
namespace mallocRules
{
using OffsetMonotoneRuleT = OffsetMonotoneRule;
using IdMaxSymmetryRuleT = IdMaxSymmetryRule;
using CapRespectRuleT = CapRespectRule;
using HMinBoundRuleT = HMinBoundRule;
using LargerBufferPriorityRuleT = LargerBufferPriorityRule;
} // namespace mallocRules

using AllMallocRuleTypes =
    std::tuple<OffsetMonotoneRule, IdMaxSymmetryRule, CapRespectRule, HMinBoundRule, LargerBufferPriorityRule>;

// =============================================================================
// CacheIterator rules
// =============================================================================
namespace cacheRules
{
using SingleUseSkipRuleT = SingleUseSkipRule;
using TinyBufferSkipRuleT = TinyBufferSkipRule;
using StorageAnchoredSkipRuleT = StorageAnchoredSkipRule;
} // namespace cacheRules

using AllCacheRuleTypes = std::tuple<SingleUseSkipRule, TinyBufferSkipRule, StorageAnchoredSkipRule>;

// =============================================================================
// Extractor (DFS) rules -- separate from pre-extraction ENode domination
// =============================================================================
namespace extractRules
{
using InfiniteCostSkipRuleT = InfiniteCostSkipRule;
using SiblingEquivalentSkipRuleT = SiblingEquivalentSkipRule;
} // namespace extractRules

using AllExtractRuleTypes = std::tuple<InfiniteCostSkipRule, SiblingEquivalentSkipRule>;

// =============================================================================
// Pre-extraction ENode domination rules (run by Planner.applyDominationRules).
// These are NOT DFS pruning rules; they're whole-graph filters applied once
// before extraction begins. Tests verify them by checking that ENodes marked
// INF are removed from the EGraph.
// =============================================================================
namespace enodeDominationRules
{
using MemCapENodeDominationRuleT = MemCapENodeDominationRule;
using FasterEquivalentENodeDominationRuleT = FasterEquivalentENodeDominationRule;
using DeadChildChainDominationRuleT = DeadChildChainDominationRule;
} // namespace enodeDominationRules

using AllENodeDominationRuleTypes =
    std::tuple<MemCapENodeDominationRule, FasterEquivalentENodeDominationRule, DeadChildChainDominationRule>;

// =============================================================================
// Rule-list helpers -- the test framework iterates over these to enumerate
// each rule by name and run a per-rule test + benchmark in isolation.
//
// RuleSpec {
//   const char* category;       // "dispatch", "bufferize", ...
//   const char* rule_name;      // human-readable name (must match rule.name())
//   FactoryFn  make_iter;       // constructs an iterator with ONLY this rule
// };
//
// Because each iterator template requires concrete rule types at compile time,
// we cannot enumerate rules purely at runtime. Instead, the test framework
// uses std::apply + index_sequence to expand the tuple into a sequence of
// per-rule test invocations at compile time.
// =============================================================================

struct RuleSpec
{
    const char *category;
    const char *rule_name;
};

// Compile-time list of rule specs per category. Keep in lock-step with the
// `All*RuleTypes` aliases above.
inline constexpr RuleSpec kAllRuleSpecs[] = {
    // bufferize
    {"bufferize", "MemSpaceMismatchInplaceRule"},
    {"bufferize", "LinearChainInplaceDominationRule"},
    {"bufferize", "IntervalSubsetDominationRule"},
    {"bufferize", "CommutativeInplaceSymmetryRule"},
    {"bufferize", "DeadBufferReuseDominationRule"},
    // malloc
    {"malloc", "OffsetMonotoneRule"},
    {"malloc", "IdMaxSymmetryRule"},
    {"malloc", "CapRespectRule"},
    {"malloc", "HMinBoundRule"},
    {"malloc", "LargerBufferPriorityRule"},
    // cache
    {"cache", "SingleUseSkipRule"},
    {"cache", "TinyBufferSkipRule"},
    {"cache", "StorageAnchoredSkipRule"},
    // extract (DFS)
    {"extract", "InfiniteCostSkipRule"},
    {"extract", "SiblingEquivalentSkipRule"},
    // extract (pre-extraction ENode domination)
    {"enode", "MemCapENodeDominationRule"},
    {"enode", "FasterEquivalentENodeDominationRule"},
    {"enode", "DeadChildChainDominationRule"},
};
