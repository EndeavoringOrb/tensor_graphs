#pragma once

#include "tests/incremental_propagators.hpp"

namespace write_after_read_cache_test
{
using namespace incremental_propagator_test;

inline void compareCold(SearchState &state, WriteAfterReadStartPropagator &warm, uint32_t bucket = 0)
{
    SearchState reference = state;
    WriteAfterReadStartPropagator cold;
    std::vector<VarId> warm_work, cold_work;
    const bool warm_ok = warm.propagate(state, start(state, 1, bucket), warm_work);
    const bool cold_ok = cold.propagate(reference, start(reference, 1, bucket), cold_work);
    require(warm_ok == cold_ok && state.domains == reference.domains && warm_work == cold_work,
            "Cached reader index differs from fresh full reader construction");
}

inline void requireCounter(const WriteAfterReadStartPropagator &prop, const std::string &key, uint64_t value)
{
#ifdef TG_PROFILE
    const std::string summary = prop.diagnosticSummary() + " ";
    require(summary.find(key + "=" + std::to_string(value) + " ") != std::string::npos,
            "Unexpected reader cache counter: " + summary);
#endif
}

inline void run()
{
    // Only allocations 1 and 2 overlap. Reader 3 reaches 1 through view 4.
    const GraphSpec spec = {{{{}}}, {{{}}}, {{{}}}, {{{4}}, {{}}}, {{{1}, {}, 1.0f, true}}};
    SearchState state = makeState(spec, 2);
    for (uint32_t b = 0; b < 2; ++b)
    {
        for (uint32_t node : {0u, 1u, 2u, 4u})
            state.setDomain(selection(state, node, b), Domain::makeFixed(1, true));
        state.setDomain(start(state, 1, b), Domain::makeFixed(0));
        state.setDomain(start(state, 2, b), Domain::makeRange(2, 9));
        state.setDomain(start(state, 3, b), Domain::makeRange(4, 9));
        state.setDomain(start(state, 4, b), Domain::makeFixed(1));
        state.setDomain(offset(state, 1, b), Domain::makeFixed(0));
        state.setDomain(offset(state, 2, b), Domain::makeFixed(0));
    }
    WriteAfterReadStartPropagator prop;
    const size_t marker = state.getTrailMarker();
    state.setDomain(selection(state, 3), Domain::makeFixed(1, true));
    compareCold(state, prop);
    require(state.domains[start(state, 2)].getMin() == 5, "View reader did not delay overwrite");
    state.setDomain(start(state, 3), Domain::makeRange(6, 9));
    compareCold(state, prop);
    require(state.domains[start(state, 2)].getMin() == 7, "Cached readers used stale start bounds");
    requireCounter(prop, "reader-rebuilds", 1);
    requireCounter(prop, "reader-cache-hits", 1);

    state.backtrackTo(marker);
    state.setDomain(selection(state, 3), Domain::makeFixed(2, true));
    compareCold(state, prop);
    require(state.domains[start(state, 2)].getMin() == 2, "Restoration retained alternate reader");
    requireCounter(prop, "reader-rebuilds", 2);
    state.backtrackTo(marker);
    state.setDomain(selection(state, 3), Domain::makeFixed(1, true));
    compareCold(state, prop);
    require(state.domains[start(state, 2)].getMin() == 5, "Restoration lost view reader");
    state.setDomain(selection(state, 3, 1), Domain::makeFixed(2, true));
    compareCold(state, prop, 1);
    require(state.domains[start(state, 2, 1)].getMin() == 2, "Buckets shared reader relationships");

    // Cache decisions change persistence without changing reader relationships.
    state.backtrackTo(marker);
    state.setDomain(selection(state, 3), Domain::makeFixed(1, true));
    state.setDomain(state.cached_vars.at(BaseEClassId{2}), Domain::makeFixed(1, true));
    compareCold(state, prop);
    require(state.domains[start(state, 2)].getMin() == 2, "Persistent allocation treated as overwrite");

    // Reader index is never needed without an eligible ordered overlap.
    SearchState lazy = makeState(spec);
    selectAll(lazy);
    lazy.setDomain(start(lazy, 0), Domain::makeFixed(0));
    lazy.setDomain(offset(lazy, 1), Domain::makeFixed(0));
    lazy.setDomain(offset(lazy, 2), Domain::makeFixed(0));
    WriteAfterReadStartPropagator lazy_prop;
    std::vector<VarId> work;
    require(lazy_prop.propagate(lazy, start(lazy, 0), work), "Lazy reader propagation failed");
    requireCounter(lazy_prop, "reader-rebuilds", 0);
    requireCounter(lazy_prop, "reader-index-skips", 1);

    // Random branch histories compare a warm cache with freshly rebuilt readers.
    // Include reader alternatives, view aliases, live bounds, conflicts, offsets,
    // persistence, both buckets, and selection undo.
    std::mt19937 rng(91823);
    SearchState random_state = makeState(spec, 2);
    for (uint32_t b = 0; b < 2; ++b)
        for (uint32_t node : {0u, 1u, 2u, 4u})
            random_state.setDomain(selection(random_state, node, b), Domain::makeFixed(1, true));
    const size_t random_marker = random_state.getTrailMarker();
    WriteAfterReadStartPropagator random_prop;
    for (uint32_t trial = 0; trial < 300; ++trial)
    {
        random_state.backtrackTo(random_marker);
        const uint32_t b = trial % 2;
        random_state.setDomain(selection(random_state, 3, b), Domain::makeFixed(1 + rng() % 2, true));
        random_state.setDomain(start(random_state, 1, b), Domain::makeFixed(0));
        random_state.setDomain(start(random_state, 2, b), Domain::makeRange(2, 2 + rng() % 8));
        random_state.setDomain(start(random_state, 3, b), Domain::makeRange(1 + rng() % 8, 9));
        random_state.setDomain(start(random_state, 4, b), Domain::makeFixed(1));
        random_state.setDomain(offset(random_state, 1, b), Domain::makeFixed(0));
        random_state.setDomain(offset(random_state, 2, b), Domain::makeFixed(rng() % 4));
        if (rng() % 4 == 0)
            random_state.setDomain(random_state.cached_vars.at(BaseEClassId{2}), Domain::makeFixed(1, true));
        compareCold(random_state, random_prop, b);
        compareCold(random_state, random_prop, b);
    }
    // A propagator reused on another state must not reuse the previous graph's index.
    lazy.setDomain(start(lazy, 1), Domain::makeFixed(0));
    lazy.setDomain(start(lazy, 2), Domain::makeRange(2, 9));
    lazy.setDomain(start(lazy, 3), Domain::makeRange(4, 9));
    lazy.setDomain(start(lazy, 4), Domain::makeFixed(1));
    compareCold(lazy, prop);
    require(lazy.domains[start(lazy, 2)].getMin() == 5, "Reader cache survived state replacement");
    std::cout << "Write-after-read cache tests passed (300 randomized restoration histories)" << std::endl;
}
} // namespace write_after_read_cache_test

inline void runWriteAfterReadCacheTests()
{
    write_after_read_cache_test::run();
}
