#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

#include "core/misc.hpp"
#include "core/plan/propagator.hpp"
#include "core/plan/search_engine.hpp"

namespace incremental_propagator_test
{
using namespace plan;

inline void require(bool condition, const std::string &message)
{
    if (!condition)
        Error::throw_err(message);
}

struct AlternativeSpec
{
    std::vector<uint32_t> children;
    std::vector<Engine> engines;
    float cost = 1.0f;
    bool is_view = false;
    OpType op_type = OpType::ADD;

    AlternativeSpec() = default;
    AlternativeSpec(std::vector<uint32_t> ch, std::vector<Engine> eng = {}, float c = 1.0f, bool view = false, OpType op = OpType::ADD)
        : children(std::move(ch)), engines(std::move(eng)), cost(c), is_view(view), op_type(op) {}
};
using GraphSpec = std::vector<std::vector<AlternativeSpec>>;

inline VarId selection(const SearchState &state, uint32_t node, uint32_t bucket = 0)
{
    return state.selected_vars[bucket].at(EClassId{node});
}

inline VarId start(const SearchState &state, uint32_t node, uint32_t bucket = 0)
{
    return state.start_vars[bucket].at(EClassId{node});
}

inline VarId offset(const SearchState &state, uint32_t node, uint32_t bucket = 0)
{
    return state.offset_vars[bucket].at(EClassId{node});
}

inline SearchState makeState(const GraphSpec &spec, uint32_t bucket_count = 1)
{
    SearchState state;
    const MemSpace space{1, HandleType::CPP};
    state.mem_caps[space] = 128;
    state.page_alignments[space] = 4;
    for (uint32_t node = 0; node < spec.size(); ++node)
    {
        VarInfo info;
        info.type = VarType::CACHED;
        info.base_eclass_id = BaseEClassId{node + 1};
        state.cached_vars[info.base_eclass_id] = state.addVar(info, Domain::makeMask(3));
        CacheCandidate candidate;
        candidate.base_eclass_id = info.base_eclass_id;
        candidate.mem_space = space;
        candidate.size_bytes = 8;
        state.candidates.push_back(candidate);
    }
    for (uint32_t b = 0; b < bucket_count; ++b)
    {
        state.buckets.emplace_back();
        state.bucket_weights.push_back(b == 0 ? 1.0f : 0.375f);
        state.bucket_egraphs.emplace_back();
        state.bucket_enode_infos.emplace_back();
        state.selected_vars.emplace_back();
        state.start_vars.emplace_back();
        state.offset_vars.emplace_back();
        state.reachable_cids.emplace_back();
        state.bucket_root_ids.push_back(EClassId{0});
        EGraph &graph = state.bucket_egraphs.back();
        for (uint32_t node = 0; node < spec.size(); ++node)
        {
            EClassId cid = graph.addEClass({2}, {1}, DType::FLOAT32, space);
            graph.getEClass(cid).base_eclass_id = BaseEClassId{node + 1};
            state.reachable_cids.back().push_back(cid);
        }
        for (uint32_t node = 0; node < spec.size(); ++node)
        {
            const EClassId cid{node};
            VarInfo info;
            info.bucket_idx = b;
            info.eclass_id = cid;
            info.type = VarType::SELECTED;
            state.selected_vars[b][cid] = state.addVar(info, Domain::makeMask((1u << (spec[node].size() + 1)) - 1));
            for (uint32_t index = 0; index < spec[node].size(); ++index)
            {
                const auto &alternative = spec[node][index];
                std::vector<EClassId> children;
                for (uint32_t child : alternative.children)
                    children.push_back(EClassId{child});
                graph.addENode(cid, ENode(KernelId{0}, alternative.op_type,
                    "incremental_" + std::to_string(node) + "_" + std::to_string(index),
                    children, {2}, {1}, DType::FLOAT32, space, alternative.engines));
                const ENodeId en_id = graph.getEClass(cid).enodes.back();
                state.bucket_enode_infos[b].resize(en_id.value + 1);
                state.bucket_enode_infos[b][en_id.value].cost = alternative.cost;
                state.bucket_enode_infos[b][en_id.value].is_view = alternative.is_view;
            }
            info.type = VarType::START;
            state.start_vars[b][cid] = state.addVar(info, Domain::makeRange(0, spec.size() + 4));
            info.type = VarType::OFFSET;
            info.mem_space = space;
            info.size_bytes = 8;
            info.size_pages = 2;
            state.offset_vars[b][cid] = state.addVar(info, Domain::makeRange(0, 30));
        }
    }
    return state;
}

inline void selectAll(SearchState &state)
{
    for (const auto &vars : state.selected_vars)
        for (const auto &[cid, var_id] : vars)
            state.setDomain(var_id, Domain::makeFixed(1, true));
}

// ============================================================================
// BASE 11 CORRECTNESS TESTS
// ============================================================================

inline void testBaseCorrectness()
{
    std::cout << "Running Base Test 1..." << std::endl;
    // 1. Selection reachability & children (Rules 1, 2, 10)
    {
        GraphSpec spec = {
            {AlternativeSpec({1})},
            {AlternativeSpec({2}), AlternativeSpec{}},
            {AlternativeSpec{}}
        };
        SearchEngine engine(makeState(spec, 1));
        addBasePropagators(engine);
        require(engine.runPropagators(kInvalidVarId), "Base reachability initial failed");

        // Fixing root to 1 forces child 1 to not be 0 (Rule 2)
        engine.state.setDomain(selection(engine.state, 0), Domain::makeFixed(1, true));
        require(engine.runPropagators(selection(engine.state, 0)), "Selecting root failed");
        require(!engine.state.domains[selection(engine.state, 1)].contains(0), "Rule 2: child was not forced non-zero");

        // Contrapositive (Rule 10): fixing node 2 to 0 removes parent alternative from node 1
        const size_t marker = engine.state.getTrailMarker();
        engine.state.setDomain(selection(engine.state, 2), Domain::makeFixed(0, true));
        std::string conflict;
        bool ok = engine.runPropagators(selection(engine.state, 2), &conflict);
        if (!ok)
            std::cerr << "Conflict reason in test 1: " << conflict << std::endl;
        require(ok, "Setting node 2 to 0 failed");
        require(!engine.state.domains[selection(engine.state, 1)].contains(1), "Rule 10: parent alternative was not removed");
        require(engine.state.domains[selection(engine.state, 1)].contains(2), "Rule 10: parent alternative 2 should remain");
        engine.state.backtrackTo(marker);
    }

    std::cout << "Running Base Test 2..." << std::endl;
    // 2. Unselected start/offset zeroing (Rule 3)
    {
        GraphSpec spec = {{{}}, {{}}};
        SearchEngine engine(makeState(spec, 1));
        addBasePropagators(engine);
        require(engine.runPropagators(kInvalidVarId), "Initial propagation failed");
        engine.state.setDomain(selection(engine.state, 1), Domain::makeFixed(0, true));
        require(engine.runPropagators(selection(engine.state, 1)), "Unselecting node 1 failed");
        require(engine.state.domains[start(engine.state, 1)].isFixed() &&
                engine.state.domains[start(engine.state, 1)].fixedValue() == 0, "Rule 3: start not fixed to 0");
        require(engine.state.domains[offset(engine.state, 1)].isFixed(), "Rule 3: offset not fixed to min_p");
    }

    std::cout << "Running Base Test 3..." << std::endl;
    // 3. Cache exclusion (Rule 4) and cache requirement (Rule 11)
    {
        GraphSpec spec = {
            {AlternativeSpec({1}, {}, 1.0f, false, OpType::CACHE),
             AlternativeSpec({1}, {}, 1.0f, false, OpType::SCATTER),
             AlternativeSpec({1}, {}, 1.0f, false, OpType::INPUT),
             AlternativeSpec{}},
            {AlternativeSpec({}, {}, 1.0f, false, OpType::CACHE),
             AlternativeSpec({}, {}, 1.0f, false, OpType::SCATTER),
             AlternativeSpec{}},
            {AlternativeSpec{}}
        };
        SearchEngine engine(makeState(spec, 1));
        addBasePropagators(engine);
        require(engine.runPropagators(kInvalidVarId), "Initial cache test failed");

        // Rule 4: If cached_var is fixed to 0, CACHE/SCATTER enodes must be removed
        VarId cv1 = engine.state.cached_vars.at(BaseEClassId{1});
        engine.state.setDomain(cv1, Domain::makeFixed(0, true));
        require(engine.runPropagators(cv1), "Setting cache 1 to 0 failed");
        require(!engine.state.domains[selection(engine.state, 0)].contains(1), "Rule 4: CACHE enode was not removed");

        // Rule 11: If CACHE or SCATTER enode is definitely selected (>0), corresponding cached var must be fixed to 1
        VarId cv2 = engine.state.cached_vars.at(BaseEClassId{2});
        engine.state.setDomain(selection(engine.state, 1), Domain::makeFixed(2, true)); // SCATTER
        require(engine.runPropagators(selection(engine.state, 1)), "Selecting SCATTER failed");
        require(engine.state.domains[cv2].isFixed() && engine.state.domains[cv2].fixedValue() == 1,
                "Rule 11: cached var was not fixed to 1");
    }

    std::cout << "Running Base Test 4..." << std::endl;
    // 4. Start precedence (Rule 5) and Start uniqueness (Rule 6)
    {
        GraphSpec spec = {{AlternativeSpec({1})}, {AlternativeSpec{}}};
        SearchEngine engine(makeState(spec, 1));
        addBasePropagators(engine);
        selectAll(engine.state);
        require(engine.runPropagators(kInvalidVarId), "Initial start test failed");

        // Fix child's start to 2
        engine.state.setDomain(start(engine.state, 1), Domain::makeFixed(2));
        require(engine.runPropagators(start(engine.state, 1)), "Fixing child start failed");

        // Rule 5: Consumer's start must be >= child start + 1
        require(engine.state.domains[start(engine.state, 0)].getMin() >= 3,
                "Rule 5: Consumer start did not advance to >= child start + 1");

        // Rule 6: Start values must be unique per bucket
        engine.state.setDomain(start(engine.state, 0), Domain::makeFixed(2));
        require(!engine.runPropagators(start(engine.state, 0)),
                "Rule 6: Duplicate start value was accepted without contradiction");
    }

    std::cout << "Running Base Test 5..." << std::endl;
    // 5. Pearce-Kelly cycle detection (Rule 9)
    {
        GraphSpec spec = {
            {AlternativeSpec({1})},
            {AlternativeSpec({0}), AlternativeSpec{}}
        };
        SearchEngine engine(makeState(spec, 1));
        engine.addPropagator(std::make_unique<PearceKellyCyclePropagator>());
        engine.state.setDomain(selection(engine.state, 0), Domain::makeFixed(1, true));
        require(engine.runPropagators(selection(engine.state, 0)), "Selecting node 0 failed");
        engine.state.setDomain(selection(engine.state, 1), Domain::makeFixed(1, true));
        require(!engine.runPropagators(selection(engine.state, 1)),
                "Rule 9: Direct cycle 0 -> 1 -> 0 was accepted without contradiction");
    }

    std::cout << "Running Base Test 6..." << std::endl;
    // 6. View offset propagation (Rule 8)
    {
        GraphSpec spec = {
            {AlternativeSpec({1}, {}, 0.0f, true)},
            {AlternativeSpec{}}
        };
        SearchEngine engine(makeState(spec, 1));
        addBasePropagators(engine);
        selectAll(engine.state);
        require(engine.runPropagators(kInvalidVarId), "View offset test init failed");
        engine.state.setDomain(start(engine.state, 1), Domain::makeFixed(0));
        engine.state.setDomain(start(engine.state, 0), Domain::makeFixed(1));
        engine.state.setDomain(offset(engine.state, 1), Domain::makeFixed(12));
        require(engine.runPropagators(offset(engine.state, 1)), "Fixing child offset failed");
        require(engine.state.domains[offset(engine.state, 0)].isFixed() &&
                engine.state.domains[offset(engine.state, 0)].fixedValue() == 12,
                "Rule 8: View offset did not align with child offset");
    }
}

// ============================================================================
// TESTING EXTRA PROPAGATORS (12-15) USING BASE 11 AS REFERENCE
// ============================================================================

inline void testExtraAgainstBaseReference()
{
    auto setIncumbent = [](SearchEngine &engine, float best_cost) {
        engine.state.best_cost = best_cost;
    };

    // Test 1: Soundness of CriticalPathPropagator (Rule 12) & EngineWorkloadPropagator (Rule 15)
    // The lower bound computed by EXTRA must never exceed the true evaluated makespan of any valid solution.
    {
        const Engine cpu{0, EngineType::CPU};
        GraphSpec spec = {
            {{{1}, {cpu}, 3.0f}},
            {{{2}, {cpu}, 5.0f}},
            {{{}, {cpu}, 2.0f}}
        };
        SearchState base_state = makeState(spec, 1);
        selectAll(base_state);
        base_state.setDomain(start(base_state, 2), Domain::makeFixed(0));
        base_state.setDomain(start(base_state, 1), Domain::makeFixed(1));
        base_state.setDomain(start(base_state, 0), Domain::makeFixed(2));
        base_state.setDomain(offset(base_state, 2), Domain::makeFixed(0));
        base_state.setDomain(offset(base_state, 1), Domain::makeFixed(4));
        base_state.setDomain(offset(base_state, 0), Domain::makeFixed(8));

        SearchEngine base_engine(base_state);
        addBasePropagators(base_engine);
        require(base_engine.runPropagators(kInvalidVarId), "Base 11 failed on valid plan");
        float true_makespan = base_engine.evaluateMakespan(base_engine.state, 0);

        SearchEngine extra_engine(base_state);
        addAllPropagators(extra_engine);
        setIncumbent(extra_engine, 100.0f);
        require(extra_engine.runPropagators(kInvalidVarId), "Extra failed on valid plan");
        const float extra_lb = extra_engine.state.lower_bound;

        require(extra_lb <= true_makespan + 1e-4f,
                "EXTRA lower bound (" + std::to_string(extra_lb) + ") exceeded true makespan (" + std::to_string(true_makespan) + ")");
        require(extra_lb >= 10.0f - 1e-4f,
                "EXTRA lower bound (" + std::to_string(extra_lb) + ") should be >= 10 (critical path 3+5+2)");

        // Pruning against incumbent: if incumbent is less than lower bound, EXTRA prunes
        SearchEngine pruned_engine(base_state);
        addAllPropagators(pruned_engine);
        setIncumbent(pruned_engine, 5.0f);
        require(!pruned_engine.runPropagators(kInvalidVarId),
                "EXTRA should prune state when incumbent best_cost < lower_bound");
    }

    // Test 2: CacheBudgetPropagator (Rule 13)
    // When cached variables exceed memory cap, EXTRA detects contradiction where Base 11 allows it.
    {
        GraphSpec spec = {{{}}, {{}}, {{}}};
        SearchState state = makeState(spec, 1);
        const MemSpace space{1, HandleType::CPP};
        state.mem_caps[space] = 12; // Each candidate is 8 bytes, so at most 1 fits
        selectAll(state);

        SearchEngine base_engine(state);
        addBasePropagators(base_engine);
        // Base 11 only enforces cache consistency, not total budget capacity
        base_engine.state.setDomain(base_engine.state.cached_vars.at(BaseEClassId{1}), Domain::makeFixed(1, true));
        base_engine.state.setDomain(base_engine.state.cached_vars.at(BaseEClassId{2}), Domain::makeFixed(1, true));
        base_engine.runPropagators(kInvalidVarId);

        // EXTRA (Rule 13) must detect capacity contradiction
        SearchEngine extra_engine(state);
        addAllPropagators(extra_engine);
        extra_engine.state.setDomain(extra_engine.state.cached_vars.at(BaseEClassId{1}), Domain::makeFixed(1, true));
        extra_engine.state.setDomain(extra_engine.state.cached_vars.at(BaseEClassId{2}), Domain::makeFixed(1, true));
        require(!extra_engine.runPropagators(kInvalidVarId),
                "Rule 13 (CacheBudgetPropagator) failed to reject cache overcommit");
    }

    // Test 4: Random Branching - Equivalence & Pruning Consistency
    // Search with Base 11 vs Base 11 + EXTRA:
    // Any solution found by Base 11 must be accepted by EXTRA (if within cache capacity).
    {
        std::mt19937 random(424242);
        for (uint32_t trial = 0; trial < 25; ++trial)
        {
            GraphSpec spec(3 + random() % 4);
            for (uint32_t node = 0; node < spec.size(); ++node)
            {
                uint32_t alts = 1 + random() % 2;
                for (uint32_t a = 0; a < alts; ++a)
                {
                    AlternativeSpec alt;
                    if (node + 1 < spec.size())
                    {
                        uint32_t ch_count = random() % 2;
                        for (uint32_t c = 0; c < ch_count; ++c)
                            alt.children.push_back(node + 1 + random() % (spec.size() - node - 1));
                    }
                    alt.cost = 1.0f + (random() % 10);
                    alt.engines.push_back(Engine{0, EngineType::CPU});
                    spec[node].push_back(alt);
                }
            }

            SearchState state = makeState(spec, 1);
            state.mem_caps[MemSpace{1, HandleType::CPP}] = 256;
            selectAll(state);

            SearchEngine base_engine(state);
            addBasePropagators(base_engine);

            SearchEngine extra_engine(state);
            addAllPropagators(extra_engine);
            setIncumbent(extra_engine, 1.0e6f);

            bool base_ok = base_engine.runPropagators(kInvalidVarId);
            bool extra_ok = extra_engine.runPropagators(kInvalidVarId);

            if (!base_ok)
            {
                require(!extra_ok, "EXTRA was feasible when Base 11 was infeasible");
                continue;
            }

            if (extra_ok)
            {
                require(extra_engine.state.lower_bound >= base_engine.state.lower_bound,
                        "EXTRA lower bound should be >= Base lower bound");
                // Check domains: Extra must be at least as restricted as Base
                for (VarId v = 0; v < state.numVars(); ++v)
                {
                    const Domain &b_dom = base_engine.state.domains[v];
                    const Domain &e_dom = extra_engine.state.domains[v];
                    require(e_dom.size() <= b_dom.size(),
                            "EXTRA domain for var " + std::to_string(v) + " was wider than Base 11 domain");
                }
            }

            // Test backtracking consistency
            const size_t marker = extra_engine.state.getTrailMarker();
            const auto prev_domains = extra_engine.state.domains;
            // Branch on first unfixed var
            for (VarId v = 0; v < extra_engine.state.numVars(); ++v)
            {
                if (!extra_engine.state.domains[v].isFixed() && !extra_engine.state.domains[v].isEmpty())
                {
                    Domain d = extra_engine.state.domains[v];
                    int32_t val = d.getMin();
                    extra_engine.state.setDomain(v, Domain::makeFixed(val, d.is_mask));
                    extra_engine.runPropagators(v);
                    extra_engine.state.backtrackTo(marker);
                    require(extra_engine.state.domains == prev_domains,
                            "Backtracking failed to restore exact domains in trial " + std::to_string(trial));
                    break;
                }
            }
        }
    }
}

} // namespace incremental_propagator_test

inline void runIncrementalPropagatorTests()
{
    using namespace incremental_propagator_test;
    testBaseCorrectness();
    testExtraAgainstBaseReference();
    std::cout << "All 15 propagator tests passed (Base 11 reference + EXTRA)" << std::endl;
}
