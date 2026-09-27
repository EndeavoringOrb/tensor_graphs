#pragma once

#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>

#include "tests/full_propagators.hpp"
#include "tests/selection_reachability.hpp"

namespace incremental_propagator_test
{
using namespace plan;

enum Checks : uint32_t
{
    CACHE = 1, TOPOLOGY = 2, ENGINE = 4, MEMORY = 8, COST = 16, SELECTION = 32, ALL = 63
};

inline void require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

struct AlternativeSpec
{
    std::vector<uint32_t> children;
    std::vector<Engine> engines;
    float cost = 1.0f;
    bool is_view = false;
    OpType op_type = OpType::INPUT;
};
using GraphSpec = std::vector<std::vector<AlternativeSpec>>;

inline VarId selection(const SearchState &state, uint32_t node, uint32_t bucket = 0)
{
    return state.selected_vars[bucket].at(EClassId{node});
}

inline VarId start(const SearchState &state, uint32_t node, uint32_t bucket = 0, uint32_t alternative = 0)
{
    return state.start_vars[bucket].at(EClassId{node})[alternative];
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
                info.type = VarType::START;
                info.enode_idx = index;
                state.start_vars[b][cid].push_back(state.addVar(info, Domain::makeRange(0, spec.size() + 4)));
            }
            info.type = VarType::OFFSET;
            info.mem_space = space;
            info.size_bytes = 8;
            info.size_pages = 2;
            state.offset_vars[b][cid] = state.addVar(info, Domain::makeRange(0, 30));
        }
    }
    return state;
}

inline void addPropagators(SearchEngine &engine, uint32_t checks)
{
    if (checks & SELECTION) engine.addPropagator(std::make_unique<SelectionPropagator>());
    if (checks & CACHE) engine.addPropagator(std::make_unique<CachePropagator>());
    if (checks & TOPOLOGY) engine.addPropagator(std::make_unique<TopologicalOrderPropagator>());
    if (checks & ENGINE) engine.addPropagator(std::make_unique<EngineSchedulePropagator>());
    if (checks & MEMORY) engine.addPropagator(std::make_unique<MemoryNonOverlapPropagator>());
    if (checks & COST) engine.addPropagator(std::make_unique<CostLowerBoundPropagator>());
}

inline bool propagateFull(SearchState &state, uint32_t checks, float &bound)
{
    FullCachePropagator cache;
    FullTopologicalOrderPropagator topology;
    FullEngineSchedulePropagator engine;
    FullMemoryNonOverlapPropagator memory;
    FullCostLowerBoundPropagator cost;
    cost.setBestCost(state.best_cost);
    std::vector<VarId> unused;
    for (uint32_t round = 0; round < 1000; ++round)
    {
        const auto before = state.domains;
        if (state.hasEmptyDomain()) return false;
        if ((checks & SELECTION) && !selection_reachability_test::propagateReference(state)) return false;
        if ((checks & CACHE) && !cache.propagate(state, 0, unused)) return false;
        if ((checks & TOPOLOGY) && !topology.propagate(state, 0, unused)) return false;
        if ((checks & ENGINE) && !engine.propagate(state, 0, unused)) return false;
        if ((checks & MEMORY) && !memory.propagate(state, 0, unused)) return false;
        if ((checks & COST) && !cost.propagate(state, 0, unused)) return false;
        if (state.hasEmptyDomain()) return false;
        if (before == state.domains)
        {
            bound = (checks & COST) ? cost.computeLowerBound(state) : 0.0f;
            return true;
        }
    }
    throw std::runtime_error("Full propagators did not converge");
}

inline void checkIndexes(const SearchState &state)
{
    state.ensurePropagationState();
    SearchState rebuilt = state;
    rebuilt.propagation = PropagationState{};
    rebuilt.ensurePropagationState();
    require(state.costLowerBound() == rebuilt.costLowerBound(), "Incremental cost differs from rebuilt sum tree");
    const auto &actual = state.propagation;
    const auto &expected = rebuilt.propagation;
    for (const auto &[space, indices] : actual.candidates_by_space)
        require(actual.fixed_cache_bytes.count(space) == 0 ? expected.fixed_cache_bytes.count(space) == 0 || expected.fixed_cache_bytes.at(space) == 0 :
                actual.fixed_cache_bytes.at(space) == (expected.fixed_cache_bytes.count(space) ? expected.fixed_cache_bytes.at(space) : 0),
                "Cache budget was not restored");
    for (size_t i = 0; i < actual.engines.size(); ++i)
    {
        require(actual.engines[i].active_starts == expected.engines[i].active_starts, "Active engine users were not restored");
        require(actual.engines[i].fixed_starts == expected.engines[i].fixed_starts, "Engine occupancy was not restored");
    }
    for (size_t b = 0; b < actual.buckets.size(); ++b)
        require(actual.buckets[b].finish_times == expected.buckets[b].finish_times, "Finish times were not restored");
}

inline bool checkPropagation(SearchEngine &engine, uint32_t checks, const std::string &context = "")
{
    SearchState reference = engine.state;
    // Selection's reference writes domains directly. Keep the full oracle
    // independent of all production indexes and its dirty-domain bookkeeping.
    reference.propagation = PropagationState{};
    float expected_bound = 0.0f;
    const bool expected = propagateFull(reference, checks, expected_bound);
    float actual_bound = 0.0f;
    const bool actual = engine.runPropagators(actual_bound, kInvalidVarId);
    require(actual == expected, context + ": incremental/full feasibility differs (checks=" + std::to_string(checks) + ")");
    if (actual)
    {
        for (VarId var_id = 0; var_id < engine.state.numVars(); ++var_id)
            require(engine.state.domains[var_id] == reference.domains[var_id], context + ": domain differs for var " +
                    std::to_string(var_id) + ": " + engine.state.domains[var_id].toString() + " vs " + reference.domains[var_id].toString());
        require(std::abs(actual_bound - expected_bound) <= 1e-5f * std::max(1.0f, expected_bound), "Cost bound differs from full summation");
    }
    checkIndexes(engine.state);
    return actual;
}

inline void selectAll(SearchState &state)
{
    for (const auto &vars : state.selected_vars)
        for (const auto &[cid, var_id] : vars)
            state.setDomain(var_id, Domain::makeFixed(1, true));
}

inline void testCacheDependencies()
{
    GraphSpec spec(3, {{{}, {}, 1.0f, false, OpType::CACHE}, {{}, {}, 1.0f, false, OpType::SCATTER}, {}});
    SearchState state = makeState(spec, 2);
    state.mem_caps[MemSpace{1, HandleType::CPP}] = 12;
    state.candidates[2].size_bytes = 16;
    SearchEngine engine(std::move(state));
    addPropagators(engine, CACHE);
    require(checkPropagation(engine, CACHE), "Initial cache propagation failed");
    const auto initial = engine.state.domains;
    const size_t marker = engine.state.getTrailMarker();
    engine.state.setDomain(selection(engine.state, 0, 1), Domain::makeFixed(1, true));
    require(checkPropagation(engine, CACHE), "Cross-bucket cache requirement failed");
    require(engine.state.domains[engine.state.cached_vars.at(BaseEClassId{1})].fixedValue() == 1, "CACHE did not require caching");
    require(engine.state.domains[selection(engine.state, 1)].mask == ((1u << 0) | (1u << 3)), "Budget did not prune CACHE and SCATTER alternatives");
    engine.state.backtrackTo(marker);
    require(engine.state.domains == initial && checkPropagation(engine, CACHE), "Cache rollback failed");
    engine.state.setDomain(engine.state.cached_vars.at(BaseEClassId{1}), Domain::makeFixed(1, true));
    engine.state.setDomain(engine.state.cached_vars.at(BaseEClassId{2}), Domain::makeFixed(1, true));
    require(!checkPropagation(engine, CACHE), "Overcommitted budget was accepted");
    engine.state.backtrackTo(marker);
    require(checkPropagation(engine, CACHE), "Cache sibling after contradiction failed");
}

inline void testTopologyAndEngines()
{
    const Engine cpu{0, EngineType::CPU};
    const Engine dma{0, EngineType::CUDA_DMA};
    GraphSpec spec(8);
    spec[0] = {{{}, {cpu}}, {{7}, {cpu}}};
    for (uint32_t node = 1; node < spec.size(); ++node)
        spec[node] = {{{node - 1, node - 1}, {cpu}}};
    SearchEngine topology(makeState(spec, 2));
    addPropagators(topology, TOPOLOGY);
    for (uint32_t node = 1; node < spec.size(); ++node)
        topology.state.setDomain(selection(topology.state, node), Domain::makeFixed(1, true));
    require(checkPropagation(topology, TOPOLOGY), "Fixed path pruning failed");
    require(!topology.state.domains[selection(topology.state, 0)].contains(2), "Transitive cycle alternative survived");
    topology.state.setDomain(selection(topology.state, 0), Domain::makeFixed(1, true));
    require(checkPropagation(topology, TOPOLOGY), "Dependency chain failed");
    require(topology.state.domains[start(topology.state, 7)].getMin() == 7, "Start bounds did not reach end of chain");

    spec = {{{{}, {cpu}}}, {{{}, {dma}}}, {{{}, {cpu, dma}}}, {{{}, {cpu}}, {{}, {dma}}}};
    SearchEngine engine(makeState(spec, 2));
    addPropagators(engine, ENGINE);
    for (uint32_t node = 0; node < 3; ++node)
        engine.state.setDomain(selection(engine.state, node), Domain::makeFixed(1, true));
    engine.state.setDomain(start(engine.state, 0), Domain::makeFixed(0));
    engine.state.setDomain(start(engine.state, 1), Domain::makeFixed(1));
    engine.state.setDomain(start(engine.state, 3, 0, 1), Domain::makeFixed(0));
    require(checkPropagation(engine, ENGINE), "Engine prefix pruning failed");
    require(engine.state.domains[start(engine.state, 2)].getMin() == 2, "Multi-engine blocked prefix survived");
    const size_t marker = engine.state.getTrailMarker();
    engine.state.setDomain(selection(engine.state, 3), Domain::makeFixed(2, true));
    require(checkPropagation(engine, ENGINE), "Selecting a previously fixed inactive start failed");
    engine.state.backtrackTo(marker);
    engine.state.setDomain(start(engine.state, 2), Domain::makeFixed(1));
    require(!checkPropagation(engine, ENGINE), "Engine collision was accepted");
    engine.state.backtrackTo(marker);
    require(checkPropagation(engine, ENGINE), "Engine rollback failed");
}

inline void testMemoryViewsAndLifetimes()
{
    GraphSpec spec = {{{}}, {{{0}, {}, 0.0f, true}}, {{{1}, {}, 0.0f, true}}, {{{2}}}, {{}}};
    SearchState state = makeState(spec, 2);
    selectAll(state);
    for (uint32_t b = 0; b < 2; ++b)
    {
        for (uint32_t node = 0; node < 5; ++node)
            state.setDomain(start(state, node, b), Domain::makeFixed(node == 3 ? 4 : node == 4 ? 3 : node));
        state.setDomain(offset(state, 0, b), Domain::makeFixed(0));
        state.setDomain(offset(state, 3, b), Domain::makeFixed(10));
    }
    SearchEngine engine(std::move(state));
    addPropagators(engine, MEMORY);
    require(checkPropagation(engine, MEMORY), "View lifetime propagation failed");
    require(engine.state.domains[offset(engine.state, 1)].fixedValue() == 0 &&
            engine.state.domains[offset(engine.state, 2)].fixedValue() == 0, "View chain offsets did not align");
    require(engine.state.domains[offset(engine.state, 4)].getMin() == 2, "Consumer through views did not extend input lifetime");
    const size_t marker = engine.state.getTrailMarker();
    const auto initial = engine.state.domains;
    engine.state.setDomain(offset(engine.state, 4), Domain::makeFixed(10));
    require(!checkPropagation(engine, MEMORY), "Live allocation collision was accepted");
    engine.state.backtrackTo(marker);
    require(engine.state.domains == initial && checkPropagation(engine, MEMORY), "Memory rollback failed");
    engine.state.setDomain(offset(engine.state, 4), Domain::makeFixed(2));
    require(checkPropagation(engine, MEMORY), "Memory sibling after collision failed");

    // A preallocated input uses its physical offset even with a wide search domain.
    state = makeState({{{}}, {{{0}}}, {{}}});
    selectAll(state);
    state.preallocated_buffers[BaseEClassId{1}].offset = 0;
    state.preallocated_buffers[BaseEClassId{1}].size = 8;
    state.setDomain(start(state, 0), Domain::makeFixed(0));
    state.setDomain(start(state, 2), Domain::makeFixed(1));
    SearchEngine preallocated(std::move(state));
    addPropagators(preallocated, MEMORY);
    require(checkPropagation(preallocated, MEMORY), "Preallocated memory propagation failed");
    preallocated.state.setDomain(start(preallocated.state, 1), Domain::makeFixed(5));
    require(checkPropagation(preallocated, MEMORY), "Open lifetime did not track new maximum finish");
    require(preallocated.state.domains[offset(preallocated.state, 2)].getMin() == 2, "Physical input offset was ignored");
}

inline void testCostAndStateOwnership()
{
    const Engine cpu{0, EngineType::CPU};
    const Engine dma{0, EngineType::CUDA_DMA};
    SearchState state = makeState({{{{}, {cpu}, 0.1f}}, {{{}, {cpu, dma}, 0.2f}}, {{{}, {dma}, TGConstants::INF}}}, 3);
    state.bucket_weights[1] = 0.0f;
    state.bucket_weights[2] = -1.0f;
    SearchEngine engine(std::move(state));
    addPropagators(engine, COST);
    const size_t before_initialization = engine.state.getTrailMarker();
    selectAll(engine.state);
    require(checkPropagation(engine, COST), "Initial cost propagation failed");
    const float bound = engine.state.costLowerBound();
    require(std::abs(bound - 0.3f) < 1e-6f, "Weights or engine cost aggregation are incorrect");
    SearchEngine copied(engine.state);
    addPropagators(copied, COST);
    copied.state.best_cost = bound;
    require(!checkPropagation(copied, COST), "Equal incumbent did not prune without domain changes");
    require(engine.state.best_cost == TGConstants::INF && checkPropagation(engine, COST), "Copied incumbent affected original state");
    engine.state.backtrackTo(before_initialization);
    require(checkPropagation(engine, COST) && engine.state.costLowerBound() == 0.0f, "Undo before initialization failed");
    for (uint32_t repeat = 0; repeat < 100; ++repeat)
    {
        const size_t marker = engine.state.getTrailMarker();
        selectAll(engine.state);
        require(checkPropagation(engine, COST) && engine.state.costLowerBound() == bound, "Cost update drifted");
        engine.state.backtrackTo(marker);
        require(engine.state.costLowerBound() == 0.0f, "Cost rollback drifted");
    }
    // Reuse the same stateless objects across independent states.
    CostLowerBoundPropagator propagator;
    std::vector<VarId> worklist;
    require(propagator.propagate(engine.state, 0, worklist) && !propagator.propagate(copied.state, 0, worklist),
            "Propagator retained another search's incumbent");
}

inline void testInterruptedInitializationAndReplay()
{
    // Fail a view equality before all initial allocations have been examined,
    // then undo only the failing decision and finish the remaining work.
    SearchState state = makeState({{{}}, {{{0}, {}, 0.0f, true}}, {{}}, {{}}}, 2);
    selectAll(state);
    for (uint32_t b = 0; b < 2; ++b)
    {
        for (uint32_t node = 0; node < 4; ++node)
            state.setDomain(start(state, node, b), Domain::makeFixed(node));
        state.setDomain(offset(state, 0, b), Domain::makeFixed(0));
        state.setDomain(offset(state, 2, b), Domain::makeFixed(8));
    }
    SearchEngine engine(std::move(state));
    addPropagators(engine, MEMORY);
    const size_t marker = engine.state.getTrailMarker();
    engine.state.setDomain(offset(engine.state, 1), Domain::makeFixed(10));
    require(!checkPropagation(engine, MEMORY), "Conflicting view offset was accepted");
    engine.state.backtrackTo(marker);
    require(checkPropagation(engine, MEMORY), "Rollback lost pending initialization");

    const Engine cpu{0, EngineType::CPU};
    SearchEngine replay(makeState({{{{}, {cpu}, 0.1f}}, {{{}, {cpu}, 0.2f}}}));
    addPropagators(replay, ENGINE | COST);
    selectAll(replay.state);
    require(checkPropagation(replay, ENGINE | COST), "Replay fixture did not propagate");
    const size_t root_marker = replay.state.getTrailMarker();
    const auto root_domains = replay.state.domains;
    auto root = std::make_shared<SearchNode>(0, UINT32_MAX,
        std::make_pair(kInvalidVarId, Domain{}), 0.0f, 0.0f, 0, root_marker);
    auto first = std::make_shared<SearchNode>(1, 0,
        std::make_pair(start(replay.state, 0), Domain::makeFixed(0)), 0.0f, 0.0f, 1);
    auto failed = std::make_shared<SearchNode>(2, 1,
        std::make_pair(start(replay.state, 1), Domain::makeFixed(0)), 0.0f, 0.0f, 2);
    auto sibling = std::make_shared<SearchNode>(3, 1,
        std::make_pair(start(replay.state, 1), Domain::makeFixed(1)), 0.0f, 0.0f, 2);
    replay.all_nodes = {root, first, failed, sibling};
    replay.current_node_id = 0;
    require(!replay.restoreNode(failed), "Conflicting replay was accepted");
    require(replay.restoreNode(sibling) && checkPropagation(replay, ENGINE | COST), "Sibling replay failed");
    require(replay.restoreNode(root) && replay.state.domains == root_domains && checkPropagation(replay, ENGINE | COST),
            "LCA rollback did not restore incremental state");
    require(replay.restoreNode(sibling) && checkPropagation(replay, ENGINE | COST), "Repeated replay failed");
}

inline void testMemoryPoolsAndBounds()
{
    SearchState state = makeState({{{}}, {{}}, {{}}});
    selectAll(state);
    const MemSpace other_space{2, HandleType::CPP};
    state.bucket_egraphs[0].getEClass(EClassId{2}).mem_space = other_space;
    state.page_alignments[other_space] = 4;
    for (uint32_t node = 0; node < 3; ++node)
        state.setDomain(start(state, node), Domain::makeFixed(node));
    state.setDomain(offset(state, 0), Domain::makeFixed(8));
    state.setDomain(offset(state, 2), Domain::makeFixed(8));
    SearchEngine engine(std::move(state));
    addPropagators(engine, MEMORY);
    require(checkPropagation(engine, MEMORY), "Separate memory pools conflicted");
    const size_t marker = engine.state.getTrailMarker();
    Domain domain = engine.state.domains[offset(engine.state, 1)];
    domain.setMax(9);
    engine.state.setDomain(offset(engine.state, 1), domain);
    require(checkPropagation(engine, MEMORY) && engine.state.domains[offset(engine.state, 1)].getMax() == 6,
            "Blocked offset suffix was not pruned");
    engine.state.backtrackTo(marker);
    domain = engine.state.domains[offset(engine.state, 1)];
    domain.setMin(7);
    engine.state.setDomain(offset(engine.state, 1), domain);
    require(checkPropagation(engine, MEMORY) && engine.state.domains[offset(engine.state, 1)].getMin() == 10,
            "Blocked offset prefix was not pruned");
}

inline void testRandomBranches()
{
    std::mt19937 random(942731);
    for (uint32_t checks : {uint32_t(CACHE), uint32_t(TOPOLOGY), uint32_t(ENGINE), uint32_t(MEMORY), uint32_t(COST), uint32_t(ALL)})
        for (uint32_t trial = 0; trial < 60; ++trial)
        {
            GraphSpec spec(3 + random() % 5);
            for (uint32_t node = 0; node < spec.size(); ++node)
                for (uint32_t index = 0, count = 1 + random() % 3; index < count; ++index)
                {
                    AlternativeSpec alternative;
                    for (uint32_t child = 0, count = random() % 3; child < count; ++child)
                    {
                        if (checks == MEMORY || (checks == ALL && trial % 2 == 0))
                        {
                            if (node + 1 < spec.size())
                                alternative.children.push_back(node + 1 + random() % (spec.size() - node - 1));
                        }
                        else
                            alternative.children.push_back(random() % spec.size());
                    }
                    if (random() % 4 != 0)
                        alternative.engines.push_back(Engine{random() % 2, EngineType::CPU});
                    if (random() % 4 == 0)
                        alternative.engines.push_back(Engine{0, EngineType::CUDA_DMA});
                    alternative.is_view = !alternative.children.empty() && random() % 4 == 0;
                    alternative.cost = alternative.is_view ? 0.0f : (random() % 31) / 10.0f;
                    if (random() % 5 == 0)
                        alternative.op_type = random() % 2 ? OpType::CACHE : OpType::SCATTER;
                    spec[node].push_back(alternative);
                }
            SearchState state = makeState(spec, 2);
            state.mem_caps[MemSpace{1, HandleType::CPP}] = 8 * (1 + random() % spec.size());
            // Exercise initial fixed domains and batched changes before any index exists.
            for (const auto &vars : state.selected_vars)
                for (const auto &[cid, var_id] : vars)
                    if (random() % 3 == 0)
                        state.setDomain(var_id, Domain::makeFixed(1, true));
            if (checks == MEMORY || checks == ENGINE || (checks == ALL && trial % 2 == 0))
                for (uint32_t b = 0; b < state.buckets.size(); ++b)
                    for (uint32_t node = 0; node < spec.size(); ++node)
                    {
                        if (random() % 3 != 0)
                            state.setDomain(selection(state, node, b), Domain::makeFixed(1, true));
                        for (uint32_t index = 0; index < spec[node].size(); ++index)
                            if (random() % 2 == 0)
                                state.setDomain(start(state, node, b, index), Domain::makeFixed(spec.size() - node));
                        // Views inherit their source's offset; leave them free initially.
                        if (random() % 3 == 0 && !spec[node][0].is_view)
                            state.setDomain(offset(state, node, b), Domain::makeFixed(4 * node));
                    }
            SearchEngine engine(std::move(state));
            addPropagators(engine, checks);
            std::function<void(uint32_t)> visit = [&](uint32_t depth) {
                const std::string context = "trial=" + std::to_string(trial) + " depth=" + std::to_string(depth);
                if (!checkPropagation(engine, checks, context) || depth == 5)
                    return;
                std::vector<VarId> choices;
                for (VarId var_id = 0; var_id < engine.state.numVars(); ++var_id)
                    if (!engine.state.domains[var_id].isFixed()) choices.push_back(var_id);
                if (choices.empty()) return;
                const VarId var_id = choices[random() % choices.size()];
                const Domain original = engine.state.domains[var_id];
                Domain left = original;
                Domain right = original;
                if (original.is_mask)
                {
                    const int32_t value = original.getMin();
                    left = Domain::makeFixed(value, true);
                    right.remove(value);
                }
                else
                {
                    const int32_t middle = original.getMin() + (original.getMax() - original.getMin()) / 2;
                    left.setMax(middle);
                    right.setMin(middle + 1);
                }
                const auto parent_domains = engine.state.domains;
                const float parent_bound = engine.state.costLowerBound();
                const size_t marker = engine.state.getTrailMarker();
                if (depth == 2 && trial % 10 == 0)
                {
                    SearchEngine copied(engine.state);
                    addPropagators(copied, checks);
                    copied.state.setDomain(var_id, left);
                    checkPropagation(copied, checks, context + " copy");
                    require(engine.state.domains == parent_domains && engine.state.costLowerBound() == parent_bound,
                            "Copied propagation modified the original state");
                }
                for (const Domain &branch : {left, right})
                {
                    engine.state.setDomain(var_id, branch);
                    visit(depth + 1);
                    engine.state.backtrackTo(marker);
                    require(engine.state.domains == parent_domains, "Random rollback changed parent domains");
                    require(engine.state.costLowerBound() == parent_bound, "Random rollback changed parent bound");
                    checkIndexes(engine.state);
                }
            };
            visit(0);
        }
}

} // namespace incremental_propagator_test

inline void runIncrementalPropagatorTests()
{
    using namespace incremental_propagator_test;
    testCacheDependencies();
    testTopologyAndEngines();
    testMemoryViewsAndLifetimes();
    testCostAndStateOwnership();
    testInterruptedInitializationAndReplay();
    testMemoryPoolsAndBounds();
    testRandomBranches();
    std::cout << "incremental propagator tests passed" << std::endl;
}
