// tensor_graphs_cpp/tests/start_run_makespan.hpp
#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/misc.hpp"
#include "core/plan/propagator.hpp"
#include "core/plan/propagators/start_run_makespan.hpp"
#include "core/plan/search_engine.hpp"

namespace start_run_makespan_test
{

using namespace plan;

inline void require(bool condition, const std::string &message)
{
    if (!condition)
        Error::throw_err(message);
}

struct NodeSpec
{
    std::vector<uint32_t> children;
    std::vector<Engine> engines;
    float cost = 1.0f;
    bool is_view = false;
    OpType op_type = OpType::ADD;

    NodeSpec() = default;
    NodeSpec(std::vector<uint32_t> ch, std::vector<Engine> eng = {}, float c = 1.0f, bool view = false, OpType op = OpType::ADD)
        : children(std::move(ch)), engines(std::move(eng)), cost(c), is_view(view), op_type(op) {}
};

// Alternative specifications for an EClass
using ClassSpec = std::vector<NodeSpec>;
using GraphSpec = std::vector<ClassSpec>;

struct StartRunTestCase
{
    std::string name;
    GraphSpec spec;
    std::vector<float> bucket_weights = {1.0f};
    std::vector<std::pair<uint32_t, int32_t>> fixed_selections;
    std::vector<std::pair<uint32_t, int32_t>> fixed_starts;
    std::vector<std::vector<std::pair<uint32_t, int32_t>>> completions;
    float expected_min_lb = 0.0f;
    float expected_exact_lb = -1.0f; // -1 if not checked
    float test_best_cost = -1.0f;
    bool expect_prune = false;
};

inline SearchState makeTestState(const GraphSpec &spec, const std::vector<float> &weights = {1.0f})
{
    SearchState state;
    const MemSpace space{1, HandleType::CPP};
    state.mem_caps[space] = 512;
    state.page_alignments[space] = 4;

    for (uint32_t node = 0; node < spec.size(); ++node)
    {
        VarInfo info;
        info.type = VarType::CACHED;
        info.base_eclass_id = BaseEClassId{node + 1};
        info.mem_space = space;
        info.size_bytes = 8;
        state.cached_vars[info.base_eclass_id] = state.addVar(info, Domain::makeMask(3));
        CacheCandidate candidate;
        candidate.base_eclass_id = info.base_eclass_id;
        candidate.mem_space = space;
        candidate.size_bytes = 8;
        state.candidates.push_back(candidate);
    }

    const uint32_t bucket_count = static_cast<uint32_t>(weights.size());
    for (uint32_t b = 0; b < bucket_count; ++b)
    {
        state.buckets.emplace_back();
        state.bucket_weights.push_back(weights[b]);
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
            VarInfo sel_info;
            sel_info.bucket_idx = b;
            sel_info.eclass_id = cid;
            sel_info.type = VarType::SELECTED;
            state.selected_vars[b][cid] = state.addVar(sel_info, Domain::makeMask((1u << (spec[node].size() + 1)) - 1));

            for (uint32_t index = 0; index < spec[node].size(); ++index)
            {
                const auto &node_spec = spec[node][index];
                std::vector<EClassId> children;
                for (uint32_t ch : node_spec.children)
                    children.push_back(EClassId{ch});

                graph.addENode(cid, ENode(KernelId{0}, node_spec.op_type,
                                         "node_" + std::to_string(node) + "_" + std::to_string(index),
                                         children, {2}, {1}, DType::FLOAT32, space, node_spec.engines));

                const ENodeId en_id = graph.getEClass(cid).enodes.back();
                state.bucket_enode_infos[b].resize(en_id.value + 1);
                state.bucket_enode_infos[b][en_id.value].cost = node_spec.cost;
                state.bucket_enode_infos[b][en_id.value].is_view = node_spec.is_view;
            }

            VarInfo st_info;
            st_info.bucket_idx = b;
            st_info.eclass_id = cid;
            st_info.type = VarType::START;
            st_info.selection_var = state.selected_vars[b][cid];
            state.start_vars[b][cid] = state.addVar(st_info, Domain::makeRange(0, static_cast<int32_t>(spec.size() + 10)));

            VarInfo off_info;
            off_info.bucket_idx = b;
            off_info.eclass_id = cid;
            off_info.type = VarType::OFFSET;
            off_info.mem_space = space;
            off_info.size_bytes = 8;
            off_info.size_pages = 2;
            state.offset_vars[b][cid] = state.addVar(off_info, Domain::makeRange(0, 64));
        }
    }
    return state;
}

inline void executeTestCase(const StartRunTestCase &tc)
{
    std::cout << "  Testing case: " << tc.name << "..." << std::endl;

    SearchEngine engine(makeTestState(tc.spec, tc.bucket_weights));
    engine.addPropagator(std::make_unique<StartRunMakespanPropagator>());

    require(engine.runPropagators(kInvalidVarId), "Initial propagation failed in test " + tc.name);

    // Apply selections: default to 1 (first alternative) for all nodes if not specified
    if (tc.fixed_selections.empty())
    {
        for (uint32_t b = 0; b < tc.bucket_weights.size(); ++b)
        {
            for (const auto &[cid, vid] : engine.state.selected_vars[b])
            {
                engine.state.setDomain(vid, Domain::makeFixed(1, true));
                require(engine.runPropagators(vid), "Default selection propagation failed in test " + tc.name);
            }
        }
    }
    else
    {
        for (const auto &[node_id, alt_idx] : tc.fixed_selections)
        {
            for (uint32_t b = 0; b < tc.bucket_weights.size(); ++b)
            {
                VarId vid = engine.state.selected_vars[b].at(EClassId{node_id});
                engine.state.setDomain(vid, Domain::makeFixed(alt_idx, true));
                require(engine.runPropagators(vid), "Explicit selection propagation failed in test " + tc.name);
            }
        }
    }

    const float initial_lb = engine.state.lower_bound;
    const size_t pre_starts_marker = engine.state.getTrailMarker();

    // Fix specified starts incrementally and assert monotonicity
    float running_lb = initial_lb;
    for (const auto &[node_id, start_val] : tc.fixed_starts)
    {
        for (uint32_t b = 0; b < tc.bucket_weights.size(); ++b)
        {
            VarId vid = engine.state.start_vars[b].at(EClassId{node_id});
            engine.state.setDomain(vid, Domain::makeFixed(start_val, false));
            require(engine.runPropagators(vid), "Fixing start for node " + std::to_string(node_id) + " failed in test " + tc.name);
        }
        const float new_lb = engine.state.lower_bound;
        require(new_lb >= running_lb - 1e-4f,
                "Monotonicity violated in test " + tc.name + ": new_lb=" + std::to_string(new_lb) +
                " < prev_lb=" + std::to_string(running_lb));
        running_lb = new_lb;
    }

    const float partial_lb = engine.state.lower_bound;

    // Check minimum expected lower bound
    require(partial_lb >= tc.expected_min_lb - 1e-4f,
            "Lower bound underflow in test " + tc.name + ": actual=" + std::to_string(partial_lb) +
            " < expected_min=" + std::to_string(tc.expected_min_lb));

    // Check exact expected lower bound if provided
    if (tc.expected_exact_lb >= 0.0f)
    {
        require(std::abs(partial_lb - tc.expected_exact_lb) <= 1e-4f,
                "Exact lower bound mismatch in test " + tc.name + ": actual=" + std::to_string(partial_lb) +
                " != expected=" + std::to_string(tc.expected_exact_lb));
    }

    // Check Admissibility against every completion
    for (size_t c_idx = 0; c_idx < tc.completions.size(); ++c_idx)
    {
        const auto &completion = tc.completions[c_idx];
        const size_t completion_marker = engine.state.getTrailMarker();

        for (const auto &[node_id, start_val] : completion)
        {
            for (uint32_t b = 0; b < tc.bucket_weights.size(); ++b)
            {
                VarId vid = engine.state.start_vars[b].at(EClassId{node_id});
                engine.state.setDomain(vid, Domain::makeFixed(start_val, false));
                require(engine.runPropagators(vid),
                        "Completion assignment failed for node " + std::to_string(node_id) + " in test " + tc.name);
            }
        }

        // Compute true makespan
        float true_weighted_makespan = 0.0f;
        for (uint32_t b = 0; b < tc.bucket_weights.size(); ++b)
        {
            const float weight = tc.bucket_weights[b];
            const float bucket_ms = engine.evaluateMakespan(engine.state, b);
            true_weighted_makespan += weight * bucket_ms;
        }

        // Admissibility: partial_lb <= true_weighted_makespan + epsilon
        require(partial_lb <= true_weighted_makespan + 1e-4f,
                "Admissibility violated in test " + tc.name + " (completion " + std::to_string(c_idx) + "): " +
                "partial_lb=" + std::to_string(partial_lb) + " > true_makespan=" + std::to_string(true_weighted_makespan));

        // Exactness when fully fixed
        const float completed_lb = engine.state.lower_bound;
        require(std::abs(completed_lb - true_weighted_makespan) <= 1e-4f,
                "Exactness violated on completion " + std::to_string(c_idx) + " in test " + tc.name + ": " +
                "completed_lb=" + std::to_string(completed_lb) + " != true_makespan=" + std::to_string(true_weighted_makespan));

        // Monotonicity to completion
        require(completed_lb >= partial_lb - 1e-4f,
                "Monotonicity from partial to completion violated in test " + tc.name);

        engine.state.backtrackTo(completion_marker);
        {
            std::vector<VarId> worklist;
            StartRunMakespanPropagator prop;
            prop.propagate(engine.state, kInvalidVarId, worklist);
        }
        require(std::abs(engine.state.lower_bound - partial_lb) <= 1e-4f,
                "Backtrack failed to restore partial lower bound in test " + tc.name + ": " +
                "current=" + std::to_string(engine.state.lower_bound) + " != partial=" + std::to_string(partial_lb));
    }

    // Check Pruning behavior if test_best_cost is specified
    if (tc.test_best_cost >= 0.0f)
    {
        engine.state.best_cost = tc.test_best_cost;
        std::vector<VarId> worklist;
        StartRunMakespanPropagator prop;
        bool prop_res = prop.propagate(engine.state, kInvalidVarId, worklist);
        if (tc.expect_prune)
        {
            require(!prop_res, "Expected prune with best_cost=" + std::to_string(tc.test_best_cost) +
                               " and lb=" + std::to_string(engine.state.lower_bound) + " in test " + tc.name);
        }
        else
        {
            require(prop_res, "Unexpected prune with best_cost=" + std::to_string(tc.test_best_cost) +
                              " and lb=" + std::to_string(engine.state.lower_bound) + " in test " + tc.name);
        }
        engine.state.best_cost = TGConstants::INF;
    }

    // Check Backtracking restoration to pre_starts_marker
    engine.state.backtrackTo(pre_starts_marker);
    {
        std::vector<VarId> worklist;
        StartRunMakespanPropagator prop;
        prop.propagate(engine.state, kInvalidVarId, worklist);
    }
    require(std::abs(engine.state.lower_bound - initial_lb) <= 1e-4f,
            "Backtrack to initial state failed to restore initial LB in test " + tc.name + ": " +
            "current=" + std::to_string(engine.state.lower_bound) + " != initial=" + std::to_string(initial_lb));
}

// ============================================================================
// TEST SUITE: TOUGH EDGE CASES
// ============================================================================

inline std::vector<StartRunTestCase> createTestCases()
{
    const Engine cpu{0, EngineType::CPU};
    const Engine gpu0{0, EngineType::CUDA_GPU};
    const Engine gpu1{1, EngineType::CUDA_GPU};
    const Engine gpu2{2, EngineType::CUDA_GPU};

    std::vector<StartRunTestCase> cases;

    // Case 1: Serial Chain on Single CPU Engine (A -> B -> C)
    {
        StartRunTestCase tc;
        tc.name = "SerialChainSingleEngine";
        tc.spec = {
            {{NodeSpec({}, {cpu}, 2.0f)}},       // 0: A
            {{NodeSpec({0}, {cpu}, 3.0f)}},      // 1: B
            {{NodeSpec({1}, {cpu}, 4.0f)}}       // 2: C
        };
        // All fixed: start_0=0, start_1=1, start_2=2
        tc.fixed_starts = {{0, 0}, {1, 1}, {2, 2}};
        tc.completions = {{{0, 0}, {1, 1}, {2, 2}}};
        tc.expected_exact_lb = 9.0f; // 2 + 3 + 4 = 9
        tc.test_best_cost = 8.5f;
        tc.expect_prune = true; // 9.0 >= 8.5
        cases.push_back(tc);
    }

    // Case 2: Serial Chain with Gap (A fixed at 0, C fixed at 2, B unfixed)
    {
        StartRunTestCase tc;
        tc.name = "SerialChainWithGap";
        tc.spec = {
            {{NodeSpec({}, {cpu}, 2.0f)}},       // 0: A
            {{NodeSpec({0}, {cpu}, 3.0f)}},      // 1: B
            {{NodeSpec({1}, {cpu}, 4.0f)}}       // 2: C
        };
        // Partial: A at 0, C at 2. Gap at 1.
        // Run 0: A [0], finishes at 2.0.
        // Run 1: C [2], on CPU starts at >= 2.0, duration 4.0 -> finish >= 6.0.
        tc.fixed_starts = {{0, 0}, {2, 2}};
        tc.completions = {
            {{0, 0}, {1, 1}, {2, 2}} // True makespan = 9.0
        };
        tc.expected_min_lb = 6.0f;
        tc.test_best_cost = 5.0f;
        tc.expect_prune = true; // 6.0 >= 5.0
        cases.push_back(tc);
    }

    // Case 3: Parallel Diamond Fork-Join with Multi-Engine Concurrency
    // A (CPU, 2.0) -> B (GPU0, 5.0) and C (GPU1, 3.0) -> D (CPU, 4.0)
    {
        StartRunTestCase tc;
        tc.name = "DiamondForkJoinMultiEngine";
        tc.spec = {
            {{NodeSpec({}, {cpu}, 2.0f)}},            // 0: A
            {{NodeSpec({0}, {gpu0}, 5.0f)}},          // 1: B
            {{NodeSpec({0}, {gpu1}, 3.0f)}},          // 2: C
            {{NodeSpec({1, 2}, {cpu}, 4.0f)}}         // 3: D
        };
        // Partial starts: A at 0, B at 1, C at 2.
        // A finishes at 2.0.
        // B on GPU0 finishes at 2.0 + 5.0 = 7.0.
        // C on GPU1 finishes at 2.0 + 3.0 = 5.0.
        // Partial LB = max(7.0, 5.0) = 7.0.
        tc.fixed_starts = {{0, 0}, {1, 1}, {2, 2}};
        tc.completions = {
            {{0, 0}, {1, 1}, {2, 2}, {3, 3}} // D starts at max(7.0, 5.0)=7.0, takes 4.0 -> true makespan = 11.0
        };
        tc.expected_exact_lb = 7.0f;
        cases.push_back(tc);
    }

    // Case 4: Diamond with Gap (Only A at 0 and D at 3 fixed)
    {
        StartRunTestCase tc;
        tc.name = "DiamondWithGapAcrossBranches";
        tc.spec = {
            {{NodeSpec({}, {cpu}, 2.0f)}},            // 0: A
            {{NodeSpec({0}, {gpu0}, 5.0f)}},          // 1: B
            {{NodeSpec({0}, {gpu1}, 3.0f)}},          // 2: C
            {{NodeSpec({1, 2}, {cpu}, 4.0f)}}         // 3: D
        };
        // A at 0, D at 3.
        // Run 0: A finishes at 2.0.
        // Run 1: D on CPU starts at >= 2.0 (CPU finish), duration 4.0 -> finish >= 6.0.
        tc.fixed_starts = {{0, 0}, {3, 3}};
        tc.completions = {
            {{0, 0}, {1, 1}, {2, 2}, {3, 3}}, // true makespan = 11.0
            {{0, 0}, {1, 2}, {2, 1}, {3, 3}}  // alternative completion, true makespan = 11.0
        };
        tc.expected_min_lb = 6.0f;
        cases.push_back(tc);
    }

    // Case 5: Disjoint Runs on Same Engine
    // 5 ops: 0 (1.0), 1 (2.0), 2 (3.0), 3 (4.0), 4 (5.0). Total sequential = 15.0.
    {
        StartRunTestCase tc;
        tc.name = "DisjointRunsSameEngine";
        tc.spec = {
            {{NodeSpec({}, {cpu}, 1.0f)}},
            {{NodeSpec({0}, {cpu}, 2.0f)}},
            {{NodeSpec({1}, {cpu}, 3.0f)}},
            {{NodeSpec({2}, {cpu}, 4.0f)}},
            {{NodeSpec({3}, {cpu}, 5.0f)}}
        };
        // Run 0: {0, 1} with starts 0, 1. Finish = 1 + 2 = 3.0.
        // Gap at 2.
        // Run 1: {3, 4} with starts 3, 4. Starts at >= 3.0, executes 3 (4.0) then 4 (5.0) -> finish = 3 + 4 + 5 = 12.0.
        tc.fixed_starts = {{0, 0}, {1, 1}, {3, 3}, {4, 4}};
        tc.completions = {
            {{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}} // total = 15.0
        };
        tc.expected_exact_lb = 12.0f;
        cases.push_back(tc);
    }

    // Case 6: Independent Multi-Engine Parallelism (No dependencies)
    // Op 0 on GPU0 (10.0), Op 1 on GPU1 (20.0), Op 2 on GPU2 (15.0)
    {
        StartRunTestCase tc;
        tc.name = "IndependentMultiEngineParallelism";
        tc.spec = {
            {{NodeSpec({}, {gpu0}, 10.0f)}},
            {{NodeSpec({}, {gpu1}, 20.0f)}},
            {{NodeSpec({}, {gpu2}, 15.0f)}}
        };
        // All fixed: start_0=0, start_1=1, start_2=2.
        // Since all run concurrently, makespan = max(10, 20, 15) = 20.0.
        tc.fixed_starts = {{0, 0}, {1, 1}, {2, 2}};
        tc.completions = {{{0, 0}, {1, 1}, {2, 2}}};
        tc.expected_exact_lb = 20.0f;
        cases.push_back(tc);
    }

    // Case 7: Zero-Duration View Operations (A -> View -> B)
    {
        StartRunTestCase tc;
        tc.name = "ZeroDurationViewOps";
        tc.spec = {
            {{NodeSpec({}, {cpu}, 3.0f)}},                          // 0: A (CPU, 3.0)
            {{NodeSpec({0}, {}, 0.0f, true, OpType::RESHAPE)}},     // 1: View (0.0, is_view=true)
            {{NodeSpec({1}, {cpu}, 4.0f)}}                          // 2: B (CPU, 4.0)
        };
        // All fixed: start_0=0, start_1=1, start_2=2
        // A finishes at 3.0. View finishes at 3.0. B starts at 3.0, finishes at 7.0.
        tc.fixed_starts = {{0, 0}, {1, 1}, {2, 2}};
        tc.completions = {{{0, 0}, {1, 1}, {2, 2}}};
        tc.expected_exact_lb = 7.0f;
        cases.push_back(tc);
    }

    // Case 8: INPUT and CACHE nodes (0 finish time, no engine execution)
    {
        StartRunTestCase tc;
        tc.name = "InputAndCacheNodes";
        tc.spec = {
            {{NodeSpec({}, {}, 0.0f, false, OpType::INPUT)}},       // 0: Input
            {{NodeSpec({}, {}, 0.0f, false, OpType::CACHE)}},       // 1: Cache
            {{NodeSpec({0, 1}, {cpu}, 5.0f)}}                       // 2: Add (CPU, 5.0)
        };
        tc.fixed_starts = {{0, 0}, {1, 1}, {2, 2}};
        tc.completions = {{{0, 0}, {1, 1}, {2, 2}}};
        tc.expected_exact_lb = 5.0f;
        cases.push_back(tc);
    }

    // Case 9: Partial Selection (Node has multiple candidate implementations)
    {
        StartRunTestCase tc;
        tc.name = "PartialSelectionCandidates";
        tc.spec = {
            // Node 0: Alt 1 (cost 6.0), Alt 2 (cost 12.0)
            {NodeSpec({}, {cpu}, 6.0f), NodeSpec({}, {cpu}, 12.0f)},
            // Node 1: Add (cost 4.0)
            {NodeSpec({0}, {cpu}, 4.0f)}
        };
        // Fix selection for node 0 to alternative 2 (cost 12.0)
        tc.fixed_selections = {{0, 2}, {1, 1}};
        tc.fixed_starts = {{0, 0}, {1, 1}};
        tc.completions = {{{0, 0}, {1, 1}}};
        tc.expected_exact_lb = 16.0f; // 12 + 4 = 16
        cases.push_back(tc);
    }

    // Case 10: Multi-Bucket Graph with Weights
    {
        StartRunTestCase tc;
        tc.name = "MultiBucketWeightedGraph";
        tc.spec = {
            {{NodeSpec({}, {cpu}, 4.0f)}},
            {{NodeSpec({0}, {cpu}, 6.0f)}}
        };
        // Bucket 0 weight 2.0, Bucket 1 weight 0.5
        // Both buckets have same graph: makespan = 10.0
        // Total LB = 2.0 * 10.0 + 0.5 * 10.0 = 25.0
        tc.bucket_weights = {2.0f, 0.5f};
        tc.fixed_starts = {{0, 0}, {1, 1}};
        tc.completions = {{{0, 0}, {1, 1}}};
        tc.expected_exact_lb = 25.0f;
        cases.push_back(tc);
    }

    return cases;
}

// ============================================================================
// EXHAUSTIVE SUBGRAPH PERMUTATION FUZZER
// ============================================================================

inline void runExhaustivePermutationFuzzer()
{
    std::cout << "  Running Exhaustive Subgraph Permutation Fuzzer (2^6 combinations)..." << std::endl;

    const Engine cpu{0, EngineType::CPU};
    const Engine gpu{0, EngineType::CUDA_GPU};

    // 6-node DAG with CPU and GPU:
    // 0: Root (CPU, 2.0)
    // 1: Fork1 (GPU, 3.0, child: 0)
    // 2: Fork2 (CPU, 1.5, child: 0)
    // 3: Mid (GPU, 4.0, child: 1)
    // 4: Join (CPU, 2.5, children: 2, 3)
    // 5: Sink (CPU, 1.0, child: 4)
    GraphSpec spec = {
        {{NodeSpec({}, {cpu}, 2.0f)}},
        {{NodeSpec({0}, {gpu}, 3.0f)}},
        {{NodeSpec({0}, {cpu}, 1.5f)}},
        {{NodeSpec({1}, {gpu}, 4.0f)}},
        {{NodeSpec({2, 3}, {cpu}, 2.5f)}},
        {{NodeSpec({4}, {cpu}, 1.0f)}}
    };

    // Ground truth topological start assignment:
    const std::vector<int32_t> true_starts = {0, 1, 2, 3, 4, 5};

    SearchEngine ref_engine(makeTestState(spec, {1.0f}));
    for (uint32_t node = 0; node < 6; ++node)
    {
        ref_engine.state.setDomain(ref_engine.state.selected_vars[0].at(EClassId{node}), Domain::makeFixed(1, true));
        ref_engine.state.setDomain(ref_engine.state.start_vars[0].at(EClassId{node}), Domain::makeFixed(true_starts[node], false));
    }
    const float ground_truth_makespan = ref_engine.evaluateMakespan(ref_engine.state, 0);
    require(ground_truth_makespan > 0.0f, "Ground truth makespan must be positive");

    // Test ALL 2^6 = 64 subsets of fixed starts
    for (uint32_t mask = 0; mask < 64; ++mask)
    {
        SearchEngine test_engine(makeTestState(spec, {1.0f}));
        test_engine.addPropagator(std::make_unique<StartRunMakespanPropagator>());
        test_engine.runPropagators(kInvalidVarId);

        for (uint32_t node = 0; node < 6; ++node)
        {
            test_engine.state.setDomain(test_engine.state.selected_vars[0].at(EClassId{node}), Domain::makeFixed(1, true));
            test_engine.runPropagators(test_engine.state.selected_vars[0].at(EClassId{node}));
        }

        // Apply subset of fixed starts
        for (uint32_t node = 0; node < 6; ++node)
        {
            if ((mask & (1u << node)) != 0)
            {
                VarId st_vid = test_engine.state.start_vars[0].at(EClassId{node});
                test_engine.state.setDomain(st_vid, Domain::makeFixed(true_starts[node], false));
                test_engine.runPropagators(st_vid);
            }
        }

        const float lb = test_engine.state.lower_bound;

        // Admissibility: LB <= Ground truth makespan + epsilon
        require(lb <= ground_truth_makespan + 1e-4f,
                "Admissibility violated in permutation mask " + std::to_string(mask) + ": " +
                "lb=" + std::to_string(lb) + " > true_makespan=" + std::to_string(ground_truth_makespan));

        // Exactness when all fixed
        if (mask == 63)
        {
            require(std::abs(lb - ground_truth_makespan) <= 1e-4f,
                    "Exactness violated when all starts fixed: lb=" + std::to_string(lb) +
                    " != true=" + std::to_string(ground_truth_makespan));
        }
    }
}

// ============================================================================
// REVIEWER REGISTRY FOR EASY TEST EXTENSION
// ============================================================================

inline std::vector<StartRunTestCase> registerReviewerTestCases()
{
    // Reviewers can append custom edge cases here to stress test StartRunMakespanPropagator.
    std::vector<StartRunTestCase> extra_cases;

    const Engine cpu{0, EngineType::CPU};
    const Engine gpu0{0, EngineType::CUDA_GPU};

    // Reviewer Case 1: All View Chain (Chain of 3 views)
    {
        StartRunTestCase tc;
        tc.name = "Reviewer_AllViewChain";
        tc.spec = {
            {{NodeSpec({}, {}, 0.0f, true, OpType::RESHAPE)}},
            {{NodeSpec({0}, {}, 0.0f, true, OpType::RESHAPE)}},
            {{NodeSpec({1}, {}, 0.0f, true, OpType::RESHAPE)}}
        };
        tc.fixed_starts = {{0, 0}, {1, 1}, {2, 2}};
        tc.completions = {{{0, 0}, {1, 1}, {2, 2}}};
        tc.expected_exact_lb = 0.0f;
        extra_cases.push_back(tc);
    }

    // Reviewer Case 2: CPU-GPU Ping Pong Pipeline
    {
        StartRunTestCase tc;
        tc.name = "Reviewer_CpuGpuPingPongPipeline";
        tc.spec = {
            {{NodeSpec({}, {cpu}, 2.0f)}},       // CPU (2)
            {{NodeSpec({0}, {gpu0}, 3.0f)}},     // GPU (3)
            {{NodeSpec({1}, {cpu}, 4.0f)}},      // CPU (4)
            {{NodeSpec({2}, {gpu0}, 5.0f)}}      // GPU (5)
        };
        // Total sequential dependency: 2 + 3 + 4 + 5 = 14.0
        tc.fixed_starts = {{0, 0}, {1, 1}, {2, 2}, {3, 3}};
        tc.completions = {{{0, 0}, {1, 1}, {2, 2}, {3, 3}}};
        tc.expected_exact_lb = 14.0f;
        extra_cases.push_back(tc);
    }

    return extra_cases;
}

} // namespace start_run_makespan_test

inline void runStartRunMakespanTests()
{
    std::cout << "==========================================================" << std::endl;
    std::cout << "Running StartRunMakespanPropagator Tough Test Suite" << std::endl;
    std::cout << "==========================================================" << std::endl;

    using namespace start_run_makespan_test;

    // 1. Run core edge cases
    auto cases = createTestCases();
    for (const auto &tc : cases)
    {
        executeTestCase(tc);
    }

    // 2. Run reviewer-added test cases
    auto reviewer_cases = registerReviewerTestCases();
    for (const auto &tc : reviewer_cases)
    {
        executeTestCase(tc);
    }

    // 3. Run exhaustive permutation fuzzer
    runExhaustivePermutationFuzzer();

    std::cout << "All StartRunMakespanPropagator tests passed successfully!" << std::endl;
}
