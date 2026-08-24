#pragma once

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/constants.hpp"
#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/misc.hpp"
#include "core/plan/extractor.hpp"
#include "core/plan/planner.hpp"
#include "core/settings.hpp"
#include "core/types.hpp"
#include "tests/common.hpp"

namespace DispatchBenchmarkTest
{

inline void buildDispatchBenchmarkGraph(Graph &graph, std::vector<LogicalId> &inputs, LogicalId &root)
{
    LogicalId in0 = graph.input({8, 8}, DType::FLOAT32);
    LogicalId in1 = graph.input({8, 8}, DType::FLOAT32);
    LogicalId in2 = graph.input({8, 8}, DType::FLOAT32);
    LogicalId in3 = graph.input({8, 8}, DType::FLOAT32);
    LogicalId in4 = graph.input({8, 8}, DType::FLOAT32);
    LogicalId in5 = graph.input({8, 8}, DType::FLOAT32);

    inputs = {in0, in1, in2, in3, in4, in5};

    // Layer 1: 6 parallel ops creating wide permutation space
    LogicalId a0 = graph.add(in0, in1);
    LogicalId a1 = graph.mul(in1, in2);
    LogicalId a2 = graph.add(in2, in3);
    LogicalId a3 = graph.mul(in3, in4);
    LogicalId a4 = graph.add(in4, in5);
    LogicalId a5 = graph.mul(in0, in5);

    // Layer 2: 4 converging ops
    LogicalId b0 = graph.add(a0, a1);
    LogicalId b1 = graph.mul(a2, a3);
    LogicalId b2 = graph.add(a4, a5);
    LogicalId b3 = graph.mul(a1, a4);

    // Layer 3: 2 ops
    LogicalId c0 = graph.add(b0, b1);
    LogicalId c1 = graph.mul(b2, b3);

    // Root
    root = graph.add(c0, c1);
}

struct BenchmarkTrialResult
{
    std::string name;
    double time_ms = 0.0;
    uint64_t total_orders = 0;
    float min_cost = TGConstants::INF;
    bool optimal = false;
    bool was_faster = false;
    double speedup = 1.0;
};

template <typename IteratorFactory>
inline BenchmarkTrialResult runTrial(const std::string &rule_name, IteratorFactory &&make_iter,
                                     const std::unordered_map<EClassId, uint32_t> &selection_map, const EGraph &egraph,
                                     const std::vector<ENodeInfo> &enodeInfos, double baseline_ms, int warmup_iters = 5,
                                     int timed_iters = 25)
{
    BenchmarkTrialResult res;
    res.name = rule_name;

    // Warmup
    for (int w = 0; w < warmup_iters; ++w)
    {
        auto iter = make_iter();
        std::vector<EClassId> order;
        while (iter.getNextDispatchOrder(selection_map, order))
        {
        }
    }

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < timed_iters; ++t)
    {
        auto iter = make_iter();
        std::vector<EClassId> order;
        while (iter.getNextDispatchOrder(selection_map, order))
        {
            if (t == 0)
            {
                res.total_orders++;
                float cost = get_cost(order, egraph, selection_map, enodeInfos);
                res.min_cost = std::min(res.min_cost, cost);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    res.time_ms = total_ms / timed_iters;

    if (baseline_ms > 0.0)
    {
        res.was_faster = (res.time_ms < baseline_ms);
        res.speedup = (res.time_ms > 0.0) ? (baseline_ms / res.time_ms) : 1.0;
    }
    else
    {
        res.was_faster = false;
        res.speedup = 1.0;
    }

    return res;
}

} // namespace DispatchBenchmarkTest

inline void runDispatchRulesBenchmark(const std::string &outputBinaryPath = "benchmarks/dispatch_rules.bin")
{
    std::cout << "\n=======================================================\n";
    std::cout << "Running Dispatch Iterator Rules Test & Benchmark...\n";
    std::cout << "=======================================================\n";

    CostModel costModel(false, "");
    std::unordered_map<MemSpace, uint64_t> mem_caps = {{MemSpace{1, HandleType::CPP}, 1024ULL * 1024 * 1024}};
    Settings settings;
    settings.mem_caps = mem_caps;

    Graph graph;
    std::vector<LogicalId> inputs;
    LogicalId root;
    DispatchBenchmarkTest::buildDispatchBenchmarkGraph(graph, inputs, root);

    std::vector<LogicalId> topo = topologicalSort({root}, graph);
    Planner planner(costModel, settings);
    planner.initBaseEGraph(root, graph, topo, nullptr);
    populateDummyRecords(costModel, planner.baseState.egraph);

    EGraph egraph = planner.baseState.egraph;
    auto enodeInfos = planner.computeENodeInfos(egraph, planner.baseState.eclassToLogical, {}, false);

    std::unordered_map<EClassId, uint32_t> selection_map;
    for (const auto &cls : egraph.getClasses())
    {
        EClassId canon = egraph.findConst(cls.id);
        if (!egraph.getEClass(canon).enodes.empty())
            selection_map[canon] = 0;
    }

    // 1. Baseline (Without any rules)
    std::cout << "  - Profiling Baseline (no rules)..." << std::flush;
    auto make_baseline = [&]() { return makeDispatchIterator(egraph, selection_map, enodeInfos); };
    auto baseline_res =
        DispatchBenchmarkTest::runTrial("Baseline (No Rules)", make_baseline, selection_map, egraph, enodeInfos, 0.0);
    baseline_res.optimal = true;
    std::cout << " Done (" << std::fixed << std::setprecision(4) << baseline_res.time_ms << " ms, "
              << baseline_res.total_orders << " orders)\n";

    // 2. Rule 1: SingleEngineDispatchDominationRule
    std::cout << "  - Profiling SingleEngineDispatchDominationRule..." << std::flush;
    auto make_r1 = [&]() {
        return makeDispatchIterator(egraph, selection_map, enodeInfos, SingleEngineDispatchDominationRule{});
    };
    auto r1_res = DispatchBenchmarkTest::runTrial("SingleEngineDispatchDomination", make_r1, selection_map, egraph,
                                                  enodeInfos, baseline_res.time_ms);
    r1_res.optimal = (std::abs(r1_res.min_cost - baseline_res.min_cost) < 1e-5f);
    std::cout << " Done (" << r1_res.time_ms << " ms, speedup=" << r1_res.speedup << "x)\n";

    // 3. Rule 2: MultiEngineCommutativityRule
    std::cout << "  - Profiling MultiEngineCommutativityRule..." << std::flush;
    auto make_r2 = [&]() {
        return makeDispatchIterator(egraph, selection_map, enodeInfos, MultiEngineCommutativityRule{});
    };
    auto r2_res = DispatchBenchmarkTest::runTrial("MultiEngineCommutativityRule", make_r2, selection_map, egraph,
                                                  enodeInfos, baseline_res.time_ms);
    r2_res.optimal = (std::abs(r2_res.min_cost - baseline_res.min_cost) < 1e-5f);
    std::cout << " Done (" << r2_res.time_ms << " ms, speedup=" << r2_res.speedup << "x)\n";

    // 4. Rule 3: DisjointSubgraphSymmetryRule
    std::cout << "  - Profiling DisjointSubgraphSymmetryRule..." << std::flush;
    auto make_r3 = [&]() {
        return makeDispatchIterator(egraph, selection_map, enodeInfos, DisjointSubgraphSymmetryRule{});
    };
    auto r3_res = DispatchBenchmarkTest::runTrial("DisjointSubgraphSymmetryRule", make_r3, selection_map, egraph,
                                                  enodeInfos, baseline_res.time_ms);
    r3_res.optimal = (std::abs(r3_res.min_cost - baseline_res.min_cost) < 1e-5f);
    std::cout << " Done (" << r3_res.time_ms << " ms, speedup=" << r3_res.speedup << "x)\n";

    // 5. Rule 4: LastReaderBufferFreeDominationRule
    std::cout << "  - Profiling LastReaderBufferFreeDominationRule..." << std::flush;
    auto make_r4 = [&]() {
        return makeDispatchIterator(egraph, selection_map, enodeInfos, LastReaderBufferFreeDominationRule{});
    };
    auto r4_res = DispatchBenchmarkTest::runTrial("LastReaderBufferFreeDominationRule", make_r4, selection_map, egraph,
                                                  enodeInfos, baseline_res.time_ms);
    r4_res.optimal = (std::abs(r4_res.min_cost - baseline_res.min_cost) < 1e-5f);
    std::cout << " Done (" << r4_res.time_ms << " ms, speedup=" << r4_res.speedup << "x)\n";

    // 6. All Rules Combined
    std::cout << "  - Profiling All Rules Combined..." << std::flush;
    auto make_all = [&]() {
        return makeDispatchIterator(egraph, selection_map, enodeInfos, SingleEngineDispatchDominationRule{},
                                    MultiEngineCommutativityRule{}, DisjointSubgraphSymmetryRule{},
                                    LastReaderBufferFreeDominationRule{});
    };
    auto all_res = DispatchBenchmarkTest::runTrial("AllRulesCombined", make_all, selection_map, egraph, enodeInfos,
                                                   baseline_res.time_ms);
    all_res.optimal = (std::abs(all_res.min_cost - baseline_res.min_cost) < 1e-5f);
    std::cout << " Done (" << all_res.time_ms << " ms, speedup=" << all_res.speedup << "x)\n";

    // Verify optimality
    std::vector<DispatchBenchmarkTest::BenchmarkTrialResult> trials = {r1_res, r2_res, r3_res, r4_res, all_res};
    for (const auto &t : trials)
    {
        if (!t.optimal)
        {
            Error::throw_err("[DispatchRulesBenchmark] Cost optimality violation in rule: " + t.name + " (Cost=" +
                             std::to_string(t.min_cost) + " vs Base=" + std::to_string(baseline_res.min_cost) + ")");
        }
    }

    // Prepare binary records
    std::vector<RuleBenchmarkRecord> binary_records;
    for (const auto &t : {r1_res, r2_res, r3_res, r4_res})
    {
        RuleBenchmarkRecord rec;
        rec.category = "dispatch";
        rec.rule_name = t.name;
        rec.was_faster = t.was_faster;
        rec.baseline_ms = baseline_res.time_ms;
        rec.test_ms = t.time_ms;
        rec.speedup = t.speedup;
        binary_records.push_back(rec);
    }

    // Save test results to binary file
    Settings::save_rule_benchmarks(outputBinaryPath, binary_records);
    std::cout << "\nSaved rule benchmark results to: " << outputBinaryPath << "\n";

    // Print summary report
    std::cout << "\n--------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(36) << "Rule / Configuration" << std::setw(12) << "Orders" << std::setw(16)
              << "Latency (ms)" << std::setw(12) << "Speedup" << std::setw(12) << "Faster?\n";
    std::cout << "--------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(36) << baseline_res.name << std::setw(12) << baseline_res.total_orders
              << std::setw(16) << baseline_res.time_ms << std::setw(12) << "1.00x" << std::setw(12) << "BASELINE\n";

    for (const auto &t : trials)
    {
        std::string faster_str = t.was_faster ? "YES (ACTIVE)" : "NO";
        std::cout << std::left << std::setw(36) << t.name << std::setw(12) << t.total_orders << std::setw(16)
                  << t.time_ms << std::setw(12) << (std::to_string(t.speedup).substr(0, 5) + "x") << std::setw(12)
                  << faster_str << "\n";
    }
    std::cout << "--------------------------------------------------------------------------------------\n";

    // Validate that loading binary via Settings works as expected
    Settings testSettings;
    if (!testSettings.load_from_binary(outputBinaryPath))
    {
        Error::throw_err("[DispatchRulesBenchmark] Failed to load saved binary file into Settings.");
    }
    testSettings.validate_dispatch_rules();

    // Validate that unconfigured Settings triggers the required error
    Settings unconfiguredSettings;
    bool caughtExpectedError = false;
    try
    {
        unconfiguredSettings.validate_dispatch_rules();
    }
    catch (const std::exception &)
    {
        caughtExpectedError = true;
    }
    if (!caughtExpectedError)
    {
        Error::throw_err(
            "[DispatchRulesBenchmark] Expected Settings::validate_dispatch_rules() to throw on unconfigured settings!");
    }

    std::cout << "Dispatch Iterator Rules Benchmark and Settings Validation Passed!\n\n";
}