#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "core/cost_model.hpp"
#include "core/graph.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include "core/session.hpp"
#include "core/types.hpp"
#include "tests/common.hpp"

namespace ConstantViewRegression
{

inline void populateAllKernelDummyRecords(CostModel &cm, float defaultRuntime = 1.0f)
{
    for (const auto &pair : KernelRegistry::get().getAllKernels())
    {
        const auto &k = pair.second;
        if (cm.records.find(k.uid) == cm.records.end())
        {
            Record r;
            r.kernelId = k.uid;
            r.buildContextId = BUILD_CONTEXT_ID;
            r.hwTag = HW_TAG;
            r.runTime = defaultRuntime;
            r.output_mem_space = k.output_mem_space;
            r.engines = k.engines;
            cm.records[k.uid].push_back(r);
        }
    }
}

// =============================================================================
// Test 1: Constants consumed strictly via view operations (e.g. g.fill)
// =============================================================================
inline void testConstantsThroughViewOps()
{
    std::cout << "  - running testConstantsThroughViewOps..." << std::endl;

    Graph graph;
    MemoryManager mem;

    // Runtime input
    LogicalId x = graph.input({2, 4}, DType::FLOAT32);

    // Constant scalar passed through a fill view
    float c_val = 3.5f;
    LogicalId c = graph.constant({1}, &c_val, DType::FLOAT32);
    LogicalId c_fill = graph.fill(c, {2, 4});

    // Compute node consuming the view of the constant
    LogicalId out = graph.mul(x, c_fill);

    Session session(graph, mem, out, "", 0, nullptr, /*disableCaching=*/true);
    populateAllKernelDummyRecords(session.costModel);
    session.compile(/*doSaturate=*/false);

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    session.writeInput(x, x_data.data(), x_data.size() * sizeof(float));

    const float *out_ptr = static_cast<const float *>(session.run());

    std::vector<float> expected(8);
    for (size_t i = 0; i < x_data.size(); ++i)
    {
        expected[i] = x_data[i] * c_val;
    }

    if (!compareOutputs(expected.data(), out_ptr, 8))
    {
        Error::throw_err("[Regression Test Failed] Constant passed through fill() was not written to memory (read as "
                         "zero/garbage)!");
    }
}

// =============================================================================
// Test 2: Multiple constants in chained arithmetic (GELU/RMSNorm pattern)
// =============================================================================
inline void testMultiConstantArithmeticPipeline()
{
    std::cout << "  - running testMultiConstantArithmeticPipeline..." << std::endl;

    Graph graph;
    MemoryManager mem;

    LogicalId x = graph.input({4}, DType::FLOAT32);

    float c1_val = 0.5f;
    float c2_val = 10.0f;
    LogicalId c1 = graph.constant({1}, &c1_val, DType::FLOAT32);
    LogicalId c2 = graph.constant({1}, &c2_val, DType::FLOAT32);

    LogicalId c1_fill = graph.fill(c1, {4});
    LogicalId c2_fill = graph.fill(c2, {4});

    // (x * 0.5) + 10.0
    LogicalId t1 = graph.mul(x, c1_fill);
    LogicalId out = graph.add(t1, c2_fill);

    Session session(graph, mem, out, "", 0, nullptr, /*disableCaching=*/true);
    populateAllKernelDummyRecords(session.costModel);
    session.compile(/*doSaturate=*/false);

    std::vector<float> x_data = {2.0f, 4.0f, 6.0f, 8.0f};
    session.writeInput(x, x_data.data(), x_data.size() * sizeof(float));

    const float *out_ptr = static_cast<const float *>(session.run());

    std::vector<float> expected = {11.0f, 12.0f, 13.0f, 14.0f};
    if (!compareOutputs(expected.data(), out_ptr, 4))
    {
        Error::throw_err("[Regression Test Failed] Multi-constant arithmetic pipeline output mismatch!");
    }
}

// =============================================================================
// Test 3: Runtime input routed through view operations (writeInput test)
// =============================================================================
inline void testWriteInputThroughViewOps()
{
    std::cout << "  - running testWriteInputThroughViewOps..." << std::endl;

    Graph graph;
    MemoryManager mem;

    LogicalId x = graph.input({2, 4}, DType::FLOAT32);
    LogicalId x_view = graph.reshape(x, {8});

    LogicalId y = graph.input({8}, DType::FLOAT32);
    LogicalId out = graph.add(x_view, y);

    Session session(graph, mem, out, "", 0, nullptr, /*disableCaching=*/true);
    populateAllKernelDummyRecords(session.costModel);
    session.compile(/*doSaturate=*/false);

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> y_data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};

    // Both writeInput calls must succeed even though x is consumed by a view op
    session.writeInput(x, x_data.data(), x_data.size() * sizeof(float));
    session.writeInput(y, y_data.data(), y_data.size() * sizeof(float));

    const float *out_ptr = static_cast<const float *>(session.run());

    std::vector<float> expected = {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f, 77.0f, 88.0f};
    if (!compareOutputs(expected.data(), out_ptr, 8))
    {
        Error::throw_err("[Regression Test Failed] writeInput through view operations failed!");
    }
}

// =============================================================================
// Test 4: Root node is a view operation (Session::run output pointer test)
// =============================================================================
inline void testRootNodeIsViewOp()
{
    std::cout << "  - running testRootNodeIsViewOp..." << std::endl;

    Graph graph;
    MemoryManager mem;

    LogicalId x = graph.input({4}, DType::FLOAT32);
    float two_val = 2.0f;
    LogicalId two = graph.fill(two_val, {4});
    LogicalId doubled = graph.mul(x, two); // [2.0, 4.0, 6.0, 8.0]

    // Root output is a slice view: [1..3] -> elements {4.0, 6.0}
    LogicalId root = graph.slice(doubled, {1}, {3}, {1});

    Session session(graph, mem, root, "", 0, nullptr, /*disableCaching=*/true);
    populateAllKernelDummyRecords(session.costModel);
    session.compile(/*doSaturate=*/false);

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    session.writeInput(x, x_data.data(), x_data.size() * sizeof(float));

    const float *out_ptr = static_cast<const float *>(session.run());

    std::vector<float> expected = {4.0f, 6.0f};
    if (!compareOutputs(expected.data(), out_ptr, 2))
    {
        Error::throw_err("[Regression Test Failed] Session::run returned incorrect pointer for root view operation!");
    }
}

inline void testAnalysisConstantsDoNotOverwriteViewStorage()
{
    std::cout << "  - running testAnalysisConstantsDoNotOverwriteViewStorage..." << std::endl;
    Graph graph;
    MemoryManager mem;
    LogicalId x = graph.input({8, 2048}, DType::FLOAT32);
    LogicalId ones = graph.fill(1.0f, {8, 2048});
    LogicalId sum = graph.add(x, ones);
    LogicalId out = graph.concat({sum, sum}, 1);
    CostModel cost_model(false, "");
    populateAllKernelDummyRecords(cost_model);
    Settings settings = Settings::get_default();
    setupTestSettings(settings);
    Planner planner(cost_model, settings);
    planner.initBaseEGraph(out, graph, topologicalSort({out}, graph));

    // InfinityDomination loads dense reference data for views during analysis.
    // Reproduce that state without depending on model weights or a reference repo.
    EClassId ones_class = planner.baseState.egraph.findConst(planner.baseState.nodeToEClass.at(ones));
    std::vector<float> dense_ones(8 * 2048, 1.0f);
    auto snapshot = std::make_shared<std::vector<uint8_t>>(dense_ones.size() * sizeof(float));
    std::memcpy(snapshot->data(), dense_ones.data(), snapshot->size());
    planner.baseState.egraph.constantStaging[ones_class] = snapshot;
    Session session(graph, mem, out, "", 0, nullptr, true);
    session.ensureFullBucket();
    Bucket bucket = session.manualBuckets.at(session.fullBucketIdx);
    CompiledGraph compiled = planner.plan(out, graph, bucket, {}, false);
    if (compiled.constantStaging.count(ones_class))
        Error::throw_err("[Regression Test Failed] Broadcast analysis snapshot became a runtime constant!");

    compiled.bucket = bucket;
    session.cachedGraphs.push_back(std::move(compiled));
    session.cachedBucketWeights = {1.0f};
    session.isPlanned = true;
    session.compile(false);
    std::vector<float> input(8 * 2048, 2.0f);
    std::vector<float> expected(8 * 4096, 3.0f);
    for (int iteration = 0; iteration < 2; ++iteration)
    {
        session.writeInput(x, input.data(), input.size() * sizeof(float));
        const float *actual = static_cast<const float *>(session.run(bucket));
        if (!compareOutputs(expected.data(), actual, expected.size()))
            Error::throw_err("[Regression Test Failed] Analysis snapshot corrupted concat inputs!");
    }
}

} // namespace ConstantViewRegression

inline void runConstantViewRegressionTests()
{
    std::cout << "constant & view session regression tests" << std::endl << std::flush;
    ConstantViewRegression::testConstantsThroughViewOps();
    ConstantViewRegression::testMultiConstantArithmeticPipeline();
    ConstantViewRegression::testWriteInputThroughViewOps();
    ConstantViewRegression::testRootNodeIsViewOp();
    ConstantViewRegression::testAnalysisConstantsDoNotOverwriteViewStorage();
}
