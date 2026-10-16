// tensor_graphs_cpp/test.cpp
#include "core/argparse.hpp"
#include "core/logging.hpp"
#include "core/settings.hpp"

// Non-pruning / Structural Regression Tests
#include "tests/constant_view_regression.hpp"
#include "tests/cuda_sync_regression.hpp"
#include "tests/fused.hpp"
#include "tests/fusion_rule_sweep_regression.hpp"
#include "tests/input_hashcons.hpp"
#include "tests/incremental_propagators.hpp"
#include "tests/planner_structure.hpp"
#include "tests/bucket_independence.hpp"
#include "tests/reference.hpp"
#include "tests/region_merge.hpp"
#include "tests/selection_reachability.hpp"
#include "tests/shape_propagation.hpp"
#include "tests/storage_output_regression.hpp"
#include "tests/view_bufferize_regression.hpp"
#include "tests/start_run_makespan.hpp"
#include "tests/start_bounds.hpp"
#include "tests/write_after_read_cache.hpp"
#include "tests/private_layout_regression.hpp"
#include "tests/start_rank_window_regression.hpp"
#include "tests/relaxation_bound.hpp"
#include "tests/scatter_rows.hpp"
#include "tests/scatter_canonical_keys.hpp"
#include "tests/bf16_transposed_wstream.hpp"
#include "tests/reduce_last_axis.hpp"

int main(int argc, char *argv[])
{
    ArgParser parser("test", "Run tests.");
    parser.add_flag({"--bucket-independence"}, "Run independent bucket and masked update regressions.");
    parser.add_flag({"--private-layout"}, "Run private-layout storage-span regression tests.");
    parser.add_flag({"--start-rank-window"}, "Run start rank window group regression tests.");
    parser.add_flag({"--relaxation-bound"}, "Run randomized relaxation lower-bound tests against brute force.");
    parser.add_flag({"--all-records"}, "Enable record-based testing.");
    parser.add_option({"--cache"}, "Path to cache file.", "");
    parser.add_flag({"--skip-fused"}, "Skip fused kernel testing.");
    parser.add_flag({"--cuda-sync"}, "Run only CUDA synchronization regression tests.");
    parser.add_flag({"--view-reg"}, "Run only view and bufferize regression tests.");
    parser.add_flag({"--const-view"}, "Run only constant and view regression tests.");
    parser.add_flag({"--selection-reachability"}, "Run only selection reachability regression tests.");
    parser.add_flag({"--incremental-propagators"}, "Run only incremental propagator equivalence tests.");
    parser.add_flag({"--write-after-read-cache"}, "Run write-after-read reader cache regression tests.");
    parser.add_flag({"--start-bounds"}, "Run start bounds and precedence regression tests.");
    parser.add_flag({"--start-run-makespan"}, "Run only start-run makespan propagator tests.");
    parser.add_flag({"--planner-structure"}, "Run only planner structure regression tests.");
    parser.add_flag({"--fusion-regression"}, "Run only fusion rule sweep regression tests.");
    parser.add_flag({"--scatter-canonical"}, "Run scatter canonical keys regression tests.");
    parser.add_positional("targetKernel", "Test only kernels whose name contain this string.", "");

    std::vector<std::string> remaining_args;
    if (!parser.parse(argc, argv, &remaining_args))
    {
        return 1;
    }

    Settings settings;
    settings.load(remaining_args);

    std::string targetKernel = parser.get_positional("targetKernel");
    bool useRecords = parser.get_flag("--all-records");
    std::string cachePath = parser.get_option("--cache");
    bool skipFused = parser.get_flag("--skip-fused");
    if (parser.get_flag("--bucket-independence"))
    {
        try { BucketIndependence::run(); }
        catch (const std::exception &error) { std::cerr << error.what() << std::endl; return 1; }
        return 0;
    }
    if (parser.get_flag("--private-layout"))
    {
        try { PrivateLayoutRegression::run(); }
        catch (const std::exception &error) { std::cerr << error.what() << std::endl; return 1; }
        return 0;
    }
    if (parser.get_flag("--start-rank-window"))
    {
        try { StartRankWindowRegression::run(); }
        catch (const std::exception &error) { std::cerr << error.what() << std::endl; return 1; }
        return 0;
    }
    if (parser.get_flag("--relaxation-bound"))
    {
        try { RelaxationBoundRegression::run(); }
        catch (const std::exception &error) { std::cerr << error.what() << std::endl; return 1; }
        return 0;
    }
    if (parser.get_flag("--write-after-read-cache"))
    {
        runWriteAfterReadCacheTests();
        return 0;
    }
    if (parser.get_flag("--start-bounds"))
    {
        runStartBoundsTests();
        return 0;
    }
    if (parser.get_flag("--planner-structure"))
    {
        runPlannerStructureTests();
        return 0;
    }
    if (parser.get_flag("--fusion-regression"))
    {
        FusionRuleSweepRegression::runFusionRuleSweepRegressionTests();
        return 0;
    }
    if (parser.get_flag("--scatter-canonical"))
    {
        runScatterCanonicalKeysTests();
        return 0;
    }
    if (parser.get_flag("--incremental-propagators"))
    {
        runIncrementalPropagatorTests();
        return 0;
    }
    if (parser.get_flag("--start-run-makespan"))
    {
        runStartRunMakespanTests();
        return 0;
    }
    if (parser.get_flag("--selection-reachability"))
    {
        runSelectionReachabilityTests();
        return 0;
    }
    if (parser.get_flag("--cuda-sync"))
    {
        runCudaSyncRegressionTests();
        return 0;
    }

    if (parser.get_flag("--view-reg"))
    {
        runViewBufferizeRegressionTests();
        return 0;
    }

    if (parser.get_flag("--const-view"))
    {
        runConstantViewRegressionTests();
        return 0;
    }

    if (targetKernel.empty() && cachePath.empty())
    {
        // Structural & Operator Correctness Tests
        runRegionMergeTests();
        runScatterRowsTests();
        runScatterCanonicalKeysTests();
        runBF16TransposedWStreamTests();
        runReduceLastAxisTests();
        runPlannerStructureTests();
        FusionRuleSweepRegression::runFusionRuleSweepRegressionTests();
        runSelectionReachabilityTests();
        runIncrementalPropagatorTests();
        runStartRunMakespanTests();
        runShapePropagationTests();
        runInputHashconsTests();
        testStorageOutputMatching();
        runViewBufferizeRegressionTests();
        runConstantViewRegressionTests();
        runCudaSyncRegressionTests();
    }

    if (!skipFused)
    {
        if (!runNonReferenceKernelTests(targetKernel, useRecords, cachePath))
        {
            LOG(ERROR) << "Fused kernel tests failed";
            return 1;
        }
    }

    LOG(INFO) << "finished testing";
    return 0;
}
