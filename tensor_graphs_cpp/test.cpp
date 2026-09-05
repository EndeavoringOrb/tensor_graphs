// tensor_graphs_cpp/test.cpp
#include "core/argparse.hpp"
#include "core/logging.hpp"
#include "core/settings.hpp"

// Unified Pruning Test Suite (Replaces all individual domination/pruning test files)
#include "tests/pruning/test.hpp"

// Non-pruning / Structural Regression Tests
#include "tests/constant_view_regression.hpp"
#include "tests/cuda_sync_regression.hpp"
#include "tests/fused.hpp"
#include "tests/input_hashcons.hpp"
#include "tests/reference.hpp"
#include "tests/region_merge.hpp"
#include "tests/shape_propagation.hpp"
#include "tests/storage_output_regression.hpp"
#include "tests/view_bufferize_regression.hpp"

int main(int argc, char *argv[])
{
    ArgParser parser("test", "Run tests.");
    parser.add_flag({"--no-records"}, "Disable record-based testing.");
    parser.add_option({"--cache"}, "Path to cache file.", "");
    parser.add_flag({"--skip-fused"}, "Skip fused kernel testing.");
    parser.add_flag({"--pruning-state"}, "Run only pruning-rule push/pop state restoration tests.");
    parser.add_flag({"--cuda-sync"}, "Run only CUDA synchronization regression tests.");
    parser.add_option({"--timeout"}, "Timeout in seconds for each pruning test run (default: 15.0).", "15.0");
    parser.add_positional("targetKernel", "Test only kernels whose name contain this string.", "");

    std::vector<std::string> remaining_args;
    if (!parser.parse(argc, argv, &remaining_args))
    {
        return 1;
    }

    Settings settings;
    settings.load(remaining_args);

    std::string targetKernel = parser.get_positional("targetKernel");
    bool useRecords = !parser.get_flag("--no-records");
    std::string cachePath = parser.get_option("--cache");
    bool skipFused = parser.get_flag("--skip-fused");
    bool pruningStateOnly = parser.get_flag("--pruning-state");
    double timeoutSeconds = 5.0;
    std::string timeoutOpt = parser.get_option("--timeout");
    if (!timeoutOpt.empty())
    {
        try
        {
            timeoutSeconds = std::stod(timeoutOpt);
        }
        catch (...)
        {
        }
    }

    if (pruningStateOnly)
    {
        return runPruningStateTests() ? 0 : 1;
    }

    if (parser.get_flag("--cuda-sync"))
    {
        runCudaSyncRegressionTests();
        return 0;
    }

    if (targetKernel.empty() && cachePath.empty())
    {
        // Auto-discovers and benchmarks all Dispatch, Bufferize, Malloc, Cache, Extract, and ENode rules and
        // combinations
        runPruningTests("benchmarks/pruning_tests.txt", "benchmarks/rules.bin", timeoutSeconds);

        // Structural & Operator Correctness Tests
        runRegionMergeTests();
        runShapePropagationTests();
        runInputHashconsTests();
        testStorageOutputMatching();
        runViewBufferizeRegressionTests();
        runConstantViewRegressionTests();
        runCudaSyncRegressionTests();
    }

    if (!skipFused)
    {
        runNonReferenceKernelTests(targetKernel, useRecords, cachePath);
    }

    LOG(INFO) << "finished testing";
    return 0;
}
