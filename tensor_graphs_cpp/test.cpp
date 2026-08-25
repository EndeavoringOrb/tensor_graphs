// tensor_graphs_cpp/test.cpp
#include "core/argparse.hpp"
#include "core/logging.hpp"
#include "core/settings.hpp"

// Unified Pruning Test Suite (Replaces all individual domination/pruning test files)
#include "tests/pruning/test.hpp"

// Non-pruning / Structural Regression Tests
#include "tests/constant_view_regression.hpp"
#include "tests/fused.hpp"
#include "tests/input_hashcons.hpp"
#include "tests/reference.hpp"
#include "tests/region_merge.hpp"
#include "tests/shape_propagation.hpp"
#include "tests/view_bufferize_regression.hpp"

int main(int argc, char *argv[])
{
    ArgParser parser("test", "Run tests.");
    parser.add_flag({"--no-records"}, "Disable record-based testing.");
    parser.add_option({"--cache"}, "Path to cache file.", "");
    parser.add_flag({"--skip-fused"}, "Skip fused kernel testing.");
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

    if (targetKernel.empty() && cachePath.empty())
    {
        // Auto-discovers and benchmarks all Dispatch, Bufferize, Malloc, Cache, Extract, and ENode rules
        runPruningTests();

        // Structural & Operator Correctness Tests
        runRegionMergeTests();
        runShapePropagationTests();
        runInputHashconsTests();
        runViewBufferizeRegressionTests();
        runConstantViewRegressionTests();
    }

    if (!skipFused)
    {
        runNonReferenceKernelTests(targetKernel, useRecords, cachePath);
    }

    LOG(INFO) << "finished testing";
    return 0;
}