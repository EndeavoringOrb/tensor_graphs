#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "core/argparse.hpp"
#include "core/common/bench_utils.hpp"
#include "core/cost_model.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/loaders/safetensors.hpp"
#include "core/memory.hpp"
#include "core/misc.hpp"
#include "core/plan/planner.hpp"
#include "core/session.hpp"
#include "core/shape_propagator.hpp"
#include "generated/kernels_all.gen.hpp"

#include "tests/bufferize_domination.hpp"
#include "tests/dispatch_domination.hpp"
#include "tests/fused.hpp"
#include "tests/input_hashcons.hpp"
#include "tests/mem_cap_prune.hpp"
#include "tests/reference.hpp"
#include "tests/region_merge.hpp"
#include "tests/shape_propagation.hpp"

int main(int argc, char *argv[])
{
    ArgParser parser("test", "Run tests.");
    parser.add_flag({"--no-records"}, "Disable record-based testing.");
    parser.add_option({"--cache"},
                      "Path to cache file. If provided, only kernel calls "
                      "present in the cache file will be tested.",
                      "");
    parser.add_positional("targetKernel", "Test only kernels whose name contain this string.", "");

    if (!parser.parse(argc, argv))
    {
        return 1;
    }

    std::string targetKernel = parser.get_positional("targetKernel");
    bool useRecords = !parser.get_flag("--no-records");
    std::string cachePath = parser.get_option("--cache");

    if (targetKernel.empty() && cachePath.empty())
    {
        runRegionMergeTests();
        runShapePropagationTests();
        runPreExtractionMemCapTests();
        runDispatchDominationTests();
        runBufferizeDominationTests();
        runInputHashconsTests();
        // runRefTests(); TODO: fix python tests
    }

    runNonReferenceKernelTests(targetKernel, useRecords, cachePath);

    return 0;
}
