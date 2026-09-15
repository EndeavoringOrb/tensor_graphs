#pragma once

#include <iostream>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/misc.hpp"
#include "core/plan/planner.hpp"
#include "core/types.hpp"

#include "tests/common.hpp"

void runRegionMergeTests()
{
    std::cout << "region merge tests" << std::endl << std::flush;
    {
        std::vector<Region> actual = mergeRegions({makeRegion({{0, 2}}), makeRegion({{2, 4}})});
        assertRegionListEquals(actual, {makeRegion({{0, 4}})}, "1D adjacent merge");
    }
    {
        std::vector<Region> actual = mergeRegions({
            makeRegion({{0, 4}, {0, 2}}),
            makeRegion({{0, 2}, {2, 4}}),
            makeRegion({{2, 4}, {2, 4}}),
        });
        assertRegionListEquals(actual,
                               {
                                   makeRegion({{0, 4}, {0, 2}}),
                                   makeRegion({{0, 4}, {2, 4}}),
                               },
                               "two-step 2D merge");
    }
    {
        std::vector<Region> actual = mergeRegions({
            makeRegion({{0, 4}, {0, 2}}),
            makeRegion({{0, 4}, {2, 4}}),
        });
        assertRegionListEquals(actual, {makeRegion({{0, 4}, {0, 4}})}, "full 2D merge");
    }
    {
        std::vector<Region> forwardA = mergeRegions({
            makeRegion({{2, 4}, {0, 1}}),
            makeRegion({{0, 2}, {0, 1}}),
        });
        std::vector<Region> forwardB = mergeRegions({
            makeRegion({{0, 2}, {0, 1}}),
            makeRegion({{2, 4}, {0, 1}}),
        });
        if (encodeRegionList(forwardA) != encodeRegionList(forwardB))
        {
            Error::throw_err("[RegionTest] merge ordering is not deterministic");
        }
    }
}