#pragma once

#include <iostream>

#include "core/cost_model.hpp"
#include "core/graph.hpp"
#include "core/plan/planner.hpp"
#include "core/settings.hpp"
#include "tests/common.hpp"

inline void runPlannerStructureTests()
{
    std::cout << "planner structure tests" << std::endl << std::flush;

    CostModel costModel(false, "");
    Settings settings;
    settings.do_saturate = false;

    Graph graph;
    const LogicalId inputId = graph.input({8, 8}, DType::FLOAT32);

    Bucket fullBucket;
    fullBucket.inputDirtyRegions[inputId] = {makeRegion({{0, 8}, {0, 8}})};
    fullBucket.outputNeededRegion = {makeRegion({{0, 8}, {0, 8}})};

    Planner planner(costModel, settings);
    const EGraph result = planner.saturateBucket(inputId, graph, fullBucket, false);

    for (const ENode &enode : result.getENodes())
    {
        if (enode.getOpType() == OpType::SCATTER)
        {
            Error::throw_err("[PlannerStructure] Full-region bucket unexpectedly contains a SCATTER enode");
        }
    }
}
