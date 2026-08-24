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

void runDispatchDominationTests()
{
    std::cout << "dispatch domination optimality tests" << std::endl << std::flush;
    CostModel costModel(false, "");
    std::unordered_map<MemSpace, uint64_t> mem_caps = {{MemSpace{1, HandleType::CPP}, 1024ULL * 1024 * 1024}};

    // -------------------------------------------------------------------------
    // Test Case 1: All Rules
    // -------------------------------------------------------------------------
    {
        Graph graph;
        LogicalId in0 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId in1 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId b1 = graph.add(in0, in1);
        LogicalId b2 = graph.mul(in0, in1);
        LogicalId root = graph.add(b1, b2);

        std::vector<LogicalId> topo = topologicalSort({root}, graph);
        Planner planner(costModel, mem_caps);
        planner.initBaseEGraph(root, graph, topo, nullptr);
        populateDummyRecords(costModel, planner.baseState.egraph);

        EGraph egraph = planner.baseState.egraph;
        std::unordered_map<EClassId, LogicalId> eclassToLogical = planner.baseState.eclassToLogical;
        std::unordered_map<LogicalId, MemSpace> cachedNodes;
        auto enodeInfos = planner.computeENodeInfos(egraph, eclassToLogical, cachedNodes, false);

        std::unordered_map<EClassId, uint32_t> selection_map;
        for (const auto &cls : egraph.getClasses())
        {
            EClassId canon = egraph.findConst(cls.id);
            if (!egraph.getEClass(canon).enodes.empty())
                selection_map[canon] = 0;
        }

        // Unconstrained (empty pruning rule set)
        auto iterUnconstrained = makeDispatchIterator(egraph, selection_map, enodeInfos);
        std::vector<EClassId> order;
        uint32_t count_unconstrained = 0;
        float min_cost_unconstrained = TGConstants::INF;
        while (iterUnconstrained.getNextDispatchOrder(selection_map, order))
        {
            count_unconstrained++;
            min_cost_unconstrained =
                std::min(min_cost_unconstrained, get_cost(order, egraph, selection_map, enodeInfos));
        }

        // All Rules -- registered as template parameters (compile-time dispatch, no vtables)
        auto iterAllRules = makeDispatchIterator(egraph, selection_map, enodeInfos,
                                                 SingleEngineDispatchDominationRule{}, MultiEngineCommutativityRule{},
                                                 DisjointSubgraphSymmetryRule{}, LastReaderBufferFreeDominationRule{});

        uint32_t count_all_rules = 0;
        float min_cost_all_rules = TGConstants::INF;
        while (iterAllRules.getNextDispatchOrder(selection_map, order))
        {
            count_all_rules++;
            min_cost_all_rules = std::min(min_cost_all_rules, get_cost(order, egraph, selection_map, enodeInfos));
        }

        if (std::abs(min_cost_all_rules - min_cost_unconstrained) > 1e-5f)
        {
            Error::throw_err("[DispatchDominationTest: All Rules] Cost mismatch! unconstrained=" +
                             std::to_string(min_cost_unconstrained) +
                             ", dominated=" + std::to_string(min_cost_all_rules));
        }

        std::cout << "  (All Rules: Unconstrained=" << count_unconstrained << ", Dominated=" << count_all_rules
                  << ", Cost=" << min_cost_all_rules << " ms)" << std::endl;
    }
}
