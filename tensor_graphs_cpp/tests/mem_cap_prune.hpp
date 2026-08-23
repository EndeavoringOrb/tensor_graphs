#pragma once

#include <iostream>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/misc.hpp"
#include "core/plan/planner.hpp"
#include "core/types.hpp"

inline void runPreExtractionMemCapTests()
{
    std::cout << "pre-extraction mem_cap pruning tests" << std::endl << std::flush;
    CostModel costModel;
    std::unordered_map<MemSpace, uint64_t> mem_caps = {{MemSpace{1, HandleType::CPP}, 8ULL * 1024 * 1024}}; // 8 MB cap

    Graph graph;
    LogicalId in0 = graph.input({1024, 1024}, DType::FLOAT32); // 4 MB
    LogicalId in1 = graph.input({1024, 1024}, DType::FLOAT32); // 4 MB
    LogicalId addNode = graph.add(in0, in1);                   // 4 MB out, 4MB in0, 4MB in1 -> 12MB out-of-place

    std::vector<LogicalId> topo = topologicalSort({addNode}, graph);
    Planner planner(costModel, mem_caps);
    planner.initBaseEGraph(addNode, graph, topo, nullptr);

    EGraph egraph = planner.baseState.egraph;
    std::unordered_map<EClassId, LogicalId> eclassToLogical = planner.baseState.eclassToLogical;
    std::unordered_map<LogicalId, MemSpace> cachedNodes;

    // 1. Test with MemCapENodeDominationRule active: out-of-place ADD (12MB) exceeds 8MB and is marked INF.
    auto enodeInfos = planner.computeENodeInfos(egraph, eclassToLogical, cachedNodes, false);
    planner.pruneEGraph(egraph, enodeInfos);

    EClassId rootEClass = egraph.findConst(planner.baseState.nodeToEClass.at(addNode));
    // Verify that non-inplace enodes exceeding 8MB are eliminated
    for (ENodeId enodeId : egraph.getEClass(rootEClass).enodes)
    {
        const ENode &node = egraph.getENode(enodeId);
        uint64_t req = (getSizeBytes(node.getShape(), node.getDType()) + 4095) & ~4095ULL;
        for (EClassId child : node.getChildren())
        {
            const EClass &cCls = egraph.getEClass(egraph.findConst(child));
            req += (getSizeBytes(cCls.shape, cCls.dtype) + 4095) & ~4095ULL;
        }
        if (req > 8ULL * 1024 * 1024)
        {
            Error::throw_err("[MemCapPruneTest] Dominated / over-cap ENode was not pruned!");
        }
    }

    // 2. Test Dependency Injection: Cleared rules preserve over-capacity nodes
    Planner unconstrainedPlanner(costModel, mem_caps);
    unconstrainedPlanner.clearDominationRules();
    EGraph unconstrainedEGraph = planner.baseState.egraph;
    auto unconstrainedInfos =
        unconstrainedPlanner.computeENodeInfos(unconstrainedEGraph, eclassToLogical, cachedNodes, false);
    unconstrainedPlanner.pruneEGraph(unconstrainedEGraph, unconstrainedInfos);

    EClassId unconstrainedRoot = unconstrainedEGraph.findConst(planner.baseState.nodeToEClass.at(addNode));
    bool foundOverCap = false;
    for (ENodeId enodeId : unconstrainedEGraph.getEClass(unconstrainedRoot).enodes)
    {
        const ENode &node = unconstrainedEGraph.getENode(enodeId);
        uint64_t req = (getSizeBytes(node.getShape(), node.getDType()) + 4095) & ~4095ULL;
        for (EClassId child : node.getChildren())
        {
            const EClass &cCls = unconstrainedEGraph.getEClass(unconstrainedEGraph.findConst(child));
            req += (getSizeBytes(cCls.shape, cCls.dtype) + 4095) & ~4095ULL;
        }
        if (req > 8ULL * 1024 * 1024)
        {
            foundOverCap = true;
            break;
        }
    }
    if (!foundOverCap)
    {
        Error::throw_err("[MemCapPruneTest] Clearing domination rules failed; over-cap ENode was still pruned!");
    }

    // 3. Test Injected MemCapENodeDominationRule specifically
    Planner injectedPlanner(costModel, mem_caps);
    injectedPlanner.clearDominationRules();
    injectedPlanner.addDominationRule(std::make_shared<MemCapENodeDominationRule>());
    EGraph injectedEGraph = planner.baseState.egraph;
    auto injectedInfos = injectedPlanner.computeENodeInfos(injectedEGraph, eclassToLogical, cachedNodes, false);
    injectedPlanner.pruneEGraph(injectedEGraph, injectedInfos);

    EClassId injectedRoot = injectedEGraph.findConst(planner.baseState.nodeToEClass.at(addNode));
    for (ENodeId enodeId : injectedEGraph.getEClass(injectedRoot).enodes)
    {
        const ENode &node = injectedEGraph.getENode(enodeId);
        uint64_t req = (getSizeBytes(node.getShape(), node.getDType()) + 4095) & ~4095ULL;
        for (EClassId child : node.getChildren())
        {
            const EClass &cCls = injectedEGraph.getEClass(injectedEGraph.findConst(child));
            req += (getSizeBytes(cCls.shape, cCls.dtype) + 4095) & ~4095ULL;
        }
        if (req > 8ULL * 1024 * 1024)
        {
            Error::throw_err("[MemCapPruneTest] Injected MemCapENodeDominationRule failed to prune over-cap ENode!");
        }
    }

    // 4. With 2MB cap: Even the 4MB inputs exceed 2MB, so deathCascade prunes everything
    std::unordered_map<MemSpace, uint64_t> tiny_caps = {{MemSpace{1, HandleType::CPP}, 2ULL * 1024 * 1024}};
    Planner tinyPlanner(costModel, tiny_caps);
    EGraph tinyEGraph = planner.baseState.egraph;
    auto tinyInfos = tinyPlanner.computeENodeInfos(tinyEGraph, eclassToLogical, cachedNodes, false);
    tinyPlanner.pruneEGraph(tinyEGraph, tinyInfos);

    EClassId tinyRoot = tinyEGraph.findConst(planner.baseState.nodeToEClass.at(addNode));
    if (!tinyEGraph.getEClass(tinyRoot).enodes.empty())
    {
        Error::throw_err("[MemCapPruneTest] deathCascade failed to eliminate root EClass on tiny memory cap!");
    }
}