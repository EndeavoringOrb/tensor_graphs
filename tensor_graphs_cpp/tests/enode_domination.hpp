#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/misc.hpp"
#include "core/plan/planner.hpp"
#include "core/types.hpp"

inline void runENodeDominationTests()
{
    std::cout << "enode domination optimality tests" << std::endl << std::flush;
    CostModel costModel;
    std::unordered_map<MemSpace, uint64_t> mem_caps = {{MemSpace{1, HandleType::CPP}, 1024ULL * 1024 * 1024}};

    // Register test kernels: fma_v1 (slower) and fma_v2 (faster)
    static bool registered = false;
    if (!registered)
    {
        KernelRegistry::get().registerKernel(
            KernelId{0xED0001}, OpType::FUSED, "fma_v1", 2, 2, nullptr, nullptr, nullptr, {}, false, false, nullptr,
            MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
            {{8, 8}, {8, 8}}, {false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

        KernelRegistry::get().registerKernel(
            KernelId{0xED0002}, OpType::FUSED, "fma_v2", 2, 2, nullptr, nullptr, nullptr, {}, false, false, nullptr,
            MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
            {{8, 8}, {8, 8}}, {false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

        KernelRegistry::get().registerKernel(
            KernelId{0xED0003}, OpType::FUSED, "fma_v3_inplace", 2, 2, nullptr, nullptr, nullptr, {0}, false, false,
            nullptr, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
            {{8, 8}, {8, 8}}, {false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

        registered = true;
    }

    // Set records for costModel: fma_v1 (2.5ms), fma_v2 (0.5ms), fma_v3_inplace (0.5ms)
    Record r1;
    r1.kernelId = KernelId{0xED0001};
    r1.buildContextId = BUILD_CONTEXT_ID;
    r1.hwTag = HW_TAG;
    r1.inputShapes = {{8, 8}, {8, 8}};
    r1.outputShape = {8, 8};
    r1.inputStrides = {{8, 1}, {8, 1}};
    r1.outputStrides = {8, 1};
    r1.inputDTypes = {DType::FLOAT32, DType::FLOAT32};
    r1.outputDType = DType::FLOAT32;
    r1.output_mem_space = MemSpace{1, HandleType::CPP};
    r1.engines = {Engine{0, EngineType::CPU}};
    r1.input_mem_spaces = {MemSpace{1, HandleType::CPP}, MemSpace{1, HandleType::CPP}};
    r1.runTime = 2.5f;
    costModel.records[r1.kernelId].push_back(r1);

    Record r2 = r1;
    r2.kernelId = KernelId{0xED0002};
    r2.runTime = 0.5f;
    costModel.records[r2.kernelId].push_back(r2);

    Record r3 = r1;
    r3.kernelId = KernelId{0xED0003};
    r3.runTime = 0.5f;
    costModel.records[r3.kernelId].push_back(r3);

    // -------------------------------------------------------------------------
    // Test 1: Faster Equivalent Implementation Prunes Slower Implementation
    // -------------------------------------------------------------------------
    {
        Graph graph;
        LogicalId in0 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId in1 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId root = graph.add(in0, in1);

        std::vector<LogicalId> topo = topologicalSort({root}, graph);
        Planner planner(costModel, mem_caps);
        planner.initBaseEGraph(root, graph, topo, nullptr);

        EGraph egraph = planner.baseState.egraph;
        std::unordered_map<EClassId, LogicalId> eclassToLogical = planner.baseState.eclassToLogical;
        std::unordered_map<LogicalId, MemSpace> cachedNodes;

        EClassId rootEClass = egraph.findConst(planner.baseState.nodeToEClass.at(root));
        EClassId in0_cls = egraph.findConst(planner.baseState.nodeToEClass.at(in0));
        EClassId in1_cls = egraph.findConst(planner.baseState.nodeToEClass.at(in1));

        // Add both fma_v1 (2.5ms) and fma_v2 (0.5ms) into rootEClass
        ENode enode_v1(KernelId{0xED0001}, OpType::FUSED, "fma_v1", {in0_cls, in1_cls}, {8, 8}, {8, 1}, DType::FLOAT32,
                       MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}});
        ENode enode_v2(KernelId{0xED0002}, OpType::FUSED, "fma_v2", {in0_cls, in1_cls}, {8, 8}, {8, 1}, DType::FLOAT32,
                       MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}});

        egraph.addENode(rootEClass, enode_v1);
        egraph.addENode(rootEClass, enode_v2);

        // A) Test with FasterEquivalentENodeDominationRule active
        auto enodeInfos = planner.computeENodeInfos(egraph, eclassToLogical, cachedNodes, false);
        planner.pruneEGraph(egraph, enodeInfos);

        bool found_v1 = false;
        bool found_v2 = false;
        for (ENodeId enodeId : egraph.getEClass(rootEClass).enodes)
        {
            const ENode &n = egraph.getENode(enodeId);
            if (n.getKernelId() == KernelId{0xED0001})
                found_v1 = true;
            if (n.getKernelId() == KernelId{0xED0002})
                found_v2 = true;
        }

        if (found_v1)
        {
            Error::throw_err(
                "[ENodeDominationTest: Test 1] Slower fma_v1 was NOT pruned by FasterEquivalentENodeDominationRule!");
        }
        if (!found_v2)
        {
            Error::throw_err("[ENodeDominationTest: Test 1] Faster fma_v2 was incorrectly pruned!");
        }

        // B) Test Dependency Injection: Clear rules and verify neither is pruned
        Planner unconstrainedPlanner(costModel, mem_caps);
        unconstrainedPlanner.clearDominationRules();

        EGraph egraphUnconstrained = planner.baseState.egraph;
        egraphUnconstrained.addENode(rootEClass, enode_v1);
        egraphUnconstrained.addENode(rootEClass, enode_v2);

        auto infosUnconstrained =
            unconstrainedPlanner.computeENodeInfos(egraphUnconstrained, eclassToLogical, cachedNodes, false);
        unconstrainedPlanner.pruneEGraph(egraphUnconstrained, infosUnconstrained);

        found_v1 = false;
        found_v2 = false;
        for (ENodeId enodeId : egraphUnconstrained.getEClass(rootEClass).enodes)
        {
            const ENode &n = egraphUnconstrained.getENode(enodeId);
            if (n.getKernelId() == KernelId{0xED0001})
                found_v1 = true;
            if (n.getKernelId() == KernelId{0xED0002})
                found_v2 = true;
        }

        if (!found_v1 || !found_v2)
        {
            Error::throw_err(
                "[ENodeDominationTest: Test 1] Unconstrained planner incorrectly pruned alternative implementations!");
        }

        std::cout << "  (Test 1 - Faster Equivalent Pruning & Dependency Injection Passed)" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 2: Different Inputs Are NOT Dominated
    // -------------------------------------------------------------------------
    {
        Graph graph;
        LogicalId in0 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId in1 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId in2 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId root = graph.add(in0, in1);

        std::vector<LogicalId> topo = topologicalSort({root}, graph);
        Planner planner(costModel, mem_caps);
        planner.initBaseEGraph(root, graph, topo, nullptr);

        EGraph egraph = planner.baseState.egraph;
        std::unordered_map<EClassId, LogicalId> eclassToLogical = planner.baseState.eclassToLogical;
        std::unordered_map<LogicalId, MemSpace> cachedNodes;

        EClassId rootEClass = egraph.findConst(planner.baseState.nodeToEClass.at(root));
        EClassId in0_cls = egraph.findConst(planner.baseState.nodeToEClass.at(in0));
        EClassId in1_cls = egraph.findConst(planner.baseState.nodeToEClass.at(in1));
        EClassId in2_cls = egraph.findConst(planner.baseState.nodeToEClass.at(in2));

        // Node A takes [in0, in1] (cost 2.5ms), Node B takes [in0, in2] (cost 0.5ms)
        ENode enode_a(KernelId{0xED0001}, OpType::FUSED, "fma_v1", {in0_cls, in1_cls}, {8, 8}, {8, 1}, DType::FLOAT32,
                      MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}});
        ENode enode_b(KernelId{0xED0002}, OpType::FUSED, "fma_v2", {in0_cls, in2_cls}, {8, 8}, {8, 1}, DType::FLOAT32,
                      MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}});

        egraph.addENode(rootEClass, enode_a);
        egraph.addENode(rootEClass, enode_b);

        auto enodeInfos = planner.computeENodeInfos(egraph, eclassToLogical, cachedNodes, false);
        planner.pruneEGraph(egraph, enodeInfos);

        bool found_a = false;
        bool found_b = false;
        for (ENodeId enodeId : egraph.getEClass(rootEClass).enodes)
        {
            const ENode &n = egraph.getENode(enodeId);
            if (n.getKernelId() == KernelId{0xED0001})
                found_a = true;
            if (n.getKernelId() == KernelId{0xED0002})
                found_b = true;
        }

        if (!found_a || !found_b)
        {
            Error::throw_err(
                "[ENodeDominationTest: Test 2] ENodes with different input dependencies were incorrectly pruned!");
        }

        std::cout << "  (Test 2 - Different Inputs Independence Passed)" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 3: Death Cascade Pruning after Domination
    // -------------------------------------------------------------------------
    {
        Graph graph;
        LogicalId in0 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId in1 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId neg_in1 = graph.neg(in1);
        LogicalId root = graph.add(in0, neg_in1);

        std::vector<LogicalId> topo = topologicalSort({root}, graph);
        Planner planner(costModel, mem_caps);
        planner.initBaseEGraph(root, graph, topo, nullptr);

        EGraph egraph = planner.baseState.egraph;
        std::unordered_map<EClassId, LogicalId> eclassToLogical = planner.baseState.eclassToLogical;
        std::unordered_map<LogicalId, MemSpace> cachedNodes;

        EClassId rootEClass = egraph.findConst(planner.baseState.nodeToEClass.at(root));
        EClassId in0_cls = egraph.findConst(planner.baseState.nodeToEClass.at(in0));
        EClassId in1_cls = egraph.findConst(planner.baseState.nodeToEClass.at(in1));
        EClassId neg_cls = egraph.findConst(planner.baseState.nodeToEClass.at(neg_in1));

        // Implementation A (slower, cost 2.5ms) uses neg_cls
        ENode enode_slow(KernelId{0xED0001}, OpType::FUSED, "fma_v1", {in0_cls, neg_cls}, {8, 8}, {8, 1},
                         DType::FLOAT32, MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}});
        // Implementation B (faster, cost 0.5ms) also uses neg_cls
        ENode enode_fast(KernelId{0xED0002}, OpType::FUSED, "fma_v2", {in0_cls, neg_cls}, {8, 8}, {8, 1},
                         DType::FLOAT32, MemSpace{1, HandleType::CPP}, {Engine{0, EngineType::CPU}});

        egraph.addENode(rootEClass, enode_slow);
        egraph.addENode(rootEClass, enode_fast);

        auto enodeInfos = planner.computeENodeInfos(egraph, eclassToLogical, cachedNodes, false);
        planner.pruneEGraph(egraph, enodeInfos);

        bool found_slow = false;
        bool found_fast = false;
        for (ENodeId enodeId : egraph.getEClass(rootEClass).enodes)
        {
            const ENode &n = egraph.getENode(enodeId);
            if (n.getKernelId() == KernelId{0xED0001})
                found_slow = true;
            if (n.getKernelId() == KernelId{0xED0002})
                found_fast = true;
        }

        if (found_slow || !found_fast)
        {
            Error::throw_err("[ENodeDominationTest: Test 3] Domination pruning failed in death cascade test!");
        }

        std::cout << "  (Test 3 - Death Cascade Integration Passed)" << std::endl;
    }
}