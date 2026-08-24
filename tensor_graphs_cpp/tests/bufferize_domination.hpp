// File: tensor_graphs_cpp/tests/bufferize_domination.hpp
#pragma once

#include <cmath>
#include <iostream>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/misc.hpp"
#include "core/plan/planner.hpp"
#include "core/plan/validators/mem.hpp"
#include "core/types.hpp"

#include "tests/common.hpp"

inline void runBufferizeDominationTests()
{
    std::cout << "bufferize domination optimality tests" << std::endl << std::flush;
    CostModel costModel(false, "");
    std::unordered_map<MemSpace, uint64_t> mem_caps = {{MemSpace{1, HandleType::CPP}, 1024ULL * 1024 * 1024}};

    // Register reference kernels with safe_inplace_idxs enabled for testing
    static bool registered = false;
    if (!registered)
    {
        KernelRegistry::get().registerKernel(
            KernelId{0xBB0001}, OpType::ADD, "", 2, 2, nullptr, nullptr, nullptr, {0, 1}, false, true, nullptr,
            MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
            {{8, 8}, {8, 8}}, {false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

        KernelRegistry::get().registerKernel(
            KernelId{0xBB0002}, OpType::MUL, "", 2, 2, nullptr, nullptr, nullptr, {0, 1}, false, true, nullptr,
            MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
            {{8, 8}, {8, 8}}, {false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

        KernelRegistry::get().registerKernel(KernelId{0xBB0003}, OpType::NEGATE, "", 1, 1, nullptr, nullptr, nullptr,
                                             {0}, false, true, nullptr, MemSpace(1, HandleType::CPP),
                                             {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{8, 8}}, {false},
                                             {{MemSpace(1, HandleType::CPP)}});

        registered = true;
    }

    for (uint64_t kid : {0xBB0001ULL, 0xBB0002ULL, 0xBB0003ULL})
    {
        Record r;
        r.kernelId = KernelId{kid};
        r.buildContextId = BUILD_CONTEXT_ID;
        r.hwTag = HW_TAG;
        r.outputShape = {8, 8};
        r.outputStrides = {8, 1};
        r.outputDType = DType::FLOAT32;
        r.inputShapes = {{8, 8}, {8, 8}};
        r.inputStrides = {{8, 1}, {8, 1}};
        r.inputDTypes = {DType::FLOAT32, DType::FLOAT32};
        r.runTime = 1.0f;
        costModel.records[r.kernelId].push_back(r);
    }

    // -------------------------------------------------------------------------
    // Test Case 1: Linear Chain & Commutative In-Place Symmetry
    // -------------------------------------------------------------------------
    {
        Graph graph;
        LogicalId in0 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId in1 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId b1 = graph.add(in0, in1);
        LogicalId b2 = graph.mul(in0, in1);
        LogicalId sum_node = graph.add(b1, b2);
        LogicalId root = graph.neg(sum_node);

        std::vector<LogicalId> topo = topologicalSort({root}, graph);
        Planner planner(costModel, mem_caps);
        planner.initBaseEGraph(root, graph, topo, nullptr);
        populateDummyRecords(costModel, planner.baseState.egraph);

        EGraph egraph = planner.baseState.egraph;
        std::unordered_map<EClassId, LogicalId> eclassToLogical = planner.baseState.eclassToLogical;
        std::unordered_map<LogicalId, MemSpace> cachedNodes;
        auto enodeInfos = planner.computeENodeInfos(egraph, eclassToLogical, cachedNodes, false);

        // Select the enode with safe_inplace_idxs
        std::unordered_map<EClassId, uint32_t> selection_map;
        for (const auto &cls : egraph.getClasses())
        {
            EClassId canon = egraph.findConst(cls.id);
            if (egraph.getEClass(canon).enodes.empty())
                continue;

            uint32_t chosen_idx = 0;
            for (uint32_t e_idx = 0; e_idx < egraph.getEClass(canon).enodes.size(); ++e_idx)
            {
                ENodeId enode_id = egraph.getEClass(canon).enodes[e_idx];
                const ENode &enode = egraph.getENode(enode_id);
                if (KernelRegistry::get().hasKernel(enode.getKernelId()))
                {
                    const auto &k = KernelRegistry::get().getKernel(enode.getKernelId());
                    if (!k.safe_inplace_idxs.empty())
                    {
                        chosen_idx = e_idx;
                        break;
                    }
                }
            }
            selection_map[canon] = chosen_idx;
        }

        // Get a valid dispatch order
        auto dispatch_iterator = makeDispatchIterator(egraph, selection_map, enodeInfos);
        std::vector<EClassId> order;
        if (!dispatch_iterator.getNextDispatchOrder(selection_map, order))
        {
            Error::throw_err("[BufferizeDominationTest] Failed to get dispatch order");
        }

        // Unconstrained BufferizeIterator
        BufferizeIterator iterUnconstrained(order, egraph, selection_map, enodeInfos, nullptr);
        iterUnconstrained.clearDominationRules();

        uint32_t count_unconstrained = 0;
        float min_cost_unconstrained = TGConstants::INF;
        std::vector<ParallelBuffer> bufs;
        std::unordered_map<EClassId, BufferId> eclass_to_buf;

        while (iterUnconstrained.getNextBufferization(bufs, eclass_to_buf))
        {
            count_unconstrained++;
            BufferId overflow;
            std::vector<ParallelBuffer> allocated;
            if (malloc_by_time_components(mem_caps.at(MemSpace{1, HandleType::CPP}), bufs, allocated, overflow))
            {
                float cost = get_cost(order, egraph, selection_map, enodeInfos);
                min_cost_unconstrained = std::min(min_cost_unconstrained, cost);
            }
        }

        // BufferizeIterator with all domination rules
        BufferizeIterator iterAllRules(order, egraph, selection_map, enodeInfos, nullptr);
        iterAllRules.addDominationRule(std::make_shared<MemSpaceMismatchInplaceRule>());
        iterAllRules.addDominationRule(std::make_shared<LinearChainInplaceDominationRule>());
        iterAllRules.addDominationRule(std::make_shared<IntervalSubsetDominationRule>());
        iterAllRules.addDominationRule(std::make_shared<CommutativeInplaceSymmetryRule>());

        uint32_t count_all_rules = 0;
        float min_cost_all_rules = TGConstants::INF;

        while (iterAllRules.getNextBufferization(bufs, eclass_to_buf))
        {
            count_all_rules++;
            BufferId overflow;
            std::vector<ParallelBuffer> allocated;
            if (malloc_by_time_components(mem_caps.at(MemSpace{1, HandleType::CPP}), bufs, allocated, overflow))
            {
                float cost = get_cost(order, egraph, selection_map, enodeInfos);
                min_cost_all_rules = std::min(min_cost_all_rules, cost);
            }
        }

        if (min_cost_all_rules == TGConstants::INF || min_cost_unconstrained == TGConstants::INF)
        {
            Error::throw_err("[BufferizeDominationTest: Test 1] Failed to allocate in memory");
        }

        if (std::abs(min_cost_all_rules - min_cost_unconstrained) > 1e-5f)
        {
            Error::throw_err("[BufferizeDominationTest: Test 1] Cost mismatch! unconstrained=" +
                             std::to_string(min_cost_unconstrained) +
                             ", dominated=" + std::to_string(min_cost_all_rules));
        }

        if (count_all_rules > count_unconstrained)
        {
            Error::throw_err(
                "[BufferizeDominationTest: Test 1] Domination rules explored more states than unconstrained!");
        }

        std::cout << "  (Test 1 - Linear Chain & Symmetry: Unconstrained=" << count_unconstrained
                  << ", Dominated=" << count_all_rules << ", Cost=" << min_cost_all_rules << " ms)" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test Case 2: Interval-Subset Domination (Different Birth Times)
    // -------------------------------------------------------------------------
    {
        Graph graph;
        LogicalId in0 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId in1 = graph.input({8, 8}, DType::FLOAT32);
        LogicalId t1 = graph.add(in0, in1);
        LogicalId t2 = graph.mul(in0, in1);
        LogicalId t3 = graph.add(t1, t2);
        LogicalId root = graph.neg(t3);

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
            if (egraph.getEClass(canon).enodes.empty())
                continue;

            uint32_t chosen_idx = 0;
            for (uint32_t e_idx = 0; e_idx < egraph.getEClass(canon).enodes.size(); ++e_idx)
            {
                ENodeId enode_id = egraph.getEClass(canon).enodes[e_idx];
                const ENode &enode = egraph.getENode(enode_id);
                if (KernelRegistry::get().hasKernel(enode.getKernelId()))
                {
                    const auto &k = KernelRegistry::get().getKernel(enode.getKernelId());
                    if (!k.safe_inplace_idxs.empty())
                    {
                        chosen_idx = e_idx;
                        break;
                    }
                }
            }
            selection_map[canon] = chosen_idx;
        }

        auto dispatch_iterator = makeDispatchIterator(egraph, selection_map, enodeInfos);
        std::vector<EClassId> order;
        if (!dispatch_iterator.getNextDispatchOrder(selection_map, order))
        {
            Error::throw_err("[BufferizeDominationTest] Failed to get dispatch order");
        }

        // Unconstrained
        BufferizeIterator iterUnconstrained(order, egraph, selection_map, enodeInfos, nullptr);
        iterUnconstrained.clearDominationRules();

        uint32_t count_unconstrained = 0;
        float min_cost_unconstrained = TGConstants::INF;
        std::vector<ParallelBuffer> bufs;
        std::unordered_map<EClassId, BufferId> eclass_to_buf;

        while (iterUnconstrained.getNextBufferization(bufs, eclass_to_buf))
        {
            count_unconstrained++;
            BufferId overflow;
            std::vector<ParallelBuffer> allocated;
            if (malloc_by_time_components(mem_caps.at(MemSpace{1, HandleType::CPP}), bufs, allocated, overflow))
            {
                float cost = get_cost(order, egraph, selection_map, enodeInfos);
                min_cost_unconstrained = std::min(min_cost_unconstrained, cost);
            }
        }

        // All Rules
        BufferizeIterator iterAllRules(order, egraph, selection_map, enodeInfos, nullptr);
        iterAllRules.addDominationRule(std::make_shared<MemSpaceMismatchInplaceRule>());
        iterAllRules.addDominationRule(std::make_shared<LinearChainInplaceDominationRule>());
        iterAllRules.addDominationRule(std::make_shared<IntervalSubsetDominationRule>());
        iterAllRules.addDominationRule(std::make_shared<CommutativeInplaceSymmetryRule>());

        uint32_t count_all_rules = 0;
        float min_cost_all_rules = TGConstants::INF;

        while (iterAllRules.getNextBufferization(bufs, eclass_to_buf))
        {
            count_all_rules++;
            BufferId overflow;
            std::vector<ParallelBuffer> allocated;
            if (malloc_by_time_components(mem_caps.at(MemSpace{1, HandleType::CPP}), bufs, allocated, overflow))
            {
                float cost = get_cost(order, egraph, selection_map, enodeInfos);
                min_cost_all_rules = std::min(min_cost_all_rules, cost);
            }
        }

        if (min_cost_all_rules == TGConstants::INF || min_cost_unconstrained == TGConstants::INF)
        {
            Error::throw_err("[BufferizeDominationTest: Test 2] Failed to allocate in memory");
        }

        if (std::abs(min_cost_all_rules - min_cost_unconstrained) > 1e-5f)
        {
            Error::throw_err("[BufferizeDominationTest: Test 2] Cost mismatch! unconstrained=" +
                             std::to_string(min_cost_unconstrained) +
                             ", dominated=" + std::to_string(min_cost_all_rules));
        }

        if (count_all_rules > count_unconstrained)
        {
            Error::throw_err(
                "[BufferizeDominationTest: Test 2] Domination rules explored more states than unconstrained!");
        }

        std::cout << "  (Test 2 - Interval Subset: Unconstrained=" << count_unconstrained
                  << ", Dominated=" << count_all_rules << ", Cost=" << min_cost_all_rules << " ms)" << std::endl;
    }
}
