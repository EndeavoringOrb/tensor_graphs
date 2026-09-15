#pragma once

#include <iostream>
#include <vector>

#include "core/cost_model.hpp"
#include "core/egraph.hpp"
#include "core/graph.hpp"
#include "core/misc.hpp"
#include "core/plan/planner.hpp"
#include "core/types.hpp"

inline void runInputHashconsTests()
{
    std::cout << "input hashcons tests" << std::endl << std::flush;
    CostModel costModel(false, "");

    // -------------------------------------------------------------------------
    // Test 1: Distinct runtime inputs with IDENTICAL shape/dtype must NOT merge
    // -------------------------------------------------------------------------
    {
        Graph graph;
        LogicalId in0 = graph.input({128, 128}, DType::FLOAT32);
        LogicalId in1 = graph.input({128, 128}, DType::FLOAT32);
        LogicalId addNode = graph.add(in0, in1);

        std::vector<LogicalId> topo = topologicalSort({addNode}, graph);
        Planner planner(costModel);
        planner.initBaseEGraph(addNode, graph, topo, nullptr);

        EClassId cls0 = planner.baseState.egraph.findConst(planner.baseState.nodeToEClass.at(in0));
        EClassId cls1 = planner.baseState.egraph.findConst(planner.baseState.nodeToEClass.at(in1));

        if (cls0 == cls1)
        {
            Error::throw_err("[InputHashcons] Distinct runtime inputs with matching shapes were incorrectly merged!");
        }

        const ENode &addENode = planner.baseState.egraph.getENode(
            planner.baseState.egraph.getEClass(planner.baseState.nodeToEClass.at(addNode)).enodes[0]);
        if (addENode.getChildren()[0] == addENode.getChildren()[1])
        {
            Error::throw_err("[InputHashcons] ADD node children collapsed onto the same input EClass!");
        }
    }

    // -------------------------------------------------------------------------
    // Test 2: Same runtime input reused across branches MUST resolve to same EClass
    // -------------------------------------------------------------------------
    {
        Graph graph;
        LogicalId in0 = graph.input({64, 64}, DType::FLOAT32);
        LogicalId addSelf = graph.add(in0, in0);

        std::vector<LogicalId> topo = topologicalSort({addSelf}, graph);
        Planner planner(costModel);
        planner.initBaseEGraph(addSelf, graph, topo, nullptr);

        const ENode &selfAddENode = planner.baseState.egraph.getENode(
            planner.baseState.egraph.getEClass(planner.baseState.nodeToEClass.at(addSelf)).enodes[0]);

        if (selfAddENode.getChildren()[0] != selfAddENode.getChildren()[1])
        {
            Error::throw_err("[InputHashcons] Identical input reused in binary op did not resolve to same EClass!");
        }
    }

    // -------------------------------------------------------------------------
    // Test 3: Identical constants SHOULD be deduplicated / merged
    // -------------------------------------------------------------------------
    {
        Graph graph;
        std::vector<int32_t> data = {1, 2, 3, 4};
        LogicalId c0 = graph.constant({4}, data.data(), DType::INT32);
        LogicalId c1 = graph.constant({4}, data.data(), DType::INT32);

        // At graph level, constant deduplication should return the same LogicalId
        if (c0 != c1)
        {
            Error::throw_err("[InputHashcons] Graph::constant did not deduplicate identical constant payloads!");
        }

        // In EGraph level, constants with identical contentHash should merge
        EGraph egraph;
        std::vector<uint8_t> byteData(data.size() * sizeof(int32_t));
        std::memcpy(byteData.data(), data.data(), byteData.size());

        EClassId ec0 = egraph.getOrAddConstant({4}, {1}, DType::INT32, byteData);
        EClassId ec1 = egraph.getOrAddConstant({4}, {1}, DType::INT32, byteData);

        if (egraph.findConst(ec0) != egraph.findConst(ec1))
        {
            Error::throw_err("[InputHashcons] EGraph::getOrAddConstant failed to deduplicate identical constants!");
        }
    }

    // -------------------------------------------------------------------------
    // Test 4: Different constants must NOT merge
    // -------------------------------------------------------------------------
    {
        Graph graph;
        std::vector<int32_t> data0 = {1, 2, 3, 4};
        std::vector<int32_t> data1 = {5, 6, 7, 8};
        LogicalId c0 = graph.constant({4}, data0.data(), DType::INT32);
        LogicalId c1 = graph.constant({4}, data1.data(), DType::INT32);

        if (c0 == c1)
        {
            Error::throw_err("[InputHashcons] Different constant payloads returned identical LogicalId!");
        }

        EGraph egraph;
        std::vector<uint8_t> b0(data0.size() * sizeof(int32_t));
        std::vector<uint8_t> b1(data1.size() * sizeof(int32_t));
        std::memcpy(b0.data(), data0.data(), b0.size());
        std::memcpy(b1.data(), data1.data(), b1.size());

        EClassId ec0 = egraph.getOrAddConstant({4}, {1}, DType::INT32, b0);
        EClassId ec1 = egraph.getOrAddConstant({4}, {1}, DType::INT32, b1);

        if (egraph.findConst(ec0) == egraph.findConst(ec1))
        {
            Error::throw_err("[InputHashcons] Different constant values were incorrectly merged in EGraph!");
        }
    }

    // -------------------------------------------------------------------------
    // Test 5: Runtime input vs Constant with identical shape/dtype must NOT merge
    // -------------------------------------------------------------------------
    {
        Graph graph;
        std::vector<float> constData = {1.0f, 2.0f, 3.0f, 4.0f};
        LogicalId in = graph.input({4}, DType::FLOAT32);
        LogicalId c = graph.constant({4}, constData.data(), DType::FLOAT32);
        LogicalId addNode = graph.add(in, c);

        std::vector<LogicalId> topo = topologicalSort({addNode}, graph);
        Planner planner(costModel);
        planner.initBaseEGraph(addNode, graph, topo, nullptr);

        EClassId inCls = planner.baseState.egraph.findConst(planner.baseState.nodeToEClass.at(in));
        EClassId constCls = planner.baseState.egraph.findConst(planner.baseState.nodeToEClass.at(c));

        if (inCls == constCls)
        {
            Error::throw_err(
                "[InputHashcons] Runtime input and constant were incorrectly merged into the same EClass!");
        }
    }

    // -------------------------------------------------------------------------
    // Test 6: Heterogeneous input DAG (Multiple runtime inputs + shared constants)
    // -------------------------------------------------------------------------
    {
        Graph graph;
        LogicalId inA = graph.input({8, 8}, DType::FLOAT32);
        LogicalId inB = graph.input({8, 8}, DType::FLOAT32);
        LogicalId inC = graph.input({8, 8}, DType::FLOAT32);

        std::vector<int32_t> axis = {-1};
        LogicalId cAxis0 = graph.constant({1}, axis.data(), DType::INT32);
        LogicalId cAxis1 = graph.constant({1}, axis.data(), DType::INT32); // identical to cAxis0

        LogicalId sumA = graph.sum(inA, cAxis0);
        LogicalId sumB = graph.sum(inB, cAxis1);
        LogicalId sumC = graph.sum(inC, cAxis0);

        LogicalId root = graph.add(graph.add(sumA, sumB), sumC);

        std::vector<LogicalId> topo = topologicalSort({root}, graph);
        Planner planner(costModel);
        planner.initBaseEGraph(root, graph, topo, nullptr);

        EClassId eA = planner.baseState.egraph.findConst(planner.baseState.nodeToEClass.at(inA));
        EClassId eB = planner.baseState.egraph.findConst(planner.baseState.nodeToEClass.at(inB));
        EClassId eC = planner.baseState.egraph.findConst(planner.baseState.nodeToEClass.at(inC));

        if (eA == eB || eA == eC || eB == eC)
        {
            Error::throw_err("[InputHashcons] Runtime inputs in multi-branch DAG were incorrectly merged!");
        }

        EClassId eAxis0 = planner.baseState.egraph.findConst(planner.baseState.nodeToEClass.at(cAxis0));
        EClassId eAxis1 = planner.baseState.egraph.findConst(planner.baseState.nodeToEClass.at(cAxis1));

        if (eAxis0 != eAxis1)
        {
            Error::throw_err("[InputHashcons] Identical axis constants across branches were not merged!");
        }
    }
}