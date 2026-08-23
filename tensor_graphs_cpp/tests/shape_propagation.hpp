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

void runShapePropagationTests()
{
    std::cout << "shape propagation tests" << std::endl << std::flush;
    ShapePropagator prop;
    auto makeIntConst = [](Graph &graph, const std::vector<int32_t> &values) -> LogicalId {
        return graph.constant({(uint32_t)values.size()}, values.data(), DType::INT32);
    };
    auto makeFloatInput = [](Graph &graph, const std::vector<uint32_t> &shape) -> LogicalId {
        return graph.input(shape, DType::FLOAT32, {});
    };
    {
        Graph graph;
        LogicalId x = makeFloatInput(graph, {4, 5});
        LogicalId axis = makeIntConst(graph, {1});
        LogicalId sumId = graph.sum(x, axis);
        prop.inferShapeRecursive(sumId, graph);
        auto forward = prop.forward(graph.getNode(sumId), graph, {{makeRegion({{1, 3}, {2, 4}})}, {}});
        assertRegionListEquals(forward, {makeRegion({{1, 3}, {0, 1}})}, "SUM forward");
        auto backward = prop.backward(graph.getNode(sumId), graph, {makeRegion({{1, 3}, {0, 1}})});
        assertRegionListEquals(backward[0], {makeRegion({{1, 3}, {0, 5}})}, "SUM backward input");
        assertRegionListEquals(backward[1], makeFull(graph.getNode(axis).getShape()), "SUM backward axis");
    }
    {
        Graph graph;
        LogicalId x = makeFloatInput(graph, {4, 5});
        LogicalId axis = makeIntConst(graph, {1});
        LogicalId maxId = graph.max(x, axis);
        prop.inferShapeRecursive(maxId, graph);
        auto forward = prop.forward(graph.getNode(maxId), graph, {{makeRegion({{0, 4}, {1, 3}})}, {}});
        assertRegionListEquals(forward, {makeRegion({{0, 4}, {0, 1}})}, "MAX forward");
        auto backward = prop.backward(graph.getNode(maxId), graph, {makeRegion({{0, 4}, {0, 1}})});
        assertRegionListEquals(backward[0], {makeRegion({{0, 4}, {0, 5}})}, "MAX backward input");
    }
    {
        Graph graph;
        LogicalId x = makeFloatInput(graph, {2, 3});
        LogicalId dims = makeIntConst(graph, {1, 0});
        LogicalId permId = graph.permute(x, dims);
        prop.inferShapeRecursive(permId, graph);
        auto forward = prop.forward(graph.getNode(permId), graph,
                                    {{makeRegion({{0, 1}, {1, 3}}), makeRegion({{1, 2}, {0, 2}})}, {}});
        assertRegionListEquals(forward,
                               {
                                   makeRegion({{1, 3}, {0, 1}}),
                                   makeRegion({{0, 2}, {1, 2}}),
                               },
                               "PERMUTE forward");
        auto backward =
            prop.backward(graph.getNode(permId), graph, {makeRegion({{1, 3}, {0, 1}}), makeRegion({{0, 2}, {1, 2}})});
        assertRegionListEquals(backward[0],
                               {
                                   makeRegion({{0, 1}, {1, 3}}),
                                   makeRegion({{1, 2}, {0, 2}}),
                               },
                               "PERMUTE backward input");
    }
    {
        Graph graph;
        LogicalId a = makeFloatInput(graph, {2, 2});
        LogicalId b = makeFloatInput(graph, {2, 2});
        LogicalId axis = makeIntConst(graph, {0});
        LogicalId concatId = graph.concat({a, b}, axis);
        prop.inferShapeRecursive(concatId, graph);
        auto forward = prop.forward(graph.getNode(concatId), graph,
                                    {{}, {makeRegion({{0, 1}, {0, 2}})}, {makeRegion({{1, 2}, {1, 2}})}});
        assertRegionListEquals(forward,
                               {
                                   makeRegion({{0, 1}, {0, 2}}),
                                   makeRegion({{3, 4}, {1, 2}}),
                               },
                               "CONCAT forward");
        auto backward =
            prop.backward(graph.getNode(concatId), graph, {makeRegion({{0, 1}, {0, 2}}), makeRegion({{3, 4}, {1, 2}})});
        assertRegionListEquals(backward[1], {makeRegion({{0, 1}, {0, 2}})}, "CONCAT backward left");
        assertRegionListEquals(backward[2], {makeRegion({{1, 2}, {1, 2}})}, "CONCAT backward right");
    }
    {
        Graph graph;
        LogicalId x = makeFloatInput(graph, {2, 2});
        LogicalId repeats = makeIntConst(graph, {3});
        LogicalId axis = makeIntConst(graph, {0});
        LogicalId repeatId = graph.repeat(x, repeats, axis);
        prop.inferShapeRecursive(repeatId, graph);
        auto forward = prop.forward(graph.getNode(repeatId), graph, {{makeRegion({{1, 2}, {0, 2}})}, {}, {}});
        assertRegionListEquals(forward, {makeRegion({{3, 6}, {0, 2}})}, "REPEAT forward");
        auto backward = prop.backward(graph.getNode(repeatId), graph, {makeRegion({{3, 6}, {0, 2}})});
        assertRegionListEquals(backward[0], {makeRegion({{1, 2}, {0, 2}})}, "REPEAT backward input");
    }
    {
        Graph graph;
        LogicalId x = makeFloatInput(graph, {8});
        LogicalId starts = makeIntConst(graph, {2});
        LogicalId ends = makeIntConst(graph, {6});
        LogicalId steps = makeIntConst(graph, {1});
        LogicalId sliceId = graph.slice(x, starts, ends, steps);
        prop.inferShapeRecursive(sliceId, graph);
        auto forward =
            prop.forward(graph.getNode(sliceId), graph, {{makeRegion({{2, 3}}), makeRegion({{5, 6}})}, {}, {}, {}});
        assertRegionListEquals(forward, {makeRegion({{0, 1}}), makeRegion({{3, 4}})}, "SLICE forward");
        auto backward = prop.backward(graph.getNode(sliceId), graph, {makeRegion({{0, 1}}), makeRegion({{3, 4}})});
        assertRegionListEquals(backward[0], {makeRegion({{2, 3}}), makeRegion({{5, 6}})}, "SLICE backward input");
    }
    {
        Graph graph;
        LogicalId target = makeFloatInput(graph, {8});
        LogicalId updates = makeFloatInput(graph, {4});
        LogicalId starts = makeIntConst(graph, {2});
        LogicalId ends = makeIntConst(graph, {6});
        LogicalId steps = makeIntConst(graph, {1});
        LogicalId scatterId = graph.scatter(target, updates, starts, ends, steps);
        prop.inferShapeRecursive(scatterId, graph);
        auto forward =
            prop.forward(graph.getNode(scatterId), graph, {{makeRegion({{0, 2}})}, {makeRegion({{1, 3}})}, {}, {}, {}});
        assertRegionListEquals(forward, {makeRegion({{0, 2}}), makeRegion({{3, 5}})}, "SCATTER forward");
        auto backward = prop.backward(graph.getNode(scatterId), graph, {makeRegion({{3, 5}})});
        assertRegionListEquals(backward[0], {makeRegion({{3, 5}})}, "SCATTER backward target");
        assertRegionListEquals(backward[1], {makeRegion({{1, 3}})}, "SCATTER backward updates");
    }
    {
        Graph graph;
        LogicalId data = makeFloatInput(graph, {4, 3});
        LogicalId idx = makeIntConst(graph, {2});
        LogicalId gatherId = graph.gather(data, idx);
        prop.inferShapeRecursive(gatherId, graph);
        auto forward =
            prop.forward(graph.getNode(gatherId), graph, {{makeRegion({{1, 3}, {0, 3}})}, {makeRegion({{0, 2}})}});
        assertRegionListEquals(forward, {makeRegion({{0, 2}, {0, 3}})}, "GATHER forward");
        auto backward = prop.backward(graph.getNode(gatherId), graph, {makeRegion({{0, 2}, {1, 3}})});
        assertRegionListEquals(backward[0], {makeRegion({{2, 3}, {1, 3}})}, "GATHER backward data");
        assertRegionListEquals(backward[1], {makeRegion({{0, 2}})}, "GATHER backward idx");
    }
    {
        Graph graph;
        LogicalId data = makeFloatInput(graph, {4, 3});
        LogicalId idxSrc = makeIntConst(graph, {1, 3, 0, 2});
        // LogicalId sliceStart = makeIntConst(graph, {0});
        // LogicalId sliceEnd = makeIntConst(graph, {2});
        // LogicalId sliceStep = makeIntConst(graph, {1});
        // LogicalId idx = graph.slice(idxSrc, sliceStart, sliceEnd, sliceStep);
        // prop.inferShapeRecursive(idx, graph);
        LogicalId gatherId = graph.gather(data, idxSrc);
        prop.inferShapeRecursive(gatherId, graph);
        auto backward = prop.backward(graph.getNode(gatherId), graph, {makeRegion({{0, 2}, {0, 3}})});
        assertRegionListEquals(backward[0], {makeRegion({{1, 2}, {0, 3}}), makeRegion({{3, 4}, {0, 3}})},
                               "GATHER backward sliced indices data");
        assertRegionListEquals(backward[1], {makeRegion({{0, 2}})}, "GATHER backward sliced indices idx");
    }
}