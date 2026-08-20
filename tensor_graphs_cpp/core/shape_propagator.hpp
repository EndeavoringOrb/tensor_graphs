#pragma once
#include "core/ops/ops.hpp"
#include "core/shapes.hpp"

struct ShapePropagator
{
    void inferShapeRecursive(LogicalId nodeId, Graph &graph)
    {
        if (!graph.hasNode(nodeId))
            return;

        if (!graph.getNode(nodeId).getShape().empty())
            return;

        if (graph.getNode(nodeId).opType == OpType::INPUT)
            return;

        for (LogicalId pid : graph.getNode(nodeId).child_ids)
        {
            inferShapeRecursive(pid, graph);
        }

        inferShape(nodeId, graph);
    }

    void inferShape(LogicalId nodeId, Graph &graph)
    {
        if (!graph.hasNode(nodeId) || !graph.getNode(nodeId).getShape().empty())
            return;
        if (graph.getNode(nodeId).opType == OpType::INPUT)
            return;

        const auto &traits = getOpTraits(graph.getNode(nodeId).opType);
        if (traits.inferShape)
            traits.inferShape(nodeId, graph);
        else
            Error::throw_err("[ShapePropagator.inferShape] Unsupported OpType: " +
                             toString(graph.getNode(nodeId).opType));

        for (auto d : graph.getNode(nodeId).getShape())
        {
            if (d == 0)
                Error::throw_err("Zero-sized dimension in tensor shape!" + toString(graph.getNode(nodeId), graph, ""));
        }
    }

    std::vector<Region> forward(const TensorNode &node, const Graph &graph,
                                const std::vector<std::vector<Region>> &parentRegions)
    {
        const auto &traits = getOpTraits(node.opType);
        if (traits.forwardRegion)
            return traits.forwardRegion(node, graph, parentRegions);
        Error::throw_err("[ShapePropagator.forward] Unsupported OpType: " + toString(node.opType));
    }

    std::vector<std::vector<Region>> backward(const TensorNode &node, const Graph &graph,
                                              const std::vector<Region> &outputRegions)
    {
        const auto &traits = getOpTraits(node.opType);
        if (traits.backwardRegion)
            return traits.backwardRegion(node, graph, outputRegions);
        Error::throw_err("[ShapePropagator.backward] Unsupported OpType: " + toString(node.opType));
    }
};