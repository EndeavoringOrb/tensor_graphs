#pragma once
#include "core/misc.hpp"
#include "core/ops/common.hpp"

struct ScatterOp
{
    static constexpr OpType op_type = OpType::SCATTER;
    static constexpr const char *name = "SCATTER";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        const auto &node = graph.getNode(nodeId);
        const std::vector<int32_t> shape = graph.getConstantInt32(node.child_ids[4]);
        if (shape.empty())
            Error::throw_err("[ScatterOp.inferShape] scatter output shape must not be empty");

        std::vector<uint32_t> output_shape;
        output_shape.reserve(shape.size());
        for (int32_t dim : shape)
        {
            if (dim <= 0)
                Error::throw_err("[ScatterOp.inferShape] scatter output dimensions must be positive");
            output_shape.push_back(static_cast<uint32_t>(dim));
        }
        graph.getNode(nodeId).setShape(output_shape);
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        if (!parentRegions[1].empty() || !parentRegions[2].empty() || !parentRegions[3].empty() ||
            !parentRegions[4].empty())
            return makeFull(node.getShape());

        const auto &updateRegions = parentRegions[0];
        if (updateRegions.empty())
            return {};

        auto starts = graph.getConstantInt32(node.child_ids[1]);
        auto ends = graph.getConstantInt32(node.child_ids[2]);
        auto steps = graph.getConstantInt32(node.child_ids[3]);

        std::vector<Region> outBoxes;
        for (const auto &region : updateRegions)
            outBoxes.push_back(mapSliceRegionBackward(region, node.getShape(), starts, ends, steps));
        return mergeRegions(outBoxes);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &graph,
                                                           const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}, {}, {}};

        auto starts = graph.getConstantInt32(node.child_ids[1]);
        auto ends = graph.getConstantInt32(node.child_ids[2]);
        auto steps = graph.getConstantInt32(node.child_ids[3]);

        std::vector<Region> updateBoxes;
        for (const auto &region : outputRegions)
            updateBoxes.push_back(mapSliceRegionForward(region, node.getShape(), starts, ends, steps));

        return {mergeRegions(updateBoxes), makeFull(graph.getNode(node.child_ids[1]).getShape()),
                makeFull(graph.getNode(node.child_ids[2]).getShape()),
                makeFull(graph.getNode(node.child_ids[3]).getShape()),
                makeFull(graph.getNode(node.child_ids[4]).getShape())};
    }

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, 0.0);
    }

    static bool isConstant(uint64_t inputIdx, uint64_t)
    {
        return inputIdx >= 1;
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.scatter(pInputs[0], pInputs[1], pInputs[2], pInputs[3], pInputs[4]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};
