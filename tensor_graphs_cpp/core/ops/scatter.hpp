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
        graph.getNode(nodeId).setShape(graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape());
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        if (!parentRegions[2].empty() || !parentRegions[3].empty() || !parentRegions[4].empty())
            return makeFull(node.getShape());

        const auto &targetRegions = parentRegions[0];
        const auto &updateRegions = parentRegions[1];
        if (targetRegions.empty() && updateRegions.empty())
            return {};

        const auto &targetShape = graph.getNode(node.child_ids[0]).getShape();
        auto starts = graph.getConstantInt32(node.child_ids[2]);
        auto ends = graph.getConstantInt32(node.child_ids[3]);
        auto steps = graph.getConstantInt32(node.child_ids[4]);

        std::vector<Region> outBoxes;
        for (const auto &region : targetRegions)
            outBoxes.push_back(region);
        for (const auto &region : updateRegions)
            outBoxes.push_back(mapSliceRegionBackward(region, targetShape, starts, ends, steps));
        return mergeRegions(outBoxes);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &graph,
                                                           const std::vector<Region> &outputRegions)
    {
        if (outputRegions.empty())
            return {{}, {}, {}, {}, {}};

        const auto &targetShape = graph.getNode(node.child_ids[0]).getShape();
        auto starts = graph.getConstantInt32(node.child_ids[2]);
        auto ends = graph.getConstantInt32(node.child_ids[3]);
        auto steps = graph.getConstantInt32(node.child_ids[4]);

        std::vector<Region> targetBoxes;
        std::vector<Region> updateBoxes;
        for (const auto &region : outputRegions)
        {
            targetBoxes.push_back(region);
            updateBoxes.push_back(mapSliceRegionForward(region, targetShape, starts, ends, steps));
        }

        return {mergeRegions(targetBoxes), mergeRegions(updateBoxes),
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
        return inputIdx == 2 || inputIdx == 3 || inputIdx == 4;
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