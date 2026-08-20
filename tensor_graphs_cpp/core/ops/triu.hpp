#pragma once
#include "core/misc.hpp"
#include "core/ops/common.hpp"

struct TriuOp
{
    static constexpr OpType op_type = OpType::TRIU;
    static constexpr const char *name = "TRIU";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        graph.getNode(nodeId).setShape(graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape());
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        if (parentRegions.size() > 1 && !parentRegions[1].empty())
            return makeFull(node.getShape());
        return mergeRegions(parentRegions[0]);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &graph,
                                                           const std::vector<Region> &outputRegions)
    {
        return {mergeRegions(outputRegions), makeFull(graph.getNode(node.child_ids[1]).getShape())};
    }

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, 0.0);
    }

    static bool isConstant(uint64_t inputIdx, uint64_t)
    {
        return inputIdx == 1;
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.triu(pInputs[0], pInputs[1]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};