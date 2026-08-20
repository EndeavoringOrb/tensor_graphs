#pragma once
#include "core/ops/common.hpp"

struct ArangeOp
{
    static constexpr OpType op_type = OpType::ARANGE;
    static constexpr const char *name = "ARANGE";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        int32_t start = graph.getConstantInt32(graph.getNode(nodeId).child_ids[0])[0];
        int32_t stop = graph.getConstantInt32(graph.getNode(nodeId).child_ids[1])[0];
        int32_t step = graph.getConstantInt32(graph.getNode(nodeId).child_ids[2])[0];
        graph.getNode(nodeId).setShape({static_cast<uint32_t>(std::max(0, (stop - start + step - 1) / step))});
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        return forwardFull(node, graph, parentRegions);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &graph,
                                                           const std::vector<Region> &outputRegions)
    {
        return backwardFull(node, graph, outputRegions);
    }

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, 0.0);
    }

    static bool isConstant(uint64_t inputIdx, uint64_t)
    {
        return inputIdx == 0 || inputIdx == 1 || inputIdx == 2;
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.arange(pInputs[0], pInputs[1], pInputs[2]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};