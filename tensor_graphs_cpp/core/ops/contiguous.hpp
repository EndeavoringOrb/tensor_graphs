#pragma once
#include "core/ops/common.hpp"

struct ContiguousOp
{
    static constexpr OpType op_type = OpType::CONTIGUOUS;
    static constexpr const char *name = "CONTIGUOUS";
    static constexpr bool is_elementwise = false;

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        graph.getNode(nodeId).setShape(graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape());
        graph.getNode(nodeId).strides = calcContiguousStrides(graph.getNode(nodeId).getShape());
    }

    static std::vector<Region> forwardRegion(const TensorNode &node, const Graph &graph,
                                             const std::vector<std::vector<Region>> &parentRegions)
    {
        return forwardElementwise(node, graph, parentRegions);
    }

    static std::vector<std::vector<Region>> backwardRegion(const TensorNode &node, const Graph &,
                                                           const std::vector<Region> &outputRegions)
    {
        return backwardElementwise(node.child_ids.size(), outputRegions);
    }

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, 0.0);
    }

    static bool isConstant(uint64_t, uint64_t)
    {
        return false;
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.contiguous(pInputs[0]);
    }

    static OpTraits traits()
    {
        return OpTraits{op_type,         name,       is_elementwise, inferShape, forwardRegion, backwardRegion,
                        computeWorkload, isConstant, buildPattern};
    }
};