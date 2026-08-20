#pragma once
#include "core/ops/common.hpp"

struct CopyToOp : public ElementwiseUnaryOp<CopyToOp>
{
    static constexpr OpType op_type = OpType::COPY_TO;
    static constexpr const char *name = "COPY_TO";

    static void inferShape(LogicalId nodeId, Graph &graph)
    {
        graph.getNode(nodeId).setShape(graph.getNode(graph.getNode(nodeId).child_ids[0]).getShape());
        graph.getNode(nodeId).strides = graph.getNode(graph.getNode(nodeId).child_ids[0]).strides;
    }

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, 0.0);
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph._copyto(pInputs[0]);
    }
};