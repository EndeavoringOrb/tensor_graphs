#pragma once
#include "core/ops/common.hpp"

struct ContiguousOp : public ElementwiseUnaryOp<ContiguousOp>
{
    static constexpr OpType op_type = OpType::CONTIGUOUS;
    static constexpr const char *name = "CONTIGUOUS";

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, 0.0);
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.contiguous(pInputs[0]);
    }
};