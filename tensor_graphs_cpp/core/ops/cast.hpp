#pragma once
#include "core/ops/common.hpp"

struct CastOp : public ElementwiseUnaryOp<CastOp>
{
    static constexpr OpType op_type = OpType::CAST;
    static constexpr const char *name = "CAST";

    static WorkloadMetrics computeWorkload(const std::vector<std::vector<uint32_t>> &inShapes,
                                           const std::vector<DType> &inDTypes, const std::vector<uint32_t> &outShape,
                                           DType outDType, const std::string &)
    {
        return op_common::defaultWorkload(inShapes, inDTypes, outShape, outDType, 0.0);
    }

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType dtype)
    {
        return pGraph.cast(pInputs[0], dtype);
    }
};