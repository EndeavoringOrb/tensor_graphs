#pragma once
#include "core/ops/common.hpp"

struct CosOp : public ElementwiseUnaryOp<CosOp>
{
    static constexpr OpType op_type = OpType::COS;
    static constexpr const char *name = "COS";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.cos(pInputs[0]);
    }
};