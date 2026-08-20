#pragma once
#include "core/ops/common.hpp"

struct NegateOp : public ElementwiseUnaryOp<NegateOp>
{
    static constexpr OpType op_type = OpType::NEGATE;
    static constexpr const char *name = "NEGATE";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.neg(pInputs[0]);
    }
};