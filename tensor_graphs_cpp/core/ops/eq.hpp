#pragma once
#include "core/ops/common.hpp"

struct EqOp : public ElementwiseBinaryOp<EqOp>
{
    static constexpr OpType op_type = OpType::EQ;
    static constexpr const char *name = "EQ";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.eq(pInputs[0], pInputs[1]);
    }
};