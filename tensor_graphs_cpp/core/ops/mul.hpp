#pragma once
#include "core/ops/common.hpp"

struct MulOp : public ElementwiseBinaryOp<MulOp>
{
    static constexpr OpType op_type = OpType::MUL;
    static constexpr const char *name = "MUL";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.mul(pInputs[0], pInputs[1]);
    }
};