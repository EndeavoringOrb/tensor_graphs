#pragma once
#include "core/ops/common.hpp"

struct LtOp : public ElementwiseBinaryOp<LtOp>
{
    static constexpr OpType op_type = OpType::LT;
    static constexpr const char *name = "LT";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.lt(pInputs[0], pInputs[1]);
    }
};