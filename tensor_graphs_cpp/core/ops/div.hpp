#pragma once
#include "core/ops/common.hpp"

struct DivideOp : public ElementwiseBinaryOp<DivideOp>
{
    static constexpr OpType op_type = OpType::DIVIDE;
    static constexpr const char *name = "DIVIDE";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.div(pInputs[0], pInputs[1]);
    }
};