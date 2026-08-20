#pragma once
#include "core/ops/common.hpp"

struct AndOp : public ElementwiseBinaryOp<AndOp>
{
    static constexpr OpType op_type = OpType::AND;
    static constexpr const char *name = "AND";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.logical_and(pInputs[0], pInputs[1]);
    }
};