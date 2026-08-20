#pragma once
#include "core/ops/common.hpp"

struct NotOp : public ElementwiseUnaryOp<NotOp>
{
    static constexpr OpType op_type = OpType::NOT;
    static constexpr const char *name = "NOT";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.logical_not(pInputs[0]);
    }
};