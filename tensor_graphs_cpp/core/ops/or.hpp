#pragma once
#include "core/ops/common.hpp"

struct OrOp : public ElementwiseBinaryOp<OrOp>
{
    static constexpr OpType op_type = OpType::OR;
    static constexpr const char *name = "OR";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.logical_or(pInputs[0], pInputs[1]);
    }
};