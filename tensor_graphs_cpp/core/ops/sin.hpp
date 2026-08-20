#pragma once
#include "core/ops/common.hpp"

struct SinOp : public ElementwiseUnaryOp<SinOp>
{
    static constexpr OpType op_type = OpType::SIN;
    static constexpr const char *name = "SIN";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.sin(pInputs[0]);
    }
};