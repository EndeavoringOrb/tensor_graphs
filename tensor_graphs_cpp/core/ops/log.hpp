#pragma once
#include "core/ops/common.hpp"

struct LogOp : public ElementwiseUnaryOp<LogOp>
{
    static constexpr OpType op_type = OpType::LOG;
    static constexpr const char *name = "LOG";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.log(pInputs[0]);
    }
};