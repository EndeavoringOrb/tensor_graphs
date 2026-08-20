#pragma once
#include "core/ops/common.hpp"

struct PowerOp : public ElementwiseBinaryOp<PowerOp>
{
    static constexpr OpType op_type = OpType::POWER;
    static constexpr const char *name = "POWER";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.pow(pInputs[0], pInputs[1]);
    }
};