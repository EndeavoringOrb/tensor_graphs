#pragma once
#include "core/ops/common.hpp"

struct AddOp : public ElementwiseBinaryOp<AddOp>
{
    static constexpr OpType op_type = OpType::ADD;
    static constexpr const char *name = "ADD";

    static LogicalId buildPattern(Graph &pGraph, const std::vector<LogicalId> &pInputs, DType)
    {
        return pGraph.add(pInputs[0], pInputs[1]);
    }
};