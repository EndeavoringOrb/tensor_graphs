#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryMulF32(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Mul requires 2 inputs");
    return graph.mul(inputs[0], inputs[1]);
}
