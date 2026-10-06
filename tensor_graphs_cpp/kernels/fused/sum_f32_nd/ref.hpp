#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactorySumF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Sum ND requires 2 inputs");

    return graph.sum(inputs[0], inputs[1]);
}
