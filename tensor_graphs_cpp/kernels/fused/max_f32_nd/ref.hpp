#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryMaxF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Max ND requires 2 inputs");

    return graph.max(inputs[0], inputs[1]);
}
