#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryNegF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 1)
        Error::throw_err("Negate ND requires 1 input");

    return graph.neg(inputs[0]);
}
