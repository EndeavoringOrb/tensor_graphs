#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryConcatF32(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() < 2)
        Error::throw_err("Concat requires at least 2 inputs");

    std::vector<LogicalId> tensors(inputs.begin() + 1, inputs.end());
    return graph.concat(tensors, inputs[0]);
}
