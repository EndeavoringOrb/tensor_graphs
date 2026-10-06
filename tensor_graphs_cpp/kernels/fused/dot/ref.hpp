#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryDotF32(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Dot requires 2 inputs");
    return graph.dot(inputs[0], inputs[1]);
}
