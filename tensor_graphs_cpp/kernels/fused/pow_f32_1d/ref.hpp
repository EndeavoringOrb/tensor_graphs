#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryPowF32_1D(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.pow(inputs[0], inputs[1]);
}
