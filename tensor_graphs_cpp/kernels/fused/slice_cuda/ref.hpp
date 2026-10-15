#pragma once
#include "core/kernels.hpp"

inline LogicalId refFactorySliceCUDA(const std::vector<LogicalId> &inputs, Graph &g)
{
    return g.slice(inputs[0], inputs[1], inputs[2], inputs[3]);
}
