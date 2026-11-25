#pragma once
#include "core/kernels.hpp"

inline LogicalId refFactoryPermuteCUDA(const std::vector<LogicalId> &inputs, Graph &g)
{
    return g.permute(inputs[0], inputs[1]);
}
