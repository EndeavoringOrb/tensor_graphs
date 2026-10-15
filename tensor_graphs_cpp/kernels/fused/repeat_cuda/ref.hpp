#pragma once
#include "core/kernels.hpp"

inline LogicalId refFactoryRepeatCUDA(const std::vector<LogicalId> &inputs, Graph &g)
{
    return g.repeat(inputs[0], inputs[1], inputs[2]);
}
