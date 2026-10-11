#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryLtF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.lt(inputs[0], inputs[1]);
}
