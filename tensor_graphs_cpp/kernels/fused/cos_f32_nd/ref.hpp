#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryCosF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.cos(inputs[0]);
}
