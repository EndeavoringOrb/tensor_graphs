#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactorySinF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.sin(inputs[0]);
}
