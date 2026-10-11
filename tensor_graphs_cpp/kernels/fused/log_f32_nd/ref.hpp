#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryLogF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.log(inputs[0]);
}
