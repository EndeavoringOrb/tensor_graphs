#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryTriuF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.triu(inputs[0], inputs[1]);
}
