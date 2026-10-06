#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryGatherF32_I32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.gather(inputs[0], inputs[1]);
}
