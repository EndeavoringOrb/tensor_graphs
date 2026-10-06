#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryScatterF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.scatter(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
}
