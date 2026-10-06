#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryArangeI32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.arange(inputs[0], inputs[1], inputs[2]);
}
