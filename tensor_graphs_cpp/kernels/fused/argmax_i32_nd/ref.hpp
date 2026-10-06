#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryArgmaxI32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.argmax(inputs[0], inputs[1], inputs[2]);
}
