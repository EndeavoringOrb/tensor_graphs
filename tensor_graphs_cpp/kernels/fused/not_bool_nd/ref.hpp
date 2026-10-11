#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryNotBool_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.logical_not(inputs[0]);
}
