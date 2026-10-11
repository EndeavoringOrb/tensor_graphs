#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryOrBool_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.logical_or(inputs[0], inputs[1]);
}
