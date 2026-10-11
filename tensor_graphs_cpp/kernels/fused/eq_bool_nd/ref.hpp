#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryEqBool_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.eq(inputs[0], inputs[1]);
}
