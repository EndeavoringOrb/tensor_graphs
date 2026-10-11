#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryCastBool_I32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.cast(inputs[0], DType::INT32);
}
