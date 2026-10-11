#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryCastBF16_F32(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.cast(inputs[0], DType::FLOAT32);
}
