#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryCastF32_Bool_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.cast(inputs[0], DType::BOOL);
}
