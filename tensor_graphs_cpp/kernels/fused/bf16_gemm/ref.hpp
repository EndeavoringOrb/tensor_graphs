#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryBF16GEMM_NEON(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId w_f32 = graph.cast(inputs[1], DType::FLOAT32);
    return graph.dot(inputs[0], w_f32);
}
