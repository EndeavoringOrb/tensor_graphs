#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryGatherBF16(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId casted = graph.cast(inputs[0], DType::FLOAT32);
    return graph.gather(casted, inputs[1]);
}
