#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryContiguous(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.contiguous(inputs[0]);
}
