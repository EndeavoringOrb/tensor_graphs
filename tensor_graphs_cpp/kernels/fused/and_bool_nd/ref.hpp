#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryAndBoolND(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.logical_and(inputs[0], inputs[1]);
}
