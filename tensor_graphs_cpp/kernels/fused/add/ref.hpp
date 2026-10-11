#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryAdd(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.add(inputs[0], inputs[1]);
}
