#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryExp3DFill(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    auto shape = g.getNode(x).getShape();
    LogicalId e_node = g.fill(TGConstants::E, shape);
    return g.pow(e_node, x);
}
