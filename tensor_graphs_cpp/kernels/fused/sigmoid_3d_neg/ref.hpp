#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactorySigmoid_3D_Neg(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x = inputs[0];
    auto shape = graph.getNode(x).getShape();
    LogicalId neg_x = graph.neg(x);
    LogicalId e_node = graph.fill(TGConstants::E, shape);
    LogicalId exp_neg_x = graph.pow(e_node, neg_x);
    LogicalId one = graph.fill(1.0f, shape);
    LogicalId den = graph.add(one, exp_neg_x);
    return graph.div(one, den);
}
