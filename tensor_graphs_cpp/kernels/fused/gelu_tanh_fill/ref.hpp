#pragma once

#include "core/kernels.hpp"
#include "core/common/constants.hpp"

inline LogicalId refFactoryGeluTanhFill(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    auto shape = g.getNode(x).getShape();

    LogicalId x_sq = g.mul(x, x);
    LogicalId x_cube = g.mul(x_sq, x);
    LogicalId c1 = g.fill(0.044715f, shape);
    LogicalId term1 = g.mul(x_cube, c1);
    LogicalId term2 = g.add(x, term1);
    LogicalId c2 = g.fill(0.79788456f, shape);
    LogicalId inner = g.mul(term2, c2);

    LogicalId neg_two = g.fill(-2.0f, shape);
    LogicalId neg_2u = g.mul(inner, neg_two);
    LogicalId exp_neg_2u = g.pow(g.fill(TGConstants::E, shape), neg_2u);
    LogicalId one = g.fill(1.0f, shape);
    LogicalId two = g.fill(2.0f, shape);
    LogicalId den = g.add(one, exp_neg_2u);
    LogicalId tanh_val = g.add(g.div(two, den), g.neg(one));

    LogicalId one_plus_tanh = g.add(one, tanh_val);
    LogicalId half_x = g.mul(x, g.fill(0.5f, shape));
    return g.mul(half_x, one_plus_tanh);
}
