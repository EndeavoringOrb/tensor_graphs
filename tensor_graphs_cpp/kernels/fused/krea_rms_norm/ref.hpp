#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryKreaRmsNorm(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    LogicalId w = inputs[1];
    auto shape = g.getNode(x).getShape();
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    LogicalId x_sq = g.mul(x, x);
    LogicalId sum_sq = g.sum(x_sq, -1);
    LogicalId mean_sq = g.div(sum_sq, g.fill(static_cast<float>(D), {1, S, 1}));
    LogicalId std = g.pow(g.add(mean_sq, g.fill(1e-6f, {1, S, 1})), g.fill(0.5f, {1, S, 1}));
    LogicalId inv_std = g.repeat(g.div(g.fill(1.0f, {1, S, 1}), std), D, 2);
    LogicalId x_norm = g.mul(x, inv_std);

    LogicalId w_exp = g.repeat(g.reshape(w, {1, 1, D}), S, 1);
    return g.mul(x_norm, w_exp);
}
