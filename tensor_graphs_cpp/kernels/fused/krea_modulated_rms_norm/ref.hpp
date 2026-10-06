#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryKreaModulatedRMSNorm(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    LogicalId w = inputs[1];
    LogicalId scale = inputs[2];
    LogicalId shift = inputs[3];

    auto shape = g.getNode(x).getShape();
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    LogicalId x_sq = g.mul(x, x);
    LogicalId sum_sq = g.sum(x_sq, -1);
    LogicalId mean_sq = g.div(sum_sq, g.fill((float)D, {1, S, 1}));
    LogicalId std = g.pow(g.add(mean_sq, g.fill(1e-6f, {1, S, 1})), g.fill(0.5f, {1, S, 1}));
    LogicalId inv_std = g.repeat(g.div(g.fill(1.0f, {1, S, 1}), std), D, 2);
    LogicalId x_norm = g.mul(x, inv_std);

    LogicalId w_exp = g.repeat(g.reshape(w, {1, 1, (int32_t)D}), S, 1);
    LogicalId x_scaled = g.mul(x_norm, w_exp);

    LogicalId one = g.fill(1.0f, {1, S, D});
    LogicalId one_plus_scale = g.add(one, scale);
    LogicalId scaled_norm = g.mul(one_plus_scale, x_scaled);
    return g.add(scaled_norm, shift);
}
