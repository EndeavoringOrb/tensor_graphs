#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryKreaModulatedRMSNormV2(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    LogicalId w = inputs[1];
    LogicalId scale = inputs[2];
    LogicalId shift = inputs[3];

    auto shape = g.getNode(x).getShape();
    uint32_t B = shape[0];
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    LogicalId x_sq = g.mul(x, x);
    LogicalId sum_sq = g.sum(x_sq, -1);
    LogicalId mean_sq = g.div(sum_sq, g.fill(static_cast<float>(D), {B, S, 1}));
    LogicalId std = g.pow(g.add(mean_sq, g.fill(1e-6f, {B, S, 1})), g.fill(0.5f, {B, S, 1}));
    LogicalId inv_std = g.repeat(g.div(g.fill(1.0f, {B, S, 1}), std), D, 2);
    LogicalId x_norm = g.mul(x, inv_std);

    LogicalId w_3d = g.reshape(w, {1, 1, static_cast<int32_t>(D)});
    LogicalId w_exp = g.repeat(g.repeat(w_3d, B, 0), S, 1);
    LogicalId one_full = g.fill(1.0f, {B, S, D});
    LogicalId w_scale = g.add(w_exp, one_full);
    LogicalId x_scaled = g.mul(x_norm, w_scale);

    LogicalId one = g.fill(1.0f, {B, S, D});
    LogicalId one_plus_scale = g.add(one, scale);
    LogicalId scaled_norm = g.mul(one_plus_scale, x_scaled);
    return g.add(scaled_norm, shift);
}
