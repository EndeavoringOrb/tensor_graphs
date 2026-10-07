#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryKreaPerHeadRMSNormRoPE(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];        // [1, H, S, D]
    LogicalId w = inputs[1];        // [D]
    LogicalId cos_node = inputs[2]; // [1, 1, S, D/2]
    LogicalId sin_node = inputs[3]; // [1, 1, S, D/2]

    auto sX = g.getNode(x).getShape();
    uint32_t H = sX[1];
    uint32_t S = sX[2];
    uint32_t D = sX[3];
    uint32_t half_dim = D / 2;

    LogicalId x_sq = g.mul(x, x);
    int32_t axis_val = -1;
    LogicalId sum_sq = g.sum(x_sq, g.constant({1}, &axis_val, DType::INT32));
    LogicalId mean_sq = g.div(sum_sq, g.fill(static_cast<float>(D), {1, H, S, 1}));
    LogicalId std = g.pow(g.add(mean_sq, g.fill(1e-6f, {1, H, S, 1})), g.fill(0.5f, {1, H, S, 1}));
    LogicalId inv_std = g.repeat(g.div(g.fill(1.0f, {1, H, S, 1}), std), D, 3);
    LogicalId x_norm = g.mul(x, inv_std);

    LogicalId w_4d = g.reshape(w, {1, 1, 1, D});
    LogicalId w_exp = g.repeat(g.repeat(w_4d, H, 1), S, 2);
    LogicalId one_full = g.fill(1.0f, {1, H, S, D});
    LogicalId scale = g.add(w_exp, one_full);
    LogicalId x_scaled = g.mul(x_norm, scale);

    LogicalId x_5d = g.reshape(x_scaled, {1, H, S, half_dim, 2});
    LogicalId x_even =
        g.contiguous(g.slice(x_5d, {0, 0, 0, 0, 0},
                             {1, static_cast<int32_t>(H), static_cast<int32_t>(S), static_cast<int32_t>(half_dim), 1}));
    x_even = g.reshape(x_even, {1, H, S, half_dim});
    LogicalId x_odd =
        g.contiguous(g.slice(x_5d, {0, 0, 0, 0, 1},
                             {1, static_cast<int32_t>(H), static_cast<int32_t>(S), static_cast<int32_t>(half_dim), 2}));
    x_odd = g.reshape(x_odd, {1, H, S, half_dim});

    LogicalId cos_exp = g.repeat(cos_node, H, 1);
    LogicalId sin_exp = g.repeat(sin_node, H, 1);

    LogicalId x_rot_even = g.add(g.mul(x_even, cos_exp), g.neg(g.mul(x_odd, sin_exp)));
    LogicalId x_rot_odd = g.add(g.mul(x_even, sin_exp), g.mul(x_odd, cos_exp));

    LogicalId e_5d = g.reshape(x_rot_even, {1, H, S, half_dim, 1});
    LogicalId o_5d = g.reshape(x_rot_odd, {1, H, S, half_dim, 1});
    LogicalId pair_5d = g.concat({e_5d, o_5d}, 4);
    return g.reshape(pair_5d, {1, H, S, D});
}
