#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryFluxRMSNorm4D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x_id = inputs[0];
    LogicalId weight_id = inputs[1];
    auto shape = g.getNode(x_id).getShape();
    uint32_t B = shape[0], H = shape[1], S = shape[2], D = shape[3];

    // sq = x * x
    LogicalId x_sq = g.mul(x_id, x_id);

    // sum_sq = sum(sq, axis=-1)
    int32_t axis_val = -1;
    LogicalId axis_node = g.constant({1}, &axis_val, DType::INT32);
    LogicalId sum_sq = g.sum(x_sq, axis_node);

    auto expand_to_4d_broadcast = [&](float val, uint32_t last_d) {
        int32_t sh[] = {1, 1, 1, 1};
        LogicalId out = g.reshape(g.constant({1}, &val, DType::FLOAT32), g.constant({4}, sh, DType::INT32));
        if (B > 1)
        {
            int32_t r = B, a = 0;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        if (H > 1)
        {
            int32_t r = H, a = 1;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        if (S > 1)
        {
            int32_t r = S, a = 2;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        if (last_d > 1)
        {
            int32_t r = last_d, a = 3;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        return out;
    };

    // mean_sq = sum_sq / HeadDim
    LogicalId head_dim_const = expand_to_4d_broadcast((float)D, 1);
    LogicalId mean_sq = g.div(sum_sq, head_dim_const);

    // std = pow(mean_sq + 1e-6, 0.5)
    LogicalId eps_node = expand_to_4d_broadcast(1e-6f, 1);
    LogicalId half_node = expand_to_4d_broadcast(0.5f, 1);
    LogicalId std_dev = g.pow(g.add(mean_sq, eps_node), half_node);

    // inv_std = 1.0 / std_dev (repeated across D)
    LogicalId one_node = expand_to_4d_broadcast(1.0f, 1);
    LogicalId inv_std_scalar = g.div(one_node, std_dev);
    int32_t r_d = D, a_d = 3;
    LogicalId inv_std =
        g.repeat(inv_std_scalar, g.constant({1}, &r_d, DType::INT32), g.constant({1}, &a_d, DType::INT32));

    // x_norm = x * inv_std
    LogicalId x_norm = g.mul(x_id, inv_std);

    // w_exp = reshape and repeat weight to match [B, H, S, D]
    int32_t sh_w[] = {1, 1, 1, (int32_t)D};
    LogicalId w_exp = g.reshape(weight_id, g.constant({4}, sh_w, DType::INT32));
    if (H > 1)
    {
        int32_t r = H, a = 1;
        w_exp = g.repeat(w_exp, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }
    if (S > 1)
    {
        int32_t r = S, a = 2;
        w_exp = g.repeat(w_exp, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }
    // Batch repeat usually handled by strides/0-stride view, but for ref pattern
    // compatibility:
    if (B > 1)
    {
        int32_t r = B, a = 0;
        w_exp = g.repeat(w_exp, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }

    return g.mul(x_norm, w_exp);
}
