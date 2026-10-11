#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryJinaRMSNorm_F32_3D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x_id = inputs[0];
    LogicalId w_id = inputs[1];

    const auto &shape = g.getNode(x_id).getShape();
    uint32_t B = shape[0];
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    auto expand_scalar_1S1 = [&](float val) -> LogicalId {
        LogicalId node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh[] = {1, 1, 1};
        LogicalId out = g.reshape(node, g.constant({3}, sh, DType::INT32));
        if (S > 1)
        {
            int32_t rep = (int32_t)S;
            int32_t ax = 1;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    auto repeat_d_axis2 = [&](LogicalId node) -> LogicalId {
        int32_t rep = (int32_t)D;
        int32_t ax = 2;
        return g.repeat(node, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));
    };

    auto expand_1d_1SD = [&](LogicalId vec) -> LogicalId {
        int32_t sh[] = {1, 1, (int32_t)D};
        LogicalId out = g.reshape(vec, g.constant({3}, sh, DType::INT32));
        if (S > 1)
        {
            int32_t rep = (int32_t)S;
            int32_t ax = 1;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    // x_sq = x * x
    LogicalId x_sq = g.mul(x_id, x_id);

    // sum_sq = sum(x_sq, axis=-1)
    int32_t ax_val = -1;
    LogicalId axis_node = g.constant({1}, &ax_val, DType::INT32);
    LogicalId sum_sq = g.sum(x_sq, axis_node);

    // mean_sq = sum_sq / D
    float d_float = (float)D;
    LogicalId n_node = expand_scalar_1S1(d_float);
    LogicalId mean_sq = g.div(sum_sq, n_node);

    // std = sqrt(mean_sq + eps)
    LogicalId eps_node = expand_scalar_1S1(1e-5f);
    LogicalId mean_sq_plus_eps = g.add(mean_sq, eps_node);
    LogicalId sqrt_node = expand_scalar_1S1(0.5f);
    LogicalId std = g.pow(mean_sq_plus_eps, sqrt_node);

    // inv_std = 1 / std
    LogicalId one_node = expand_scalar_1S1(1.0f);
    LogicalId inv_std = g.div(one_node, std);
    LogicalId inv_std_expanded = repeat_d_axis2(inv_std);

    // x_norm = x * inv_std
    LogicalId x_norm = g.mul(x_id, inv_std_expanded);

    // apply weight
    LogicalId w_exp = expand_1d_1SD(w_id);
    return g.mul(x_norm, w_exp);
}
