#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryJinaLayerNormWB_F32_3D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x_id = inputs[0];
    LogicalId w_id = inputs[1];
    LogicalId b_id = inputs[2];

    const auto &shape = g.getNode(x_id).getShape();
    uint32_t B = shape[0];
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    // Helper: expand_scalar_to_3d(val, 1, S, 1) → {1, S, 1}
    // Mirrors JinaV5OmniNanoRetrievalModel::expand_scalar_to_3d exactly.
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

    // Helper: repeat_3d_axis(node, D, 2) → broadcast axis 2 by D
    auto repeat_d_axis2 = [&](LogicalId node) -> LogicalId {
        int32_t rep = (int32_t)D;
        int32_t ax = 2;
        return g.repeat(node, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));
    };

    // Helper: expand_1d_to_3d(vec, D, 1, S) → {1, S, D}
    // Mirrors JinaV5OmniNanoRetrievalModel::expand_1d_to_3d exactly.
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

    // --- axis = -1 ---
    int32_t ax_val = -1;
    LogicalId axis_node = g.constant({1}, &ax_val, DType::INT32);

    // --- mean ---
    LogicalId sum_x = g.sum(x_id, axis_node);      // {B, S, 1}
    float d_float = (float)D;                      // 768.0f
    LogicalId d_node = expand_scalar_1S1(d_float); // {1, S, 1}
    LogicalId mean_val = g.div(sum_x, d_node);     // {B, S, 1}
    LogicalId mean = repeat_d_axis2(mean_val);     // {B, S, D}

    // --- x - mean ---
    LogicalId x_sub = g.add(x_id, g.neg(mean)); // {B, S, D}

    // --- variance ---
    LogicalId sq = g.mul(x_sub, x_sub);      // {B, S, D}
    LogicalId sum_sq = g.sum(sq, axis_node); // {B, S, 1}
    LogicalId var = g.div(sum_sq, d_node);   // {B, S, 1}

    // --- std = sqrt(var + eps) ---
    LogicalId eps_node = expand_scalar_1S1(1e-6f);     // {1, S, 1}
    LogicalId var_plus_eps = g.add(var, eps_node);     // {B, S, 1}
    LogicalId sqrt_exp = expand_scalar_1S1(0.5f);      // {1, S, 1}
    LogicalId std_dev = g.pow(var_plus_eps, sqrt_exp); // {B, S, 1}

    // --- inv_std = 1 / std ---
    LogicalId one_node = expand_scalar_1S1(1.0f);    // {1, S, 1}
    LogicalId inv_std = g.div(one_node, std_dev);    // {B, S, 1}
    LogicalId inv_std_exp = repeat_d_axis2(inv_std); // {B, S, D}

    // --- normalize ---
    LogicalId normalized = g.mul(x_sub, inv_std_exp); // {B, S, D}

    // --- apply weight ---
    LogicalId w_exp = expand_1d_1SD(w_id); // {1, S, D}
    normalized = g.mul(normalized, w_exp); // {B, S, D}

    // --- apply bias ---
    LogicalId b_exp = expand_1d_1SD(b_id); // {1, S, D}
    return g.add(normalized, b_exp);       // {B, S, D}
}
