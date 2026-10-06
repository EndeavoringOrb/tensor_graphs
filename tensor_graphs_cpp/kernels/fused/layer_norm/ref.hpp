#pragma once

#include "core/kernels.hpp"

static constexpr float LAYERNORM_DEFAULT_EPS = 1e-6f;

inline LogicalId ref_ln_expand_scalar_1S1(Graph &g, float val, uint32_t S)
{
    LogicalId node = g.constant({1}, &val, DType::FLOAT32);
    int32_t sh3[] = {1, 1, 1};
    LogicalId out = g.reshape(node, g.constant({3}, sh3, DType::INT32));
    if (S > 1)
    {
        int32_t rep = (int32_t)S;
        int32_t axis = 1;
        out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &axis, DType::INT32));
    }
    return out;
}

inline LogicalId ref_ln_repeat_ax2(Graph &g, LogicalId node, uint32_t D)
{
    if (D <= 1)
        return node;
    int32_t rep = (int32_t)D;
    int32_t axis = 2;
    return g.repeat(node, g.constant({1}, &rep, DType::INT32), g.constant({1}, &axis, DType::INT32));
}

inline LogicalId refFactoryLayerNorm(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x_id = inputs[0];
    const auto &shape = graph.getNode(x_id).getShape();
    uint32_t B = shape[0];
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    // axis = -1 (reduce over last dimension)
    int32_t ax_val = -1;
    LogicalId ax_node = graph.constant({1}, &ax_val, DType::INT32);

    // --- Mean ---
    // sum(x, axis=-1) -> {B, S, 1}
    LogicalId sum_x = graph.sum(x_id, ax_node);

    // D as float, expanded to {1, S, 1}
    float d_float = (float)D;
    LogicalId d_node = ref_ln_expand_scalar_1S1(graph, d_float, S);

    // mean_val = sum(x, -1) / D -> {B, S, 1}
    LogicalId mean_val = graph.div(sum_x, d_node);

    // mean = repeat_ax(mean_val, D, 2) -> {B, S, D}
    LogicalId mean = ref_ln_repeat_ax2(graph, mean_val, D);

    // --- Centered ---
    // x_sub = x + neg(mean) = x - mean
    LogicalId x_sub = graph.add(x_id, graph.neg(mean));

    // --- Variance ---
    // sq = x_sub * x_sub
    LogicalId sq = graph.mul(x_sub, x_sub);

    // sum(sq, axis=-1) -> {B, S, 1}
    LogicalId sum_sq = graph.sum(sq, ax_node);

    // var = sum(sq, -1) / D -> {B, S, 1}
    LogicalId var = graph.div(sum_sq, d_node);

    // --- Standard Deviation ---
    // var + eps -> {B, S, 1}
    float eps = LAYERNORM_DEFAULT_EPS;
    LogicalId eps_node = ref_ln_expand_scalar_1S1(graph, eps, S);
    LogicalId var_plus_eps = graph.add(var, eps_node);

    // std = pow(var + eps, 0.5) -> {B, S, 1}
    float half_val = 0.5f;
    LogicalId sqrt_exp = ref_ln_expand_scalar_1S1(graph, half_val, S);
    LogicalId std_dev = graph.pow(var_plus_eps, sqrt_exp);

    // --- Inverse Std ---
    // 1.0 / std -> {B, S, 1}
    float one_val = 1.0f;
    LogicalId one_node = ref_ln_expand_scalar_1S1(graph, one_val, S);
    LogicalId inv_std = graph.div(one_node, std_dev);

    // repeat_ax(inv_std, D, 2) -> {B, S, D}
    LogicalId inv_std_expanded = ref_ln_repeat_ax2(graph, inv_std, D);

    // --- Result ---
    // x_sub * inv_std_expanded
    return graph.mul(x_sub, inv_std_expanded);
}
