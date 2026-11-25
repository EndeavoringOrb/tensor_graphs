#pragma once

#include "core/kernels.hpp"
#include <cmath>

inline LogicalId refFactoryGemmaAttentionEx(LogicalId q, LogicalId k, LogicalId v, LogicalId mask_id,
                                            uint32_t n_heads, uint32_t seq_len, uint32_t head_dim, Graph &g)
{
    float scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim));
    LogicalId scale_node =
        g.fill(g.constant({1}, &scale_val, DType::FLOAT32), {n_heads, seq_len, head_dim});
    LogicalId scaled_q = g.mul(q, scale_node);

    int32_t perm_k[] = {0, 2, 1};
    LogicalId k_t = g.permute(k, g.constant({3}, perm_k, DType::INT32));
    k_t = g.contiguous(k_t);

    LogicalId scores = g.dot(scaled_q, k_t);
    LogicalId mask_expanded = g.repeat(mask_id, n_heads, 0);
    scores = g.add(scores, mask_expanded);

    int32_t axis_val = -1;
    LogicalId max_scores = g.max(scores, g.constant({1}, &axis_val, DType::INT32));
    max_scores = g.repeat(max_scores, seq_len, 2);
    LogicalId shifted_scores = g.add(scores, g.neg(max_scores));

    float e_val = 2.718281828459045f;
    LogicalId e_node = g.fill(g.constant({1}, &e_val, DType::FLOAT32), {n_heads, seq_len, seq_len});
    LogicalId exp_scores = g.pow(e_node, shifted_scores);

    LogicalId sum_exp = g.sum(exp_scores, g.constant({1}, &axis_val, DType::INT32));
    sum_exp = g.repeat(sum_exp, seq_len, 2);

    LogicalId probs = g.div(exp_scores, sum_exp);
    LogicalId context = g.dot(probs, v);

    int32_t ctx_shape4[] = {1, static_cast<int32_t>(n_heads), static_cast<int32_t>(seq_len), static_cast<int32_t>(head_dim)};
    LogicalId ctx_4d = g.reshape(context, g.constant({4}, ctx_shape4, DType::INT32));

    int32_t perm_ctx[] = {0, 2, 1, 3};
    LogicalId ctx_perm = g.permute(ctx_4d, g.constant({4}, perm_ctx, DType::INT32));
    ctx_perm = g.contiguous(ctx_perm);

    int32_t ctx_shape3[] = {1, static_cast<int32_t>(seq_len), static_cast<int32_t>(n_heads * head_dim)};
    return g.reshape(ctx_perm, g.constant({3}, ctx_shape3, DType::INT32));
}

inline LogicalId refFactoryGemmaAttention(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId q = inputs[0];
    LogicalId k = inputs[1];
    LogicalId v = inputs[2];
    LogicalId mask_id = inputs[3];
    const auto &q_shape = g.getNode(q).getShape();
    return refFactoryGemmaAttentionEx(q, k, v, mask_id, q_shape[0], q_shape[1], q_shape[2], g);
}
