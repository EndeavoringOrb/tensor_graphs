#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/constants.hpp"
#include "core/fused_row_restriction.hpp"
#include "core/kernels.hpp"

// softmax(q k^T / sqrt(D) + mask) v for q [H, Sq, D], k and v [H, Sk, D] and an
// additive mask [1, Sq, Sk] shared by all heads; output [1, Sq, H * D]. With
// Sq == Sk this is refFactoryGemmaAttentionEx; a row restriction (fewer query
// rows than keys) slices q and the mask rows together.
inline LogicalId refFactoryMaskedAttention(const std::vector<LogicalId> &inputs, Graph &g)
{
    const LogicalId q = inputs[0], k = inputs[1], v = inputs[2], mask = inputs[3];
    const auto &q_shape = g.getNode(q).getShape();
    const uint32_t heads = q_shape[0], q_rows = q_shape[1], head_dim = q_shape[2];
    const uint32_t keys = g.getNode(k).getShape()[1];

    float scale_val = 1.0f / std::sqrt(static_cast<float>(head_dim));
    LogicalId scale_node = g.fill(g.constant({1}, &scale_val, DType::FLOAT32), {heads, q_rows, head_dim});
    LogicalId scaled_q = g.mul(q, scale_node);

    int32_t perm_k[] = {0, 2, 1};
    LogicalId k_t = g.contiguous(g.permute(k, g.constant({3}, perm_k, DType::INT32)));
    LogicalId scores = g.dot(scaled_q, k_t);
    scores = g.add(scores, g.repeat(mask, heads, 0));

    int32_t axis_val = -1;
    LogicalId max_scores = g.max(scores, g.constant({1}, &axis_val, DType::INT32));
    max_scores = g.repeat(max_scores, keys, 2);
    LogicalId shifted_scores = g.add(scores, g.neg(max_scores));

    float e_val = 2.718281828459045f;
    LogicalId e_node = g.fill(g.constant({1}, &e_val, DType::FLOAT32), {heads, q_rows, keys});
    LogicalId exp_scores = g.pow(e_node, shifted_scores);
    LogicalId sum_exp = g.sum(exp_scores, g.constant({1}, &axis_val, DType::INT32));
    sum_exp = g.repeat(sum_exp, keys, 2);
    LogicalId context = g.dot(g.div(exp_scores, sum_exp), v);

    int32_t ctx_shape4[] = {1, static_cast<int32_t>(heads), static_cast<int32_t>(q_rows), static_cast<int32_t>(head_dim)};
    LogicalId ctx_4d = g.reshape(context, g.constant({4}, ctx_shape4, DType::INT32));
    int32_t perm_ctx[] = {0, 2, 1, 3};
    LogicalId ctx_perm = g.contiguous(g.permute(ctx_4d, g.constant({4}, perm_ctx, DType::INT32)));
    int32_t ctx_shape3[] = {1, static_cast<int32_t>(q_rows), static_cast<int32_t>(heads * head_dim)};
    return g.reshape(ctx_perm, g.constant({3}, ctx_shape3, DType::INT32));
}

inline bool matchMaskedAttention(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 4)
        return false;
    const auto &q = inputs[0].getShape(), &k = inputs[1].getShape(), &v = inputs[2].getShape();
    const auto &mask = inputs[3].getShape(), &out = output.getShape();
    if (q.size() != 3 || k.size() != 3 || v.size() != 3 || mask.size() != 3 || out.size() != 3)
        return false;
    for (const TensorNode &in : inputs)
        if (in.dtype != DType::FLOAT32 || !isContiguous(in))
            return false;
    return k == v && k[0] == q[0] && k[2] == q[2] && q[2] > 0 && mask[0] == 1 && mask[1] == q[1] &&
           mask[2] == k[1] && out[0] == 1 && out[1] == q[1] && out[2] == q[0] * q[2] &&
           output.dtype == DType::FLOAT32 && isContiguous(output);
}

// Rows [task_begin, task_end) of the (head, query row) task list. Keys whose
// mask entry is NEG_INF contribute exp(NEG_INF - max) == 0 and are skipped
// (never read). A row with no unmasked key follows the reference exactly.
inline void maskedAttentionTasks(const KernelContext &ctx, uint32_t task_begin, uint32_t task_end)
{
    const float *q_ptr = static_cast<const float *>(ctx.inputs[0]);
    const float *k_ptr = static_cast<const float *>(ctx.inputs[1]);
    const float *v_ptr = static_cast<const float *>(ctx.inputs[2]);
    const float *mask_ptr = static_cast<const float *>(ctx.inputs[3]);
    float *out_ptr = static_cast<float *>(ctx.outputs[0]);
    const auto &q_shape = ctx.inViews[0].getShape();
    const uint32_t q_rows = q_shape[1], head_dim = q_shape[2];
    const uint32_t heads = q_shape[0];
    const uint32_t keys = ctx.inViews[1].getShape()[1];
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> scores(keys);
    std::vector<uint32_t> visible(keys);
    for (uint32_t task = task_begin; task < task_end; ++task)
    {
        const uint32_t h = task / q_rows, s = task % q_rows;
        const float *q_row = q_ptr + (static_cast<uint64_t>(h) * q_rows + s) * head_dim;
        const float *mask_row = mask_ptr + static_cast<uint64_t>(s) * keys;
        uint32_t count = 0;
        for (uint32_t j = 0; j < keys; ++j)
            if (mask_row[j] != TGConstants::NEG_INF)
                visible[count++] = j;
        const bool dense = count == 0;
        if (dense)
            for (uint32_t j = 0; j < keys; ++j)
                visible[count++] = j;

        float max_score = -INFINITY;
        for (uint32_t i = 0; i < count; ++i)
        {
            const uint32_t j = visible[i];
            const float *k_row = k_ptr + (static_cast<uint64_t>(h) * keys + j) * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d)
                dot += (q_row[d] * scale) * k_row[d];
            scores[i] = dot + mask_row[j];
            max_score = std::max(max_score, scores[i]);
        }
        float total = 0.0f;
        for (uint32_t i = 0; i < count; ++i)
        {
            scores[i] = std::exp(scores[i] - max_score);
            total += scores[i];
        }
        float *out_row = out_ptr + static_cast<uint64_t>(s) * heads * head_dim + static_cast<uint64_t>(h) * head_dim;
        std::fill(out_row, out_row + head_dim, 0.0f);
        const float inv_total = 1.0f / total;
        for (uint32_t i = 0; i < count; ++i)
        {
            const float p = scores[i] * inv_total;
            const float *v_row = v_ptr + (static_cast<uint64_t>(h) * keys + visible[i]) * head_dim;
            for (uint32_t d = 0; d < head_dim; ++d)
                out_row[d] += p * v_row[d];
        }
    }
}

inline uint32_t maskedAttentionTaskCount(const KernelContext &ctx)
{
    const auto &q_shape = ctx.inViews[0].getShape();
    return q_shape[0] * q_shape[1];
}

// Query rows follow output rows; the mask rows go with them.
// (A function, not a variable: kernel registrars use it during static
// initialization, before an inline variable would be guaranteed initialized.)
inline FusedRowRestriction maskedAttentionRowRestriction()
{
    return FusedRowRestriction{1, {1, -1, -1, 1}, -1};
}
