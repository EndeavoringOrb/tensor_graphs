#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "core/common/constants.hpp"
#include "core/fused_row_restriction.hpp"
#include "core/kernels.hpp"
#include "kernels/fused/masked_attention/ref.hpp"

// Attention under the causal mask built in the graph:
//   mask[i, j] = NEG_INF where j >= i + diag_k, 0 elsewhere
// (triu(ones, diag_k) * NEG_INF, the construction models use for their causal
// mask), with diag_k an INT32 [1] input. Inputs q [H, Sq, D], k and v
// [H, Sk, D], diag_k; output [1, Sq, H * D]. Restricting output rows to
// [a, b) slices q and adds a to diag_k, so local row r still sees keys
// j < (a + r) + diag_k. The kernel computes the visible key range from diag_k
// and never reads (or stores) the mask.
inline LogicalId refFactoryCausalAttention(const std::vector<LogicalId> &inputs, Graph &g)
{
    const LogicalId q = inputs[0], k = inputs[1], v = inputs[2], diag_k = inputs[3];
    const uint32_t q_rows = g.getNode(q).getShape()[1];
    const uint32_t keys = g.getNode(k).getShape()[1];

    int32_t mask_shape[] = {static_cast<int32_t>(q_rows), static_cast<int32_t>(keys)};
    float one_val = 1.0f;
    LogicalId ones = g.fill(g.constant({1}, &one_val, DType::FLOAT32), g.constant({2}, mask_shape, DType::INT32));
    // The e-graph holds the triu input as CONTIGUOUS(FILL): the planner makes
    // TRIU's zero-stride input contiguous, and the two layouts are different
    // classes. The pattern names that form so it matches.
    LogicalId blocked = g.triu(g.contiguous(ones), diag_k);
    float neg_inf_val = TGConstants::NEG_INF;
    int32_t neg_inf_shape[] = {1, 1};
    LogicalId neg_inf = g.reshape(g.constant({1}, &neg_inf_val, DType::FLOAT32),
                                  g.constant({2}, neg_inf_shape, DType::INT32));
    neg_inf = g.repeat(g.repeat(neg_inf, q_rows, 0), keys, 1);
    int32_t final_shape[] = {1, static_cast<int32_t>(q_rows), static_cast<int32_t>(keys)};
    LogicalId mask = g.reshape(g.mul(blocked, neg_inf), g.constant({3}, final_shape, DType::INT32));
    return refFactoryMaskedAttention({q, k, v, mask}, g);
}

inline bool matchCausalAttention(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 4)
        return false;
    const auto &q = inputs[0].getShape(), &k = inputs[1].getShape(), &v = inputs[2].getShape();
    const auto &diag = inputs[3].getShape(), &out = output.getShape();
    if (q.size() != 3 || k.size() != 3 || v.size() != 3 || out.size() != 3)
        return false;
    for (size_t i = 0; i < 3; ++i)
        if (inputs[i].dtype != DType::FLOAT32 || !isContiguous(inputs[i]))
            return false;
    return inputs[3].dtype == DType::INT32 && diag.size() == 1 && diag[0] == 1 && k == v && k[0] == q[0] &&
           k[2] == q[2] && q[2] > 0 && out[0] == 1 && out[1] == q[1] && out[2] == q[0] * q[2] &&
           output.dtype == DType::FLOAT32 && isContiguous(output);
}

// Rows [task_begin, task_end) of the (head, query row) task list. Query row s
// sees keys [0, min(Sk, s + diag_k)); a row that sees none matches the
// reference, where every key carries the same NEG_INF bias.
inline void causalAttentionTasks(const KernelContext &ctx, uint32_t task_begin, uint32_t task_end)
{
    const float *q_ptr = static_cast<const float *>(ctx.inputs[0]);
    const float *k_ptr = static_cast<const float *>(ctx.inputs[1]);
    const float *v_ptr = static_cast<const float *>(ctx.inputs[2]);
    int32_t diag_k;
    std::memcpy(&diag_k, ctx.inputs[3], sizeof(diag_k));
    float *out_ptr = static_cast<float *>(ctx.outputs[0]);
    const auto &q_shape = ctx.inViews[0].getShape();
    const uint32_t heads = q_shape[0], q_rows = q_shape[1], head_dim = q_shape[2];
    const uint32_t keys = ctx.inViews[1].getShape()[1];
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> scores(keys);
    for (uint32_t task = task_begin; task < task_end; ++task)
    {
        const uint32_t h = task / q_rows, s = task % q_rows;
        const float *q_row = q_ptr + (static_cast<uint64_t>(h) * q_rows + s) * head_dim;
        const int64_t visible_end = std::clamp<int64_t>(static_cast<int64_t>(s) + diag_k, 0, keys);
        const bool dense = visible_end == 0;
        const uint32_t count = dense ? keys : static_cast<uint32_t>(visible_end);
        const float bias = dense ? TGConstants::NEG_INF : 0.0f;

        float max_score = -INFINITY;
        for (uint32_t j = 0; j < count; ++j)
        {
            const float *k_row = k_ptr + (static_cast<uint64_t>(h) * keys + j) * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d)
                dot += (q_row[d] * scale) * k_row[d];
            scores[j] = dot + bias;
            max_score = std::max(max_score, scores[j]);
        }
        float total = 0.0f;
        for (uint32_t j = 0; j < count; ++j)
        {
            scores[j] = std::exp(scores[j] - max_score);
            total += scores[j];
        }
        float *out_row = out_ptr + static_cast<uint64_t>(s) * heads * head_dim + static_cast<uint64_t>(h) * head_dim;
        std::fill(out_row, out_row + head_dim, 0.0f);
        const float inv_total = 1.0f / total;
        for (uint32_t j = 0; j < count; ++j)
        {
            const float p = scores[j] * inv_total;
            const float *v_row = v_ptr + (static_cast<uint64_t>(h) * keys + j) * head_dim;
            for (uint32_t d = 0; d < head_dim; ++d)
                out_row[d] += p * v_row[d];
        }
    }
}

inline uint32_t causalAttentionTaskCount(const KernelContext &ctx)
{
    const auto &q_shape = ctx.inViews[0].getShape();
    return q_shape[0] * q_shape[1];
}

// Query rows follow output rows; diag_k absorbs the row offset.
inline FusedRowRestriction causalAttentionRowRestriction()
{
    return FusedRowRestriction{1, {1, -1, -1, -1}, 3};
}
