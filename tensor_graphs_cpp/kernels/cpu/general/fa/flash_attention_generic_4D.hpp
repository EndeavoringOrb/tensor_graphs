#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "core/common/constants.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

inline bool matchFlashAttentionGeneric4D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &sQ = inputs[0].getShape();
    if (sQ.size() != 4)
        return false;

    const auto &sK = inputs[1].getShape();
    const auto &sV = inputs[2].getShape();
    const auto &sO = output.getShape();

    if (sK != sQ || sV != sQ || sO != sQ)
        return false;

    return isContiguous(output);
}

inline void runFlashAttentionGeneric4D(const KernelContext &ctx)
{
    const float *Q_ptr = static_cast<const float *>(ctx.inputs[0]);
    const float *K_ptr = static_cast<const float *>(ctx.inputs[1]);
    const float *V_ptr = static_cast<const float *>(ctx.inputs[2]);
    float *O_ptr = static_cast<float *>(ctx.outputs[0]);

    const auto &sQ = ctx.inViews[0].getShape();
    const uint32_t B = sQ[0];
    const uint32_t H = sQ[1];
    const uint32_t S = sQ[2];
    const uint32_t D = sQ[3];

    const uint64_t stride_H = static_cast<uint64_t>(S) * D;
    const uint64_t stride_B = static_cast<uint64_t>(H) * stride_H;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    uint32_t total_tasks = B * H * S;
    num_threads = std::min(num_threads, total_tasks);

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t tasks_per_thread = (total_tasks + num_threads - 1) / num_threads;
        uint32_t start_task = t * tasks_per_thread;
        uint32_t end_task = std::min(start_task + tasks_per_thread, total_tasks);

        std::vector<float> row_scores(S);

        for (uint32_t task = start_task; task < end_task; ++task)
        {
            uint32_t b = task / (H * S);
            uint32_t h = (task / S) % H;
            uint32_t s = task % S;

            const float *q_row = Q_ptr + b * stride_B + h * stride_H + s * D;
            const float *K_h = K_ptr + b * stride_B + h * stride_H;
            const float *V_h = V_ptr + b * stride_B + h * stride_H;
            float *o_row = O_ptr + b * stride_B + h * stride_H + s * D;

            float m = -1e30f;
            for (uint32_t j = 0; j < S; ++j)
            {
                const float *k_row = K_h + j * D;
                float score = 0.0f;
#if defined(TG_HAS_NEON)
                float32x4_t acc = vdupq_n_f32(0.0f);
                uint32_t d = 0;
                for (; d + 4 <= D; d += 4)
                {
                    acc = vfmaq_f32(acc, vld1q_f32(q_row + d), vld1q_f32(k_row + d));
                }
                score = vaddvq_f32(acc);
                for (; d < D; ++d)
                {
                    score += q_row[d] * k_row[d];
                }
#else
                for (uint32_t d = 0; d < D; ++d)
                {
                    score += q_row[d] * k_row[d];
                }
#endif
                row_scores[j] = score;
                if (score > m)
                    m = score;
            }

            float l = 0.0f;
            for (uint32_t j = 0; j < S; ++j)
            {
                float exp_val = std::exp(row_scores[j] - m);
                row_scores[j] = exp_val;
                l += exp_val;
            }

            float inv_l = 1.0f / l;
            for (uint32_t d = 0; d < D; ++d)
            {
                o_row[d] = 0.0f;
            }

            for (uint32_t j = 0; j < S; ++j)
            {
                float prob = row_scores[j] * inv_l;
                const float *v_row = V_h + j * D;
#if defined(TG_HAS_NEON)
                float32x4_t p_vec = vdupq_n_f32(prob);
                uint32_t d = 0;
                for (; d + 4 <= D; d += 4)
                {
                    float32x4_t v_v = vld1q_f32(v_row + d);
                    float32x4_t v_o = vld1q_f32(o_row + d);
                    vst1q_f32(o_row + d, vfmaq_f32(v_o, p_vec, v_v));
                }
                for (; d < D; ++d)
                {
                    o_row[d] += prob * v_row[d];
                }
#else
                for (uint32_t d = 0; d < D; ++d)
                {
                    o_row[d] += prob * v_row[d];
                }
#endif
            }
        }
    });
}

inline LogicalId refFactoryFlashAttentionGeneric4D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId Q = inputs[0], K = inputs[1], V = inputs[2];
    auto sQ = g.getNode(Q).getShape();
    uint32_t B = sQ[0];
    uint32_t num_heads = sQ[1];
    uint32_t S = sQ[2];

    int32_t perm[] = {0, 1, 3, 2};
    LogicalId K_t = g.contiguous(g.permute(K, g.constant({4}, perm, DType::INT32)));
    LogicalId scores = g.dot(Q, K_t);

    LogicalId max_s = g.repeat(g.max(scores, -1), S, 3);
    LogicalId shifted = g.add(scores, g.neg(max_s));
    LogicalId exps = g.pow(g.fill(TGConstants::E, {B, num_heads, S, S}), shifted);
    LogicalId sums = g.repeat(g.sum(exps, -1), S, 3);
    LogicalId probs = g.div(exps, sums);

    return g.dot(probs, V);
}

REGISTER_KERNEL("Flash_Attention_Generic_4D", 3, 3, matchFlashAttentionGeneric4D, runFlashAttentionGeneric4D,
                refFactoryFlashAttentionGeneric4D, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
                {{1, 48, 1152, 128}, {1, 48, 1152, 128}, {1, 48, 1152, 128}}, {true, true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});