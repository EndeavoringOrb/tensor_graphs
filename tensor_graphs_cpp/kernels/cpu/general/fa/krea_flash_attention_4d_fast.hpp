#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/constants.hpp"
#include "core/common/thread_pool.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

static inline float32x4_t krea_fa_fast_exp(float32x4_t x)
{
    x = vmaxq_f32(x, vdupq_n_f32(-80.0f));
    x = vminq_f32(x, vdupq_n_f32(80.0f));

    const float32x4_t v_log2e = vdupq_n_f32(1.4426950408889634f);
    float32x4_t y = vmulq_f32(x, v_log2e);
    float32x4_t n = vrndnq_f32(y);
    float32x4_t f = vsubq_f32(y, n);

    const float32x4_t c0 = vdupq_n_f32(1.0f);
    const float32x4_t c1 = vdupq_n_f32(0.6931471805599453f);
    const float32x4_t c2 = vdupq_n_f32(0.2402265069591007f);
    const float32x4_t c3 = vdupq_n_f32(0.0555041086648216f);
    const float32x4_t c4 = vdupq_n_f32(0.0096181291076285f);

    float32x4_t f2 = vmulq_f32(f, f);
    float32x4_t f3 = vmulq_f32(f2, f);
    float32x4_t f4 = vmulq_f32(f3, f);

    float32x4_t poly = vaddq_f32(
        c0, vaddq_f32(vmulq_f32(c1, f), vaddq_f32(vmulq_f32(c2, f2), vaddq_f32(vmulq_f32(c3, f3), vmulq_f32(c4, f4)))));

    int32x4_t v_n = vcvtq_s32_f32(n);
    int32x4_t exp_bits = vshlq_n_s32(vaddq_s32(v_n, vdupq_n_s32(127)), 23);
    float32x4_t scale = vreinterpretq_f32_s32(exp_bits);

    return vmulq_f32(poly, scale);
}
#endif

inline bool matchKreaFlashAttention4DFast(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &sQ = inputs[0].getShape();
    const auto &sK = inputs[1].getShape();
    const auto &sV = inputs[2].getShape();
    const auto &sO = output.getShape();

    if (sQ.size() != 4 || sK.size() != 4 || sV.size() != 4 || sO.size() != 4)
        return false;

    if (sK != sQ || sV != sQ || sO != sQ)
        return false;

    return isContiguous(output);
}

inline void runKreaFlashAttention4DFast(const KernelContext &ctx)
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

    // Cache-tiling tuned for ARM CPU caches (BR=32, BC=64)
    constexpr uint32_t BR = 32;
    constexpr uint32_t BC = 64;

    const uint32_t num_q_blocks = (S + BR - 1) / BR;
    const uint32_t num_k_blocks = (S + BC - 1) / BC;
    const uint32_t total_tasks = B * H * num_q_blocks;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    num_threads = std::min(num_threads, total_tasks);

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t tasks_per_thread = (total_tasks + num_threads - 1) / num_threads;
        uint32_t start_task = t * tasks_per_thread;
        uint32_t end_task = std::min(start_task + tasks_per_thread, total_tasks);
        if (start_task >= end_task)
            return;

        std::vector<float> O_block(BR * D);
        std::vector<float> m_vec(BR);
        std::vector<float> l_vec(BR);
        std::vector<float> S_tile(BR * BC);
        std::vector<float> P_tile(BR * BC);

        for (uint32_t task = start_task; task < end_task; ++task)
        {
            uint32_t b = task / (H * num_q_blocks);
            uint32_t rem = task % (H * num_q_blocks);
            uint32_t h = rem / num_q_blocks;
            uint32_t qb = rem % num_q_blocks;

            uint32_t q_start = qb * BR;
            uint32_t q_end = std::min(q_start + BR, S);
            uint32_t cur_BR = q_end - q_start;

            std::fill(O_block.begin(), O_block.end(), 0.0f);
            std::fill(m_vec.begin(), m_vec.end(), -1e30f);
            std::fill(l_vec.begin(), l_vec.end(), 0.0f);

            const float *Q_head = Q_ptr + b * stride_B + h * stride_H;
            const float *K_head = K_ptr + b * stride_B + h * stride_H;
            const float *V_head = V_ptr + b * stride_B + h * stride_H;
            float *O_head = O_ptr + b * stride_B + h * stride_H;

            for (uint32_t kb = 0; kb < num_k_blocks; ++kb)
            {
                uint32_t k_start = kb * BC;
                uint32_t k_end = std::min(k_start + BC, S);
                uint32_t cur_BC = k_end - k_start;

                // Step 1: Compute S_tile = Q_blk @ K_blk^T
                for (uint32_t i = 0; i < cur_BR; ++i)
                {
                    const float *q_row = Q_head + (q_start + i) * D;
                    float *s_row = S_tile.data() + i * BC;

                    for (uint32_t j = 0; j < cur_BC; ++j)
                    {
                        const float *k_row = K_head + (k_start + j) * D;
#if defined(TG_HAS_NEON)
                        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
                        float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
                        uint32_t d = 0;
                        for (; d + 16 <= D; d += 16)
                        {
                            a0 = vfmaq_f32(a0, vld1q_f32(q_row + d + 0), vld1q_f32(k_row + d + 0));
                            a1 = vfmaq_f32(a1, vld1q_f32(q_row + d + 4), vld1q_f32(k_row + d + 4));
                            a2 = vfmaq_f32(a2, vld1q_f32(q_row + d + 8), vld1q_f32(k_row + d + 8));
                            a3 = vfmaq_f32(a3, vld1q_f32(q_row + d + 12), vld1q_f32(k_row + d + 12));
                        }
                        float score = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
                        for (; d < D; ++d)
                            score += q_row[d] * k_row[d];
                        s_row[j] = score;
#else
                        float score = 0.0f;
                        for (uint32_t d = 0; d < D; ++d)
                            score += q_row[d] * k_row[d];
                        s_row[j] = score;
#endif
                    }
                }

                // Step 2 & 3: Online Softmax and Output Accumulation
                for (uint32_t i = 0; i < cur_BR; ++i)
                {
                    const float *s_row = S_tile.data() + i * BC;
                    float *p_row = P_tile.data() + i * BC;
                    float *o_row = O_block.data() + i * D;

                    float m_tile = -1e30f;
                    for (uint32_t j = 0; j < cur_BC; ++j)
                        m_tile = std::max(m_tile, s_row[j]);

                    float m_new = std::max(m_vec[i], m_tile);
                    float alpha = std::exp(m_vec[i] - m_new);

                    float l_tile = 0.0f;
                    uint32_t j = 0;
#if defined(TG_HAS_NEON)
                    float32x4_t v_m_new = vdupq_n_f32(m_new);
                    float32x4_t v_l_sum = vdupq_n_f32(0.0f);
                    for (; j + 4 <= cur_BC; j += 4)
                    {
                        float32x4_t s_vec = vld1q_f32(s_row + j);
                        float32x4_t p_vec = krea_fa_fast_exp(vsubq_f32(s_vec, v_m_new));
                        vst1q_f32(p_row + j, p_vec);
                        v_l_sum = vaddq_f32(v_l_sum, p_vec);
                    }
                    l_tile = vaddvq_f32(v_l_sum);
#endif
                    for (; j < cur_BC; ++j)
                    {
                        float p = std::exp(s_row[j] - m_new);
                        p_row[j] = p;
                        l_tile += p;
                    }

                    float l_new = alpha * l_vec[i] + l_tile;

#if defined(TG_HAS_NEON)
                    float32x4_t v_alpha = vdupq_n_f32(alpha);
                    uint32_t d = 0;
                    for (; d + 16 <= D; d += 16)
                    {
                        vst1q_f32(o_row + d + 0, vmulq_f32(vld1q_f32(o_row + d + 0), v_alpha));
                        vst1q_f32(o_row + d + 4, vmulq_f32(vld1q_f32(o_row + d + 4), v_alpha));
                        vst1q_f32(o_row + d + 8, vmulq_f32(vld1q_f32(o_row + d + 8), v_alpha));
                        vst1q_f32(o_row + d + 12, vmulq_f32(vld1q_f32(o_row + d + 12), v_alpha));
                    }
                    for (; d < D; ++d)
                        o_row[d] *= alpha;
#else
                    for (uint32_t d = 0; d < D; ++d)
                        o_row[d] *= alpha;
#endif

                    // P_blk @ V_blk accumulation
                    for (uint32_t k_idx = 0; k_idx < cur_BC; ++k_idx)
                    {
                        float p_val = p_row[k_idx];
                        const float *v_row = V_head + (k_start + k_idx) * D;
#if defined(TG_HAS_NEON)
                        float32x4_t v_p = vdupq_n_f32(p_val);
                        d = 0;
                        for (; d + 16 <= D; d += 16)
                        {
                            vst1q_f32(o_row + d + 0, vfmaq_f32(vld1q_f32(o_row + d + 0), v_p, vld1q_f32(v_row + d + 0)));
                            vst1q_f32(o_row + d + 4, vfmaq_f32(vld1q_f32(o_row + d + 4), v_p, vld1q_f32(v_row + d + 4)));
                            vst1q_f32(o_row + d + 8, vfmaq_f32(vld1q_f32(o_row + d + 8), v_p, vld1q_f32(v_row + d + 8)));
                            vst1q_f32(o_row + d + 12, vfmaq_f32(vld1q_f32(o_row + d + 12), v_p, vld1q_f32(v_row + d + 12)));
                        }
                        for (; d < D; ++d)
                            o_row[d] += p_val * v_row[d];
#else
                        for (uint32_t d = 0; d < D; ++d)
                            o_row[d] += p_val * v_row[d];
#endif
                    }

                    m_vec[i] = m_new;
                    l_vec[i] = l_new;
                }
            }

            // Normalization
            for (uint32_t i = 0; i < cur_BR; ++i)
            {
                float inv_l = 1.0f / l_vec[i];
                float *o_row = O_block.data() + i * D;
                float *dst_row = O_head + (q_start + i) * D;

#if defined(TG_HAS_NEON)
                float32x4_t v_inv = vdupq_n_f32(inv_l);
                uint32_t d = 0;
                for (; d + 16 <= D; d += 16)
                {
                    vst1q_f32(dst_row + d + 0, vmulq_f32(vld1q_f32(o_row + d + 0), v_inv));
                    vst1q_f32(dst_row + d + 4, vmulq_f32(vld1q_f32(o_row + d + 4), v_inv));
                    vst1q_f32(dst_row + d + 8, vmulq_f32(vld1q_f32(o_row + d + 8), v_inv));
                    vst1q_f32(dst_row + d + 12, vmulq_f32(vld1q_f32(o_row + d + 12), v_inv));
                }
                for (; d < D; ++d)
                    dst_row[d] = o_row[d] * inv_l;
#else
                for (uint32_t d = 0; d < D; ++d)
                    dst_row[d] = o_row[d] * inv_l;
#endif
            }
        }
    });
}

inline LogicalId refFactoryKreaFlashAttention4DFast(const std::vector<LogicalId> &inputs, Graph &g)
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

REGISTER_KERNEL("Krea_Flash_Attention_4D_Fast", 3, 3, matchKreaFlashAttention4DFast, runKreaFlashAttention4DFast,
                refFactoryKreaFlashAttention4DFast, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
                {{1, 48, 4224, 128}, {1, 48, 4224, 128}, {1, 48, 4224, 128}}, {true, true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});