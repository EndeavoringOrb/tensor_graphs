#pragma once
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "core/common/thread_pool.hpp"

static inline float32x4_t flash_vexpq_f32_v2(float32x4_t x)
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

inline bool matchFlashAttentionNeon(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &sQ = inputs[0].getShape();
    // rigidly bound to the [1, 12, 5040, 64] fast-path constraint
    if (sQ.size() != 4)
        return false;
    if (sQ[0] != 1 || sQ[1] != 12 || sQ[2] != 5040 || sQ[3] != 64)
        return false;

    const auto &sK = inputs[1].getShape();
    const auto &sV = inputs[2].getShape();
    const auto &sO = output.getShape();

    if (sK != sQ || sV != sQ || sO != sQ)
        return false;

    return true;
}

inline void runFlashAttentionNeon(const KernelContext &ctx)
{
    const float *Q_ptr = static_cast<const float *>(ctx.inputs[0]);
    const float *K_ptr = static_cast<const float *>(ctx.inputs[1]);
    const float *V_ptr = static_cast<const float *>(ctx.inputs[2]);
    float *O_ptr = static_cast<float *>(ctx.outputs[0]);

    const uint32_t H = 12;
    const uint32_t S = 5040;
    const uint32_t D = 64;
    const uint32_t stride_H = S * D; // 322560 elements per head

    // Fits EXACTLY in 96KB L1d (18KB Q + 18KB O + 30KB K + 30KB V = 96KB limit)
    constexpr uint32_t BR = 72;
    constexpr uint32_t BC = 120;
    constexpr uint32_t num_q_blocks = S / BR; // 70
    constexpr uint32_t num_k_blocks = S / BC; // 42

    const uint32_t total_blocks = H * num_q_blocks; // 840
    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    num_threads = std::min(num_threads, total_blocks);

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        const uint32_t blocks_per_thread = (total_blocks + num_threads - 1) / num_threads;
        uint32_t start_block = t * blocks_per_thread;
        uint32_t end_block = std::min(start_block + blocks_per_thread, total_blocks);
        if (start_block >= end_block)
            return;

        alignas(64) float O_block[BR * D];
        alignas(64) float m_vec[BR];
        alignas(64) float s_vec[BR];
        alignas(64) float S_row[BC];

        for (uint32_t blk = start_block; blk < end_block; ++blk)
        {
            uint32_t h = blk / num_q_blocks;
            uint32_t qb = blk % num_q_blocks;
            uint32_t q_start = qb * BR;

            // Init O, m, s context
            for (uint32_t i = 0; i < BR * D; ++i)
                O_block[i] = 0.0f;
            for (uint32_t i = 0; i < BR; ++i)
            {
                m_vec[i] = -1e30f;
                s_vec[i] = 0.0f;
            }

            const float *Q_h = Q_ptr + h * stride_H;
            const float *K_h = K_ptr + h * stride_H;
            const float *V_h = V_ptr + h * stride_H;

            for (uint32_t kb = 0; kb < num_k_blocks; ++kb)
            {
                uint32_t k_start = kb * BC;

                for (uint32_t i = 0; i < BR; ++i)
                {
                    const float *Q_row = Q_h + (q_start + i) * D;

                    // ---- PHASE 1: Q @ K^T (16 register Q-Cache mapped, latency
                    // decoupled) ----
                    float32x4_t q0 = vld1q_f32(Q_row + 0);
                    float32x4_t q1 = vld1q_f32(Q_row + 4);
                    float32x4_t q2 = vld1q_f32(Q_row + 8);
                    float32x4_t q3 = vld1q_f32(Q_row + 12);
                    float32x4_t q4 = vld1q_f32(Q_row + 16);
                    float32x4_t q5 = vld1q_f32(Q_row + 20);
                    float32x4_t q6 = vld1q_f32(Q_row + 24);
                    float32x4_t q7 = vld1q_f32(Q_row + 28);
                    float32x4_t q8 = vld1q_f32(Q_row + 32);
                    float32x4_t q9 = vld1q_f32(Q_row + 36);
                    float32x4_t q10 = vld1q_f32(Q_row + 40);
                    float32x4_t q11 = vld1q_f32(Q_row + 44);
                    float32x4_t q12 = vld1q_f32(Q_row + 48);
                    float32x4_t q13 = vld1q_f32(Q_row + 52);
                    float32x4_t q14 = vld1q_f32(Q_row + 56);
                    float32x4_t q15 = vld1q_f32(Q_row + 60);

                    for (uint32_t k = 0; k < BC; k += 2)
                    {
                        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
                        float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0), b2 = vdupq_n_f32(0), b3 = vdupq_n_f32(0);

                        const float *k0_ptr = K_h + (k_start + k) * D;
                        const float *k1_ptr = K_h + (k_start + k + 1) * D;

// Pipeline K0 fully, then K1 fully ensuring 4+ cycle separation for accumulator
// latency
#define FMA_K0(idx, acc) acc = vfmaq_f32(acc, q##idx, vld1q_f32(k0_ptr + idx * 4));
                        FMA_K0(0, a0);
                        FMA_K0(1, a1);
                        FMA_K0(2, a2);
                        FMA_K0(3, a3);
                        FMA_K0(4, a0);
                        FMA_K0(5, a1);
                        FMA_K0(6, a2);
                        FMA_K0(7, a3);
                        FMA_K0(8, a0);
                        FMA_K0(9, a1);
                        FMA_K0(10, a2);
                        FMA_K0(11, a3);
                        FMA_K0(12, a0);
                        FMA_K0(13, a1);
                        FMA_K0(14, a2);
                        FMA_K0(15, a3);

#define FMA_K1(idx, acc) acc = vfmaq_f32(acc, q##idx, vld1q_f32(k1_ptr + idx * 4));
                        FMA_K1(0, b0);
                        FMA_K1(1, b1);
                        FMA_K1(2, b2);
                        FMA_K1(3, b3);
                        FMA_K1(4, b0);
                        FMA_K1(5, b1);
                        FMA_K1(6, b2);
                        FMA_K1(7, b3);
                        FMA_K1(8, b0);
                        FMA_K1(9, b1);
                        FMA_K1(10, b2);
                        FMA_K1(11, b3);
                        FMA_K1(12, b0);
                        FMA_K1(13, b1);
                        FMA_K1(14, b2);
                        FMA_K1(15, b3);

                        a0 = vaddq_f32(a0, a1);
                        a2 = vaddq_f32(a2, a3);
                        a0 = vaddq_f32(a0, a2);
                        b0 = vaddq_f32(b0, b1);
                        b2 = vaddq_f32(b2, b3);
                        b0 = vaddq_f32(b0, b2);

                        S_row[k] = vaddvq_f32(a0);
                        S_row[k + 1] = vaddvq_f32(b0);
                    }

                    // ---- PHASE 2: Online Softmax (Max, Exp, P-gen & Normalization
                    // Scaling) ----
                    float32x4_t v_max = vdupq_n_f32(-1e30f);
                    for (uint32_t k = 0; k < BC; k += 4)
                    {
                        v_max = vmaxq_f32(v_max, vld1q_f32(S_row + k));
                    }
                    float row_max = vmaxvq_f32(v_max);
                    float m_old = m_vec[i];
                    float m_new = std::max(m_old, row_max);
                    float alpha = std::exp(m_old - m_new);

                    float32x4_t v_m_new = vdupq_n_f32(m_new);
                    float32x4_t v_sum = vdupq_n_f32(0);
                    for (uint32_t k = 0; k < BC; k += 4)
                    {
                        float32x4_t s_val = vld1q_f32(S_row + k);
                        float32x4_t p_val = flash_vexpq_f32_v2(vsubq_f32(s_val, v_m_new));
                        vst1q_f32(S_row + k, p_val);
                        v_sum = vaddq_f32(v_sum, p_val);
                    }
                    float row_sum = vaddvq_f32(v_sum);

                    m_vec[i] = m_new;
                    s_vec[i] = alpha * s_vec[i] + row_sum;

                    // ---- PHASE 3: P @ V (16 register O-Cache mapped, ILP spaced
                    // across iter) ----
                    float *O_row = O_block + i * D;
                    float32x4_t o0 = vld1q_f32(O_row + 0);
                    float32x4_t o1 = vld1q_f32(O_row + 4);
                    float32x4_t o2 = vld1q_f32(O_row + 8);
                    float32x4_t o3 = vld1q_f32(O_row + 12);
                    float32x4_t o4 = vld1q_f32(O_row + 16);
                    float32x4_t o5 = vld1q_f32(O_row + 20);
                    float32x4_t o6 = vld1q_f32(O_row + 24);
                    float32x4_t o7 = vld1q_f32(O_row + 28);
                    float32x4_t o8 = vld1q_f32(O_row + 32);
                    float32x4_t o9 = vld1q_f32(O_row + 36);
                    float32x4_t o10 = vld1q_f32(O_row + 40);
                    float32x4_t o11 = vld1q_f32(O_row + 44);
                    float32x4_t o12 = vld1q_f32(O_row + 48);
                    float32x4_t o13 = vld1q_f32(O_row + 52);
                    float32x4_t o14 = vld1q_f32(O_row + 56);
                    float32x4_t o15 = vld1q_f32(O_row + 60);

                    float32x4_t v_alpha = vdupq_n_f32(alpha);
                    o0 = vmulq_f32(o0, v_alpha);
                    o1 = vmulq_f32(o1, v_alpha);
                    o2 = vmulq_f32(o2, v_alpha);
                    o3 = vmulq_f32(o3, v_alpha);
                    o4 = vmulq_f32(o4, v_alpha);
                    o5 = vmulq_f32(o5, v_alpha);
                    o6 = vmulq_f32(o6, v_alpha);
                    o7 = vmulq_f32(o7, v_alpha);
                    o8 = vmulq_f32(o8, v_alpha);
                    o9 = vmulq_f32(o9, v_alpha);
                    o10 = vmulq_f32(o10, v_alpha);
                    o11 = vmulq_f32(o11, v_alpha);
                    o12 = vmulq_f32(o12, v_alpha);
                    o13 = vmulq_f32(o13, v_alpha);
                    o14 = vmulq_f32(o14, v_alpha);
                    o15 = vmulq_f32(o15, v_alpha);

                    for (uint32_t k = 0; k < BC; k += 2)
                    {
                        float32x4_t p0 = vdupq_n_f32(S_row[k]);
                        const float *v0_ptr = V_h + (k_start + k) * D;
#define FMA_P_V_P0(idx) o##idx = vfmaq_f32(o##idx, p0, vld1q_f32(v0_ptr + idx * 4));
                        FMA_P_V_P0(0);
                        FMA_P_V_P0(1);
                        FMA_P_V_P0(2);
                        FMA_P_V_P0(3);
                        FMA_P_V_P0(4);
                        FMA_P_V_P0(5);
                        FMA_P_V_P0(6);
                        FMA_P_V_P0(7);
                        FMA_P_V_P0(8);
                        FMA_P_V_P0(9);
                        FMA_P_V_P0(10);
                        FMA_P_V_P0(11);
                        FMA_P_V_P0(12);
                        FMA_P_V_P0(13);
                        FMA_P_V_P0(14);
                        FMA_P_V_P0(15);

                        float32x4_t p1 = vdupq_n_f32(S_row[k + 1]);
                        const float *v1_ptr = V_h + (k_start + k + 1) * D;
#define FMA_P_V_P1(idx) o##idx = vfmaq_f32(o##idx, p1, vld1q_f32(v1_ptr + idx * 4));
                        FMA_P_V_P1(0);
                        FMA_P_V_P1(1);
                        FMA_P_V_P1(2);
                        FMA_P_V_P1(3);
                        FMA_P_V_P1(4);
                        FMA_P_V_P1(5);
                        FMA_P_V_P1(6);
                        FMA_P_V_P1(7);
                        FMA_P_V_P1(8);
                        FMA_P_V_P1(9);
                        FMA_P_V_P1(10);
                        FMA_P_V_P1(11);
                        FMA_P_V_P1(12);
                        FMA_P_V_P1(13);
                        FMA_P_V_P1(14);
                        FMA_P_V_P1(15);
                    }

                    vst1q_f32(O_row + 0, o0);
                    vst1q_f32(O_row + 4, o1);
                    vst1q_f32(O_row + 8, o2);
                    vst1q_f32(O_row + 12, o3);
                    vst1q_f32(O_row + 16, o4);
                    vst1q_f32(O_row + 20, o5);
                    vst1q_f32(O_row + 24, o6);
                    vst1q_f32(O_row + 28, o7);
                    vst1q_f32(O_row + 32, o8);
                    vst1q_f32(O_row + 36, o9);
                    vst1q_f32(O_row + 40, o10);
                    vst1q_f32(O_row + 44, o11);
                    vst1q_f32(O_row + 48, o12);
                    vst1q_f32(O_row + 52, o13);
                    vst1q_f32(O_row + 56, o14);
                    vst1q_f32(O_row + 60, o15);
                }
            } // End K block loop

            // Final scalar normalization
            float *O_h = O_ptr + h * stride_H;
            for (uint32_t i = 0; i < BR; ++i)
            {
                float inv_s = 1.0f / s_vec[i];
                float32x4_t v_inv = vdupq_n_f32(inv_s);
                float *O_row = O_block + i * D;
                float *O_dst = O_h + (q_start + i) * D;

                for (uint32_t d = 0; d < D; d += 4)
                {
                    vst1q_f32(O_dst + d, vmulq_f32(vld1q_f32(O_row + d), v_inv));
                }
            }
        } // End QB loop
    });
}

inline LogicalId refFactoryFlashAttention4D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId Q = inputs[0], K = inputs[1], V = inputs[2];
    int32_t perm[] = {0, 1, 3, 2};
    LogicalId K_t = g.contiguous(g.permute(K, g.constant({4}, perm, DType::INT32)));
    LogicalId scores = g.dot(Q, K_t);

    int32_t axis = -1;
    LogicalId axis_node = g.constant({1}, &axis, DType::INT32);
    auto Q_shape = g.getNode(Q).getShape();
    auto K_shape = g.getNode(K).getShape();
    std::vector<uint32_t> s_shape = {Q_shape[0], Q_shape[1], Q_shape[2], K_shape[2]};
    int32_t S_val = (int32_t)s_shape[s_shape.size() - 1];
    LogicalId m_rep = g.constant({1}, &S_val, DType::INT32);
    LogicalId ax_rep = g.constant({1}, &axis, DType::INT32);

    LogicalId max_s = g.max(scores, axis_node);
    LogicalId max_expanded = g.repeat(max_s, m_rep, ax_rep);
    LogicalId shifted = g.add(scores, g.neg(max_expanded));

    float e_v = 2.718281828459045f;
    LogicalId e_n = g.constant({1}, &e_v, DType::FLOAT32);
    int32_t sh4[] = {1, 1, 1, 1};
    LogicalId e_b = g.reshape(e_n, g.constant({4}, sh4, DType::INT32));

    for (int i = 0; i < 4; ++i)
    {
        int32_t r = (int32_t)s_shape[i];
        if (r <= 1)
            continue;
        int32_t a = i;
        e_b = g.repeat(e_b, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }

    LogicalId exps = g.pow(e_b, shifted);
    LogicalId sums = g.repeat(g.sum(exps, axis_node), m_rep, ax_rep);
    LogicalId probs = g.div(exps, sums);

    return g.dot(probs, V);
}

REGISTER_KERNEL("Flash_Attention_Neon_Fused_4D", 3, 3, matchFlashAttentionNeon, runFlashAttentionNeon,
                refFactoryFlashAttention4D, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
                {{1, 12, 5040, 64}, {1, 12, 5040, 64}, {1, 12, 5040, 64}}, {true, true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON