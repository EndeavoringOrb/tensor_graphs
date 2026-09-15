#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline float fp8_e4m3_to_fp32_lut_val(uint8_t input)
{
    if (input == 0x7F || input == 0xFF)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float sign = (input & 0x80) ? -1.0f : 1.0f;
    uint32_t exp = (input & 0x78) >> 3;
    uint32_t mant = input & 0x07;
    if (exp == 0)
    {
        if (mant == 0)
        {
            return sign * 0.0f;
        }
        else
        {
            return sign * std::ldexp(static_cast<float>(mant), -9);
        }
    }
    else
    {
        return sign * std::ldexp(1.0f + static_cast<float>(mant) * 0.125f, static_cast<int>(exp) - 7);
    }
}

// 1 KB static LUT aligned to 64-byte cache lines, permanently pinned in L1d cache (96 KB)
inline const float *get_fp8_e4m3_gemm_lut()
{
    static const auto lut = []() {
        alignas(64) std::array<float, 256> table{};
        for (int i = 0; i < 256; ++i)
        {
            table[i] = fp8_e4m3_to_fp32_lut_val(static_cast<uint8_t>(i));
        }
        return table;
    }();
    return lut.data();
}

inline bool matchF8E4M3TransposedGEMM_v2(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &sX = inputs[0].getShape(); // [B, S, K]
    const auto &sW = inputs[1].getShape(); // [N, K]
    const auto &sO = output.getShape();    // [B, S, N]

    if (sX.size() != 3 || sW.size() != 2 || sO.size() != 3)
        return false;
    if (sX[2] != sW[1] || sO[2] != sW[0])
        return false;
    if (sO[0] != sX[0] || sO[1] != sX[1])
        return false;

    return isContiguous(output);
}

inline void runF8E4M3TransposedGEMM_v2(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const uint8_t *W = static_cast<const uint8_t *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t B = ctx.inViews[0].getShape()[0];
    const uint32_t S = ctx.inViews[0].getShape()[1];
    const uint32_t K = ctx.inViews[0].getShape()[2];
    const uint32_t N = ctx.inViews[1].getShape()[0];

    const uint32_t M = B * S;
    const float *lut = get_fp8_e4m3_gemm_lut();

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    num_threads = std::min(num_threads, std::max(1u, (N + 3u) / 4u));

    uint32_t n_per_thread = ((N + num_threads - 1) / num_threads + 3u) & ~3u;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t n_start = t * n_per_thread;
        uint32_t n_end = std::min(n_start + n_per_thread, N);
        if (n_start >= n_end)
            return;

        const uint32_t m_rem = M & ~3u;
        const uint32_t n_rem = n_end & ~3u;
        const uint32_t k_rem16 = K & ~15u;
        const uint32_t k_rem4 = K & ~3u;

        for (uint32_t m = 0; m < m_rem; m += 4)
        {
            const float *x0_ptr = X + static_cast<uint64_t>(m + 0) * K;
            const float *x1_ptr = X + static_cast<uint64_t>(m + 1) * K;
            const float *x2_ptr = X + static_cast<uint64_t>(m + 2) * K;
            const float *x3_ptr = X + static_cast<uint64_t>(m + 3) * K;

            for (uint32_t n = n_start; n < n_rem; n += 4)
            {
                const uint8_t *w0_ptr = W + static_cast<uint64_t>(n + 0) * K;
                const uint8_t *w1_ptr = W + static_cast<uint64_t>(n + 1) * K;
                const uint8_t *w2_ptr = W + static_cast<uint64_t>(n + 2) * K;
                const uint8_t *w3_ptr = W + static_cast<uint64_t>(n + 3) * K;

                // 16 accumulators in registers (4x4 tile)
                float32x4_t acc00 = vdupq_n_f32(0.0f), acc01 = vdupq_n_f32(0.0f), acc02 = vdupq_n_f32(0.0f),
                            acc03 = vdupq_n_f32(0.0f);
                float32x4_t acc10 = vdupq_n_f32(0.0f), acc11 = vdupq_n_f32(0.0f), acc12 = vdupq_n_f32(0.0f),
                            acc13 = vdupq_n_f32(0.0f);
                float32x4_t acc20 = vdupq_n_f32(0.0f), acc21 = vdupq_n_f32(0.0f), acc22 = vdupq_n_f32(0.0f),
                            acc23 = vdupq_n_f32(0.0f);
                float32x4_t acc30 = vdupq_n_f32(0.0f), acc31 = vdupq_n_f32(0.0f), acc32 = vdupq_n_f32(0.0f),
                            acc33 = vdupq_n_f32(0.0f);

                uint32_t k = 0;
                // Main loop: unrolled by 16 for ILP on 4-wide execution pipelines
                for (; k < k_rem16; k += 16)
                {
                    __builtin_prefetch(w0_ptr + k + 64);
                    __builtin_prefetch(w1_ptr + k + 64);
                    __builtin_prefetch(w2_ptr + k + 64);
                    __builtin_prefetch(w3_ptr + k + 64);
                    __builtin_prefetch(x0_ptr + k + 64);

                    // Sub-step 0
                    float32x4_t xv0 = vld1q_f32(x0_ptr + k + 0);
                    float32x4_t xv1 = vld1q_f32(x1_ptr + k + 0);
                    float32x4_t xv2 = vld1q_f32(x2_ptr + k + 0);
                    float32x4_t xv3 = vld1q_f32(x3_ptr + k + 0);

                    float32x4_t wv0 = {lut[w0_ptr[k + 0]], lut[w0_ptr[k + 1]], lut[w0_ptr[k + 2]], lut[w0_ptr[k + 3]]};
                    float32x4_t wv1 = {lut[w1_ptr[k + 0]], lut[w1_ptr[k + 1]], lut[w1_ptr[k + 2]], lut[w1_ptr[k + 3]]};
                    float32x4_t wv2 = {lut[w2_ptr[k + 0]], lut[w2_ptr[k + 1]], lut[w2_ptr[k + 2]], lut[w2_ptr[k + 3]]};
                    float32x4_t wv3 = {lut[w3_ptr[k + 0]], lut[w3_ptr[k + 1]], lut[w3_ptr[k + 2]], lut[w3_ptr[k + 3]]};

                    acc00 = vfmaq_f32(acc00, xv0, wv0);
                    acc01 = vfmaq_f32(acc01, xv0, wv1);
                    acc02 = vfmaq_f32(acc02, xv0, wv2);
                    acc03 = vfmaq_f32(acc03, xv0, wv3);

                    acc10 = vfmaq_f32(acc10, xv1, wv0);
                    acc11 = vfmaq_f32(acc11, xv1, wv1);
                    acc12 = vfmaq_f32(acc12, xv1, wv2);
                    acc13 = vfmaq_f32(acc13, xv1, wv3);

                    acc20 = vfmaq_f32(acc20, xv2, wv0);
                    acc21 = vfmaq_f32(acc21, xv2, wv1);
                    acc22 = vfmaq_f32(acc22, xv2, wv2);
                    acc23 = vfmaq_f32(acc23, xv2, wv3);

                    acc30 = vfmaq_f32(acc30, xv3, wv0);
                    acc31 = vfmaq_f32(acc31, xv3, wv1);
                    acc32 = vfmaq_f32(acc32, xv3, wv2);
                    acc33 = vfmaq_f32(acc33, xv3, wv3);

                    // Sub-step 1
                    xv0 = vld1q_f32(x0_ptr + k + 4);
                    xv1 = vld1q_f32(x1_ptr + k + 4);
                    xv2 = vld1q_f32(x2_ptr + k + 4);
                    xv3 = vld1q_f32(x3_ptr + k + 4);

                    wv0 = {lut[w0_ptr[k + 4]], lut[w0_ptr[k + 5]], lut[w0_ptr[k + 6]], lut[w0_ptr[k + 7]]};
                    wv1 = {lut[w1_ptr[k + 4]], lut[w1_ptr[k + 5]], lut[w1_ptr[k + 6]], lut[w1_ptr[k + 7]]};
                    wv2 = {lut[w2_ptr[k + 4]], lut[w2_ptr[k + 5]], lut[w2_ptr[k + 6]], lut[w2_ptr[k + 7]]};
                    wv3 = {lut[w3_ptr[k + 4]], lut[w3_ptr[k + 5]], lut[w3_ptr[k + 6]], lut[w3_ptr[k + 7]]};

                    acc00 = vfmaq_f32(acc00, xv0, wv0);
                    acc01 = vfmaq_f32(acc01, xv0, wv1);
                    acc02 = vfmaq_f32(acc02, xv0, wv2);
                    acc03 = vfmaq_f32(acc03, xv0, wv3);

                    acc10 = vfmaq_f32(acc10, xv1, wv0);
                    acc11 = vfmaq_f32(acc11, xv1, wv1);
                    acc12 = vfmaq_f32(acc12, xv1, wv2);
                    acc13 = vfmaq_f32(acc13, xv1, wv3);

                    acc20 = vfmaq_f32(acc20, xv2, wv0);
                    acc21 = vfmaq_f32(acc21, xv2, wv1);
                    acc22 = vfmaq_f32(acc22, xv2, wv2);
                    acc23 = vfmaq_f32(acc23, xv2, wv3);

                    acc30 = vfmaq_f32(acc30, xv3, wv0);
                    acc31 = vfmaq_f32(acc31, xv3, wv1);
                    acc32 = vfmaq_f32(acc32, xv3, wv2);
                    acc33 = vfmaq_f32(acc33, xv3, wv3);

                    // Sub-step 2
                    xv0 = vld1q_f32(x0_ptr + k + 8);
                    xv1 = vld1q_f32(x1_ptr + k + 8);
                    xv2 = vld1q_f32(x2_ptr + k + 8);
                    xv3 = vld1q_f32(x3_ptr + k + 8);

                    wv0 = {lut[w0_ptr[k + 8]], lut[w0_ptr[k + 9]], lut[w0_ptr[k + 10]], lut[w0_ptr[k + 11]]};
                    wv1 = {lut[w1_ptr[k + 8]], lut[w1_ptr[k + 9]], lut[w1_ptr[k + 10]], lut[w1_ptr[k + 11]]};
                    wv2 = {lut[w2_ptr[k + 8]], lut[w2_ptr[k + 9]], lut[w2_ptr[k + 10]], lut[w2_ptr[k + 11]]};
                    wv3 = {lut[w3_ptr[k + 8]], lut[w3_ptr[k + 9]], lut[w3_ptr[k + 10]], lut[w3_ptr[k + 11]]};

                    acc00 = vfmaq_f32(acc00, xv0, wv0);
                    acc01 = vfmaq_f32(acc01, xv0, wv1);
                    acc02 = vfmaq_f32(acc02, xv0, wv2);
                    acc03 = vfmaq_f32(acc03, xv0, wv3);

                    acc10 = vfmaq_f32(acc10, xv1, wv0);
                    acc11 = vfmaq_f32(acc11, xv1, wv1);
                    acc12 = vfmaq_f32(acc12, xv1, wv2);
                    acc13 = vfmaq_f32(acc13, xv1, wv3);

                    acc20 = vfmaq_f32(acc20, xv2, wv0);
                    acc21 = vfmaq_f32(acc21, xv2, wv1);
                    acc22 = vfmaq_f32(acc22, xv2, wv2);
                    acc23 = vfmaq_f32(acc23, xv2, wv3);

                    acc30 = vfmaq_f32(acc30, xv3, wv0);
                    acc31 = vfmaq_f32(acc31, xv3, wv1);
                    acc32 = vfmaq_f32(acc32, xv3, wv2);
                    acc33 = vfmaq_f32(acc33, xv3, wv3);

                    // Sub-step 3
                    xv0 = vld1q_f32(x0_ptr + k + 12);
                    xv1 = vld1q_f32(x1_ptr + k + 12);
                    xv2 = vld1q_f32(x2_ptr + k + 12);
                    xv3 = vld1q_f32(x3_ptr + k + 12);

                    wv0 = {lut[w0_ptr[k + 12]], lut[w0_ptr[k + 13]], lut[w0_ptr[k + 14]], lut[w0_ptr[k + 15]]};
                    wv1 = {lut[w1_ptr[k + 12]], lut[w1_ptr[k + 13]], lut[w1_ptr[k + 14]], lut[w1_ptr[k + 15]]};
                    wv2 = {lut[w2_ptr[k + 12]], lut[w2_ptr[k + 13]], lut[w2_ptr[k + 14]], lut[w2_ptr[k + 15]]};
                    wv3 = {lut[w3_ptr[k + 12]], lut[w3_ptr[k + 13]], lut[w3_ptr[k + 14]], lut[w3_ptr[k + 15]]};

                    acc00 = vfmaq_f32(acc00, xv0, wv0);
                    acc01 = vfmaq_f32(acc01, xv0, wv1);
                    acc02 = vfmaq_f32(acc02, xv0, wv2);
                    acc03 = vfmaq_f32(acc03, xv0, wv3);

                    acc10 = vfmaq_f32(acc10, xv1, wv0);
                    acc11 = vfmaq_f32(acc11, xv1, wv1);
                    acc12 = vfmaq_f32(acc12, xv1, wv2);
                    acc13 = vfmaq_f32(acc13, xv1, wv3);

                    acc20 = vfmaq_f32(acc20, xv2, wv0);
                    acc21 = vfmaq_f32(acc21, xv2, wv1);
                    acc22 = vfmaq_f32(acc22, xv2, wv2);
                    acc23 = vfmaq_f32(acc23, xv2, wv3);

                    acc30 = vfmaq_f32(acc30, xv3, wv0);
                    acc31 = vfmaq_f32(acc31, xv3, wv1);
                    acc32 = vfmaq_f32(acc32, xv3, wv2);
                    acc33 = vfmaq_f32(acc33, xv3, wv3);
                }

                // Remaining 4-element chunks
                for (; k < k_rem4; k += 4)
                {
                    float32x4_t xv0 = vld1q_f32(x0_ptr + k);
                    float32x4_t xv1 = vld1q_f32(x1_ptr + k);
                    float32x4_t xv2 = vld1q_f32(x2_ptr + k);
                    float32x4_t xv3 = vld1q_f32(x3_ptr + k);

                    float32x4_t wv0 = {lut[w0_ptr[k + 0]], lut[w0_ptr[k + 1]], lut[w0_ptr[k + 2]], lut[w0_ptr[k + 3]]};
                    float32x4_t wv1 = {lut[w1_ptr[k + 0]], lut[w1_ptr[k + 1]], lut[w1_ptr[k + 2]], lut[w1_ptr[k + 3]]};
                    float32x4_t wv2 = {lut[w2_ptr[k + 0]], lut[w2_ptr[k + 1]], lut[w2_ptr[k + 2]], lut[w2_ptr[k + 3]]};
                    float32x4_t wv3 = {lut[w3_ptr[k + 0]], lut[w3_ptr[k + 1]], lut[w3_ptr[k + 2]], lut[w3_ptr[k + 3]]};

                    acc00 = vfmaq_f32(acc00, xv0, wv0);
                    acc01 = vfmaq_f32(acc01, xv0, wv1);
                    acc02 = vfmaq_f32(acc02, xv0, wv2);
                    acc03 = vfmaq_f32(acc03, xv0, wv3);

                    acc10 = vfmaq_f32(acc10, xv1, wv0);
                    acc11 = vfmaq_f32(acc11, xv1, wv1);
                    acc12 = vfmaq_f32(acc12, xv1, wv2);
                    acc13 = vfmaq_f32(acc13, xv1, wv3);

                    acc20 = vfmaq_f32(acc20, xv2, wv0);
                    acc21 = vfmaq_f32(acc21, xv2, wv1);
                    acc22 = vfmaq_f32(acc22, xv2, wv2);
                    acc23 = vfmaq_f32(acc23, xv2, wv3);

                    acc30 = vfmaq_f32(acc30, xv3, wv0);
                    acc31 = vfmaq_f32(acc31, xv3, wv1);
                    acc32 = vfmaq_f32(acc32, xv3, wv2);
                    acc33 = vfmaq_f32(acc33, xv3, wv3);
                }

                // Horizontal reduction
                float s00 = vaddvq_f32(acc00), s01 = vaddvq_f32(acc01), s02 = vaddvq_f32(acc02),
                      s03 = vaddvq_f32(acc03);
                float s10 = vaddvq_f32(acc10), s11 = vaddvq_f32(acc11), s12 = vaddvq_f32(acc12),
                      s13 = vaddvq_f32(acc13);
                float s20 = vaddvq_f32(acc20), s21 = vaddvq_f32(acc21), s22 = vaddvq_f32(acc22),
                      s23 = vaddvq_f32(acc23);
                float s30 = vaddvq_f32(acc30), s31 = vaddvq_f32(acc31), s32 = vaddvq_f32(acc32),
                      s33 = vaddvq_f32(acc33);

                // Tail reduction along K
                for (uint32_t kt = k_rem4; kt < K; ++kt)
                {
                    float x0_val = x0_ptr[kt], x1_val = x1_ptr[kt], x2_val = x2_ptr[kt], x3_val = x3_ptr[kt];
                    float w0_val = lut[w0_ptr[kt]], w1_val = lut[w1_ptr[kt]], w2_val = lut[w2_ptr[kt]],
                          w3_val = lut[w3_ptr[kt]];

                    s00 += x0_val * w0_val;
                    s01 += x0_val * w1_val;
                    s02 += x0_val * w2_val;
                    s03 += x0_val * w3_val;
                    s10 += x1_val * w0_val;
                    s11 += x1_val * w1_val;
                    s12 += x1_val * w2_val;
                    s13 += x1_val * w3_val;
                    s20 += x2_val * w0_val;
                    s21 += x2_val * w1_val;
                    s22 += x2_val * w2_val;
                    s23 += x2_val * w3_val;
                    s30 += x3_val * w0_val;
                    s31 += x3_val * w1_val;
                    s32 += x3_val * w2_val;
                    s33 += x3_val * w3_val;
                }

                // Vectorized row stores (assigned to lvalues to avoid macro argument comma parsing)
                float32x4_t row0 = {s00, s01, s02, s03};
                float32x4_t row1 = {s10, s11, s12, s13};
                float32x4_t row2 = {s20, s21, s22, s23};
                float32x4_t row3 = {s30, s31, s32, s33};

                vst1q_f32(Out + static_cast<uint64_t>(m + 0) * N + n, row0);
                vst1q_f32(Out + static_cast<uint64_t>(m + 1) * N + n, row1);
                vst1q_f32(Out + static_cast<uint64_t>(m + 2) * N + n, row2);
                vst1q_f32(Out + static_cast<uint64_t>(m + 3) * N + n, row3);
            }

            // N-remainder for 4-row M block
            for (uint32_t n = n_rem; n < n_end; ++n)
            {
                const uint8_t *w_ptr = W + static_cast<uint64_t>(n) * K;
                for (uint32_t mi = 0; mi < 4; ++mi)
                {
                    const float *x_ptr = X + static_cast<uint64_t>(m + mi) * K;
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    uint32_t k = 0;
                    for (; k < k_rem4; k += 4)
                    {
                        float32x4_t xv = vld1q_f32(x_ptr + k);
                        float32x4_t wv = {lut[w_ptr[k + 0]], lut[w_ptr[k + 1]], lut[w_ptr[k + 2]], lut[w_ptr[k + 3]]};
                        acc = vfmaq_f32(acc, xv, wv);
                    }
                    float sum = vaddvq_f32(acc);
                    for (; k < K; ++k)
                    {
                        sum += x_ptr[k] * lut[w_ptr[k]];
                    }
                    Out[static_cast<uint64_t>(m + mi) * N + n] = sum;
                }
            }
        }

        // M-remainder rows
        for (uint32_t m = m_rem; m < M; ++m)
        {
            const float *x_ptr = X + static_cast<uint64_t>(m) * K;

            for (uint32_t n = n_start; n < n_rem; n += 4)
            {
                const uint8_t *w0_ptr = W + static_cast<uint64_t>(n + 0) * K;
                const uint8_t *w1_ptr = W + static_cast<uint64_t>(n + 1) * K;
                const uint8_t *w2_ptr = W + static_cast<uint64_t>(n + 2) * K;
                const uint8_t *w3_ptr = W + static_cast<uint64_t>(n + 3) * K;

                float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f), acc2 = vdupq_n_f32(0.0f),
                            acc3 = vdupq_n_f32(0.0f);

                uint32_t k = 0;
                for (; k < k_rem4; k += 4)
                {
                    float32x4_t xv = vld1q_f32(x_ptr + k);
                    float32x4_t wv0 = {lut[w0_ptr[k + 0]], lut[w0_ptr[k + 1]], lut[w0_ptr[k + 2]], lut[w0_ptr[k + 3]]};
                    float32x4_t wv1 = {lut[w1_ptr[k + 0]], lut[w1_ptr[k + 1]], lut[w1_ptr[k + 2]], lut[w1_ptr[k + 3]]};
                    float32x4_t wv2 = {lut[w2_ptr[k + 0]], lut[w2_ptr[k + 1]], lut[w2_ptr[k + 2]], lut[w2_ptr[k + 3]]};
                    float32x4_t wv3 = {lut[w3_ptr[k + 0]], lut[w3_ptr[k + 1]], lut[w3_ptr[k + 2]], lut[w3_ptr[k + 3]]};

                    acc0 = vfmaq_f32(acc0, xv, wv0);
                    acc1 = vfmaq_f32(acc1, xv, wv1);
                    acc2 = vfmaq_f32(acc2, xv, wv2);
                    acc3 = vfmaq_f32(acc3, xv, wv3);
                }

                float s0 = vaddvq_f32(acc0), s1 = vaddvq_f32(acc1), s2 = vaddvq_f32(acc2), s3 = vaddvq_f32(acc3);
                for (; k < K; ++k)
                {
                    float xv = x_ptr[k];
                    s0 += xv * lut[w0_ptr[k]];
                    s1 += xv * lut[w1_ptr[k]];
                    s2 += xv * lut[w2_ptr[k]];
                    s3 += xv * lut[w3_ptr[k]];
                }

                float32x4_t row = {s0, s1, s2, s3};
                vst1q_f32(Out + static_cast<uint64_t>(m) * N + n, row);
            }

            for (uint32_t n = n_rem; n < n_end; ++n)
            {
                const uint8_t *w_ptr = W + static_cast<uint64_t>(n) * K;
                float32x4_t acc = vdupq_n_f32(0.0f);
                uint32_t k = 0;
                for (; k < k_rem4; k += 4)
                {
                    float32x4_t xv = vld1q_f32(x_ptr + k);
                    float32x4_t wv = {lut[w_ptr[k + 0]], lut[w_ptr[k + 1]], lut[w_ptr[k + 2]], lut[w_ptr[k + 3]]};
                    acc = vfmaq_f32(acc, xv, wv);
                }
                float sum = vaddvq_f32(acc);
                for (; k < K; ++k)
                {
                    sum += x_ptr[k] * lut[w_ptr[k]];
                }
                Out[static_cast<uint64_t>(m) * N + n] = sum;
            }
        }
    });
}

inline LogicalId refFactoryF8E4M3TransposedGEMM_v2(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId w_cast = graph.cast(inputs[1], DType::FLOAT32);
    int32_t perm[] = {1, 0};
    LogicalId w_t = graph.contiguous(graph.permute(w_cast, graph.constant({2}, perm, DType::INT32)));
    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, static_cast<int32_t>(w_shape[1]), static_cast<int32_t>(w_shape[0])};
    return graph.dot(inputs[0], graph.reshape(w_t, graph.constant({3}, s3, DType::INT32)));
}

REGISTER_KERNEL("F8_E4M3_Transposed_GEMM_NEON_v2", 2, 2, matchF8E4M3TransposedGEMM_v2, runF8E4M3TransposedGEMM_v2,
                refFactoryF8E4M3TransposedGEMM_v2, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::F8_E4M3}, {{1, 8, 2048}, {2048, 2048}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON