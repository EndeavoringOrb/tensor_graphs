#pragma once
#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

inline bool matchF8E4M3TransposedGEMM(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

static inline float f8e4m3fn_to_fp32_lut_val(uint8_t input)
{
    if (input == 0x7F || input == 0xFF)
        return std::numeric_limits<float>::quiet_NaN();
    float sign = (input & 0x80) ? -1.0f : 1.0f;
    uint32_t exp = (input & 0x78) >> 3;
    uint32_t mant = input & 0x07;
    if (exp == 0)
    {
        if (mant == 0)
            return sign * 0.0f;
        return sign * std::ldexp(static_cast<float>(mant), -9);
    }
    return sign * std::ldexp(1.0f + static_cast<float>(mant) * 0.125f, static_cast<int>(exp) - 7);
}

static inline const float *get_f8e4m3_gemm_lut()
{
    static const auto lut = []() {
        std::array<float, 256> table{};
        for (int i = 0; i < 256; ++i)
            table[i] = f8e4m3fn_to_fp32_lut_val(static_cast<uint8_t>(i));
        return table;
    }();
    return lut.data();
}

inline void runF8E4M3TransposedGEMM(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const uint8_t *W = static_cast<const uint8_t *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t B = ctx.inViews[0].getShape()[0];
    const uint32_t S = ctx.inViews[0].getShape()[1];
    const uint32_t K = ctx.inViews[0].getShape()[2];
    const uint32_t N = ctx.inViews[1].getShape()[0];

    const float *lut = get_f8e4m3_gemm_lut();

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    num_threads = std::min(num_threads, std::max(1u, N / 4u));

    uint32_t n_block = ((N + num_threads - 1) / num_threads + 3u) & ~3u;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t n_start = t * n_block;
        if (n_start >= N)
            return;
        uint32_t n_end = std::min(n_start + n_block, N);

        const uint32_t K4 = K & ~3u;
        const uint32_t S4 = S & ~3u;
        const uint32_t N4 = n_end & ~3u;

        for (uint32_t b = 0; b < B; ++b)
        {
            const float *batch_x = X + static_cast<uint64_t>(b) * S * K;
            float *batch_out = Out + static_cast<uint64_t>(b) * S * N;

            // 4x4 tile micro-kernel (4 S-rows x 4 N-columns)
            for (uint32_t s = 0; s < S4; s += 4)
            {
                const float *x0_ptr = batch_x + static_cast<uint64_t>(s + 0) * K;
                const float *x1_ptr = batch_x + static_cast<uint64_t>(s + 1) * K;
                const float *x2_ptr = batch_x + static_cast<uint64_t>(s + 2) * K;
                const float *x3_ptr = batch_x + static_cast<uint64_t>(s + 3) * K;

                for (uint32_t n = n_start; n < N4; n += 4)
                {
                    const uint8_t *w0 = W + static_cast<uint64_t>(n + 0) * K;
                    const uint8_t *w1 = W + static_cast<uint64_t>(n + 1) * K;
                    const uint8_t *w2 = W + static_cast<uint64_t>(n + 2) * K;
                    const uint8_t *w3 = W + static_cast<uint64_t>(n + 3) * K;

#if defined(TG_HAS_NEON)
                    float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f), c02 = vdupq_n_f32(0.0f),
                                c03 = vdupq_n_f32(0.0f);
                    float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f), c12 = vdupq_n_f32(0.0f),
                                c13 = vdupq_n_f32(0.0f);
                    float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f), c22 = vdupq_n_f32(0.0f),
                                c23 = vdupq_n_f32(0.0f);
                    float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f), c32 = vdupq_n_f32(0.0f),
                                c33 = vdupq_n_f32(0.0f);

                    for (uint32_t k = 0; k < K4; k += 4)
                    {
                        float32x4_t x0 = vld1q_f32(x0_ptr + k);
                        float32x4_t x1 = vld1q_f32(x1_ptr + k);
                        float32x4_t x2 = vld1q_f32(x2_ptr + k);
                        float32x4_t x3 = vld1q_f32(x3_ptr + k);

                        float32x4_t w0_v = {lut[w0[k]], lut[w0[k + 1]], lut[w0[k + 2]], lut[w0[k + 3]]};
                        float32x4_t w1_v = {lut[w1[k]], lut[w1[k + 1]], lut[w1[k + 2]], lut[w1[k + 3]]};
                        float32x4_t w2_v = {lut[w2[k]], lut[w2[k + 1]], lut[w2[k + 2]], lut[w2[k + 3]]};
                        float32x4_t w3_v = {lut[w3[k]], lut[w3[k + 1]], lut[w3[k + 2]], lut[w3[k + 3]]};

                        c00 = vfmaq_f32(c00, x0, w0_v);
                        c01 = vfmaq_f32(c01, x0, w1_v);
                        c02 = vfmaq_f32(c02, x0, w2_v);
                        c03 = vfmaq_f32(c03, x0, w3_v);

                        c10 = vfmaq_f32(c10, x1, w0_v);
                        c11 = vfmaq_f32(c11, x1, w1_v);
                        c12 = vfmaq_f32(c12, x1, w2_v);
                        c13 = vfmaq_f32(c13, x1, w3_v);

                        c20 = vfmaq_f32(c20, x2, w0_v);
                        c21 = vfmaq_f32(c21, x2, w1_v);
                        c22 = vfmaq_f32(c22, x2, w2_v);
                        c23 = vfmaq_f32(c23, x2, w3_v);

                        c30 = vfmaq_f32(c30, x3, w0_v);
                        c31 = vfmaq_f32(c31, x3, w1_v);
                        c32 = vfmaq_f32(c32, x3, w2_v);
                        c33 = vfmaq_f32(c33, x3, w3_v);
                    }

                    auto store_res = [&](uint32_t s_off, float32x4_t a0, float32x4_t a1, float32x4_t a2,
                                         float32x4_t a3) {
                        float *out_row = batch_out + static_cast<uint64_t>(s + s_off) * N + n;
                        float res[4] = {vaddvq_f32(a0), vaddvq_f32(a1), vaddvq_f32(a2), vaddvq_f32(a3)};
                        const float *x_tail = batch_x + static_cast<uint64_t>(s + s_off) * K;
                        for (uint32_t k = K4; k < K; ++k)
                        {
                            float xv = x_tail[k];
                            res[0] += xv * lut[w0[k]];
                            res[1] += xv * lut[w1[k]];
                            res[2] += xv * lut[w2[k]];
                            res[3] += xv * lut[w3[k]];
                        }
                        vst1q_f32(out_row, vld1q_f32(res));
                    };

                    store_res(0, c00, c01, c02, c03);
                    store_res(1, c10, c11, c12, c13);
                    store_res(2, c20, c21, c22, c23);
                    store_res(3, c30, c31, c32, c33);
#else
                    for (uint32_t si = 0; si < 4; ++si)
                    {
                        const float *x_row = batch_x + static_cast<uint64_t>(s + si) * K;
                        float *out_row = batch_out + static_cast<uint64_t>(s + si) * N + n;
                        for (uint32_t ni = 0; ni < 4; ++ni)
                        {
                            const uint8_t *w_row = W + static_cast<uint64_t>(n + ni) * K;
                            float sum = 0.0f;
                            for (uint32_t k = 0; k < K; ++k)
                                sum += x_row[k] * lut[w_row[k]];
                            out_row[ni] = sum;
                        }
                    }
#endif
                }
            }

            // S-tail
            for (uint32_t s = S4; s < S; ++s)
            {
                const float *x_row = batch_x + static_cast<uint64_t>(s) * K;
                for (uint32_t n = n_start; n < N4; n += 4)
                {
                    const uint8_t *w0 = W + static_cast<uint64_t>(n + 0) * K;
                    const uint8_t *w1 = W + static_cast<uint64_t>(n + 1) * K;
                    const uint8_t *w2 = W + static_cast<uint64_t>(n + 2) * K;
                    const uint8_t *w3 = W + static_cast<uint64_t>(n + 3) * K;

                    float *out_row = batch_out + static_cast<uint64_t>(s) * N + n;
#if defined(TG_HAS_NEON)
                    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f), a2 = vdupq_n_f32(0.0f),
                                a3 = vdupq_n_f32(0.0f);
                    for (uint32_t k = 0; k < K4; k += 4)
                    {
                        float32x4_t xv = vld1q_f32(x_row + k);
                        a0 = vfmaq_f32(a0, xv, float32x4_t{lut[w0[k]], lut[w0[k + 1]], lut[w0[k + 2]], lut[w0[k + 3]]});
                        a1 = vfmaq_f32(a1, xv, float32x4_t{lut[w1[k]], lut[w1[k + 1]], lut[w1[k + 2]], lut[w1[k + 3]]});
                        a2 = vfmaq_f32(a2, xv, float32x4_t{lut[w2[k]], lut[w2[k + 1]], lut[w2[k + 2]], lut[w2[k + 3]]});
                        a3 = vfmaq_f32(a3, xv, float32x4_t{lut[w3[k]], lut[w3[k + 1]], lut[w3[k + 2]], lut[w3[k + 3]]});
                    }
                    float res[4] = {vaddvq_f32(a0), vaddvq_f32(a1), vaddvq_f32(a2), vaddvq_f32(a3)};
                    for (uint32_t k = K4; k < K; ++k)
                    {
                        float xv = x_row[k];
                        res[0] += xv * lut[w0[k]];
                        res[1] += xv * lut[w1[k]];
                        res[2] += xv * lut[w2[k]];
                        res[3] += xv * lut[w3[k]];
                    }
                    vst1q_f32(out_row, vld1q_f32(res));
#else
                    for (uint32_t ni = 0; ni < 4; ++ni)
                    {
                        const uint8_t *w_row = W + static_cast<uint64_t>(n + ni) * K;
                        float sum = 0.0f;
                        for (uint32_t k = 0; k < K; ++k)
                            sum += x_row[k] * lut[w_row[k]];
                        out_row[ni] = sum;
                    }
#endif
                }
            }

            // N-tail
            for (uint32_t n = N4; n < n_end; ++n)
            {
                const uint8_t *w_row = W + static_cast<uint64_t>(n) * K;
                for (uint32_t s = 0; s < S; ++s)
                {
                    const float *x_row = batch_x + static_cast<uint64_t>(s) * K;
                    float sum = 0.0f;
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        sum += x_row[k] * lut[w_row[k]];
                    }
                    batch_out[static_cast<uint64_t>(s) * N + n] = sum;
                }
            }
        }
    });
}

inline LogicalId refFactoryF8E4M3TransposedGEMM(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId w_cast = graph.cast(inputs[1], DType::FLOAT32);
    int32_t perm[] = {1, 0};
    LogicalId w_t = graph.contiguous(graph.permute(w_cast, graph.constant({2}, perm, DType::INT32)));
    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, static_cast<int32_t>(w_shape[1]), static_cast<int32_t>(w_shape[0])};
    return graph.dot(inputs[0], graph.reshape(w_t, graph.constant({3}, s3, DType::INT32)));
}

REGISTER_KERNEL("F8_E4M3_Transposed_GEMM_NEON", 2, 2, matchF8E4M3TransposedGEMM, runF8E4M3TransposedGEMM,
                refFactoryF8E4M3TransposedGEMM, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::F8_E4M3}, {{1, 8, 64}, {1024, 64}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});