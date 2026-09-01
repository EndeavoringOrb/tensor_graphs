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
#endif

inline bool matchF8E4M3TransposedGEMMv4(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

static inline float fp8_v4_e4m3_lut_val(uint8_t input)
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

static inline const float *get_v4_fp8_e4m3_lut()
{
    static const auto lut = []() {
        alignas(64) std::array<float, 256> table{};
        for (int i = 0; i < 256; ++i)
            table[i] = fp8_v4_e4m3_lut_val(static_cast<uint8_t>(i));
        return table;
    }();
    return lut.data();
}

inline void runF8E4M3TransposedGEMMv4(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const uint8_t *W = static_cast<const uint8_t *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t B = ctx.inViews[0].getShape()[0];
    const uint32_t S = ctx.inViews[0].getShape()[1];
    const uint32_t K = ctx.inViews[0].getShape()[2];
    const uint32_t N = ctx.inViews[1].getShape()[0];

    const uint32_t M = B * S;
    const float *lut = get_v4_fp8_e4m3_lut();

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    // Outer parallelization along N dimension
    uint32_t n_panels = (N + 63u) / 64u;
    num_threads = std::min(num_threads, std::max(1u, n_panels));

    uint32_t n_per_thread = ((N + num_threads - 1) / num_threads + 3u) & ~3u;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t n_start = t * n_per_thread;
        uint32_t n_end = std::min(n_start + n_per_thread, N);
        if (n_start >= n_end)
            return;

        constexpr uint32_t N_BLK = 64;
        constexpr uint32_t K_BLK = 256;
        std::vector<float> W_tile(N_BLK * K_BLK);

        for (uint32_t n_outer = n_start; n_outer < n_end; n_outer += N_BLK)
        {
            uint32_t cur_N = std::min(N_BLK, n_end - n_outer);
            uint32_t cur_N4 = cur_N & ~3u;

            for (uint32_t k_outer = 0; k_outer < K; k_outer += K_BLK)
            {
                uint32_t cur_K = std::min(K_BLK, K - k_outer);
                uint32_t cur_K4 = cur_K & ~3u;
                bool is_first_k = (k_outer == 0);

                // Pack / unpack FP8 chunk into L1/L2 tile
                for (uint32_t ni = 0; ni < cur_N; ++ni)
                {
                    const uint8_t *src_w = W + static_cast<uint64_t>(n_outer + ni) * K + k_outer;
                    float *dst_w = W_tile.data() + static_cast<uint64_t>(ni) * K_BLK;
                    for (uint32_t ki = 0; ki < cur_K; ++ki)
                    {
                        dst_w[ki] = lut[src_w[ki]];
                    }
                }

                for (uint32_t m = 0; m < (M & ~3u); m += 4)
                {
                    const float *x0 = X + static_cast<uint64_t>(m + 0) * K + k_outer;
                    const float *x1 = X + static_cast<uint64_t>(m + 1) * K + k_outer;
                    const float *x2 = X + static_cast<uint64_t>(m + 2) * K + k_outer;
                    const float *x3 = X + static_cast<uint64_t>(m + 3) * K + k_outer;

                    for (uint32_t ni = 0; ni < cur_N4; ni += 4)
                    {
                        const float *w0 = W_tile.data() + static_cast<uint64_t>(ni + 0) * K_BLK;
                        const float *w1 = W_tile.data() + static_cast<uint64_t>(ni + 1) * K_BLK;
                        const float *w2 = W_tile.data() + static_cast<uint64_t>(ni + 2) * K_BLK;
                        const float *w3 = W_tile.data() + static_cast<uint64_t>(ni + 3) * K_BLK;

                        float *out0 = Out + static_cast<uint64_t>(m + 0) * N + n_outer + ni;
                        float *out1 = Out + static_cast<uint64_t>(m + 1) * N + n_outer + ni;
                        float *out2 = Out + static_cast<uint64_t>(m + 2) * N + n_outer + ni;
                        float *out3 = Out + static_cast<uint64_t>(m + 3) * N + n_outer + ni;

#if defined(TG_HAS_NEON)
                        float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f), c02 = vdupq_n_f32(0.0f), c03 = vdupq_n_f32(0.0f);
                        float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f), c12 = vdupq_n_f32(0.0f), c13 = vdupq_n_f32(0.0f);
                        float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f), c22 = vdupq_n_f32(0.0f), c23 = vdupq_n_f32(0.0f);
                        float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f), c32 = vdupq_n_f32(0.0f), c33 = vdupq_n_f32(0.0f);

                        for (uint32_t k = 0; k < cur_K4; k += 4)
                        {
                            float32x4_t xv0 = vld1q_f32(x0 + k);
                            float32x4_t xv1 = vld1q_f32(x1 + k);
                            float32x4_t xv2 = vld1q_f32(x2 + k);
                            float32x4_t xv3 = vld1q_f32(x3 + k);

                            float32x4_t wv0 = vld1q_f32(w0 + k);
                            float32x4_t wv1 = vld1q_f32(w1 + k);
                            float32x4_t wv2 = vld1q_f32(w2 + k);
                            float32x4_t wv3 = vld1q_f32(w3 + k);

                            c00 = vfmaq_f32(c00, xv0, wv0);
                            c01 = vfmaq_f32(c01, xv0, wv1);
                            c02 = vfmaq_f32(c02, xv0, wv2);
                            c03 = vfmaq_f32(c03, xv0, wv3);

                            c10 = vfmaq_f32(c10, xv1, wv0);
                            c11 = vfmaq_f32(c11, xv1, wv1);
                            c12 = vfmaq_f32(c12, xv1, wv2);
                            c13 = vfmaq_f32(c13, xv1, wv3);

                            c20 = vfmaq_f32(c20, xv2, wv0);
                            c21 = vfmaq_f32(c21, xv2, wv1);
                            c22 = vfmaq_f32(c22, xv2, wv2);
                            c23 = vfmaq_f32(c23, xv2, wv3);

                            c30 = vfmaq_f32(c30, xv3, wv0);
                            c31 = vfmaq_f32(c31, xv3, wv1);
                            c32 = vfmaq_f32(c32, xv3, wv2);
                            c33 = vfmaq_f32(c33, xv3, wv3);
                        }

                        float s00 = vaddvq_f32(c00), s01 = vaddvq_f32(c01), s02 = vaddvq_f32(c02), s03 = vaddvq_f32(c03);
                        float s10 = vaddvq_f32(c10), s11 = vaddvq_f32(c11), s12 = vaddvq_f32(c12), s13 = vaddvq_f32(c13);
                        float s20 = vaddvq_f32(c20), s21 = vaddvq_f32(c21), s22 = vaddvq_f32(c22), s23 = vaddvq_f32(c23);
                        float s30 = vaddvq_f32(c30), s31 = vaddvq_f32(c31), s32 = vaddvq_f32(c32), s33 = vaddvq_f32(c33);

                        for (uint32_t k = cur_K4; k < cur_K; ++k)
                        {
                            s00 += x0[k] * w0[k]; s01 += x0[k] * w1[k]; s02 += x0[k] * w2[k]; s03 += x0[k] * w3[k];
                            s10 += x1[k] * w0[k]; s11 += x1[k] * w1[k]; s12 += x1[k] * w2[k]; s13 += x1[k] * w3[k];
                            s20 += x2[k] * w0[k]; s21 += x2[k] * w1[k]; s22 += x2[k] * w2[k]; s23 += x2[k] * w3[k];
                            s30 += x3[k] * w0[k]; s31 += x3[k] * w1[k]; s32 += x3[k] * w2[k]; s33 += x3[k] * w3[k];
                        }

                        if (is_first_k)
                        {
                            float32x4_t r0 = {s00, s01, s02, s03};
                            float32x4_t r1 = {s10, s11, s12, s13};
                            float32x4_t r2 = {s20, s21, s22, s23};
                            float32x4_t r3 = {s30, s31, s32, s33};
                            vst1q_f32(out0, r0);
                            vst1q_f32(out1, r1);
                            vst1q_f32(out2, r2);
                            vst1q_f32(out3, r3);
                        }
                        else
                        {
                            float32x4_t r0 = vaddq_f32(vld1q_f32(out0), float32x4_t{s00, s01, s02, s03});
                            float32x4_t r1 = vaddq_f32(vld1q_f32(out1), float32x4_t{s10, s11, s12, s13});
                            float32x4_t r2 = vaddq_f32(vld1q_f32(out2), float32x4_t{s20, s21, s22, s23});
                            float32x4_t r3 = vaddq_f32(vld1q_f32(out3), float32x4_t{s30, s31, s32, s33});
                            vst1q_f32(out0, r0);
                            vst1q_f32(out1, r1);
                            vst1q_f32(out2, r2);
                            vst1q_f32(out3, r3);
                        }
#else
                        for (uint32_t r = 0; r < 4; ++r)
                        {
                            const float *xr = X + static_cast<uint64_t>(m + r) * K + k_outer;
                            float *out_r = Out + static_cast<uint64_t>(m + r) * N + n_outer + ni;
                            for (uint32_t c = 0; c < 4; ++c)
                            {
                                const float *wc = W_tile.data() + static_cast<uint64_t>(ni + c) * K_BLK;
                                float sum = is_first_k ? 0.0f : out_r[c];
                                for (uint32_t k = 0; k < cur_K; ++k)
                                    sum += xr[k] * wc[k];
                                out_r[c] = sum;
                            }
                        }
#endif
                    }
                }
            }
        }
    });
}

inline LogicalId refFactoryF8E4M3TransposedGEMMv4(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId w_cast = graph.cast(inputs[1], DType::FLOAT32);
    int32_t perm[] = {1, 0};
    LogicalId w_t = graph.contiguous(graph.permute(w_cast, graph.constant({2}, perm, DType::INT32)));
    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, static_cast<int32_t>(w_shape[1]), static_cast<int32_t>(w_shape[0])};
    return graph.dot(inputs[0], graph.reshape(w_t, graph.constant({3}, s3, DType::INT32)));
}

REGISTER_KERNEL("F8_E4M3_Transposed_GEMM_NEON_v4", 2, 2, matchF8E4M3TransposedGEMMv4, runF8E4M3TransposedGEMMv4,
                refFactoryF8E4M3TransposedGEMMv4, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::F8_E4M3}, {{1, 4224, 6144}, {6144, 6144}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});