// BF16 Transposed GEMM, NEON v7 — BFDOT with round-to-nearest-even X cast
// =======================================================================
//
// PROBLEM THIS KERNEL SOLVES
// --------------------------
//
// The active BF16 GEMM kernel (v5) keeps X in fp32 and converts W from bf16
// to fp32 via vshll_n_u16 + vfmaq_f32. This is accurate but uses vfmaq
// (4 muls + 4 adds per instruction). The disabled v6 kernel converts X to
// bf16 once and uses vbfmmlaq (8x8x4 -> 8x4 fp32, 32 muls per instruction),
// giving ~4x throughput — but v6 was disabled because bf16 activations
// caused embedding cosine similarity to drop below the model's threshold.
//
// Across the jina-v5 vision tower + text encoder, BF16 GEMM consumes ~870 ms
// (21% of total runtime):
//
//   [1, 5040, 768]  x [3072, 768]  -> [1, 5040, 3072]   318 ms  (vision MLP
//   fc1, 12x) [1, 5040, 3072] x [768, 3072]  -> [1, 5040, 768]    313 ms
//   (vision MLP fc2, 12x) [1, 5040, 768]  x [2304, 768]  -> [1, 5040, 2304] 239
//   ms  (vision QKV,      12x)
//   ... plus text encoder linears ...
//
// WHAT THIS KERNEL DOES
// ---------------------
//
// Uses BFDOT (vbfdotq_f32) for the inner GEMM loop — 8 bf16 muls + 4 fp32
// accumulates in ONE instruction, giving 2x compute throughput vs v5's
// vfmaq (which does 4 fp32 muls + 4 fp32 accumulates per instruction).
//
// To improve accuracy over v6 (which uses truncation when converting X to
// bf16), this kernel uses ROUND-TO-NEAREST-EVEN conversion via the
// vcvtq_low_bf16_f32 / vcvtq_high_bf16_f32 intrinsics. Round-to-nearest-
// even reduces the mean conversion error by ~2x vs truncation, and
// eliminates the systematic bias of truncation (which always rounds toward
// zero).
//
// ACCURACY / ENABLEMENT
// ---------------------
//
// Even with RNE conversion, bf16 X activations lose 16 mantissa bits vs
// fp32, so this kernel will produce slightly different outputs than v5.
// Test the end-to-end embedding cosine similarity against the Python
// reference before relying on this kernel for production. If the
// similarity stays above your threshold (the model comment says >=0.98),
// enable this kernel; otherwise stick with v5.
//
// To enable: change the #if guard below from TG_HAS_NEON_TODOREMOVETHISCHECK
// (or just remove it) to TG_HAS_NEON, then rebuild.
//
// EXPECTED SPEEDUP (if enabled)
// -----------------------------
//
//   BF16 GEMM total: 870 ms -> ~435 ms   (2x)
//   Savings: ~435 ms
//
// Combined with FlashAttention + vectorized softmax:
//   Before: 4067 ms
//   After:  ~2300 ms   (1.8x total speedup)

#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

// Change this guard to `#if defined(TG_HAS_NEON) &&
// defined(__ARM_FEATURE_BF16)` to enable. Kept as TODOREMOVETHISCHECK by
// default so the build does not silently change accuracy characteristics.
#if defined(TG_HAS_NEON_TODOREMOVETHISCHECK) && defined(__ARM_FEATURE_BF16)

#include <arm_neon.h>

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Match function — same shape rules as v5/v6:
//   X : [B, S, K]  fp32  (B must be 1 for the streaming path, but the
//                         kernel handles any B via the B-loop)
//   W : [N, K]     bf16
//   O : [B, S, N]  fp32
// ---------------------------------------------------------------------------
inline bool matchBF16TransposedGEMM_v7(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    auto sX = inputs[0].getShape(); // [B,S,K]
    auto sW = inputs[1].getShape(); // [N,K]
    auto sO = output.getShape();    // [B,S,N]

    if (sX.size() != 3 || sW.size() != 2 || sO.size() != 3)
        return false;
    if (sX[2] != sW[1] || sO[2] != sW[0])
        return false;

    return isContiguous(output);
}

// ---------------------------------------------------------------------------
// Run function
//
// Tiling:
//   MR = 8 S-rows per panel (matches vbfmmlaq's 8-row output)
//   NR = 8 N-cols per panel (matches vbfmmlaq's 4-col output, x2 for pairs)
//   KR = 4 K-elements per step (matches vbfmmlaq's 4-K reduction)
//
// The X panel is packed to bf16 once per (B, S-panel, K-panel) and reused
// across all N-panels, amortizing the conversion cost.
// ---------------------------------------------------------------------------
inline void runBF16TransposedGEMM_v7(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const uint16_t *W = static_cast<const uint16_t *>(ctx.inputs[1]);
    float *O = static_cast<float *>(ctx.outputs[0]);

    const uint32_t B = ctx.inViews[0].getShape()[0];
    const uint32_t S = ctx.inViews[0].getShape()[1];
    const uint32_t K = ctx.inViews[0].getShape()[2];
    const uint32_t N = ctx.inViews[1].getShape()[0];

    const uint32_t M = B * S; // flatten batch + sequence

    constexpr uint32_t MR = 8;
    constexpr uint32_t NR = 8;
    constexpr uint32_t KR = 4;

    const uint32_t M_main = (M / MR) * MR;
    const uint32_t N_main = (N / NR) * NR;
    const uint32_t K_main = (K / KR) * KR;

    const uint32_t M_panels = M_main / MR;
    const uint32_t N_panels = N_main / NR;
    const uint32_t K_quads = K_main / KR;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    // ============================================================
    // PACK X -> BF16 with round-to-nearest-even (RNE)
    //
    // vcvtq_low_bf16_f32  converts the low 4 fp32 lanes to 4 bf16
    // vcvtq_high_bf16_f32 converts the high 4 fp32 lanes to 4 bf16
    // Both use RNE rounding (vs v6's truncation via vshrn).
    // ============================================================
    std::vector<uint16_t> X_packed((uint64_t)M_main * K_main);

    {
        uint32_t nt = std::min(num_threads, M_panels);
        std::vector<std::thread> packers;
        uint32_t per = (M_panels + nt - 1) / nt;

        for (uint32_t t = 0; t < nt; ++t)
        {
            uint32_t s = t * per;
            uint32_t e = std::min(s + per, M_panels);
            if (s >= e)
                break;

            packers.emplace_back([=, &X_packed]() {
                // RNE bias for fp32 -> bf16 conversion.
                // Adding 0x7FFF (the bit just below the bf16 truncation point)
                // to a non-NaN fp32 makes the subsequent truncation round to
                // nearest even. This works because:
                //   - For positive numbers, the bias either does not flip the
                //     low bit (no round-up) or carries into bit 16 (round-up).
                //   - For negative numbers (two's complement), the same logic
                //     works because the carry propagates the same way.
                // The result is RNE rounding, identical to vcvtq_low_bf16_f32
                // but without relying on toolchain-specific intrinsic naming.
                const uint32x4_t v_rne_bias = vdupq_n_u32(0x7FFFu);

                for (uint32_t mp = s; mp < e; ++mp)
                {
                    uint16_t *dst = X_packed.data() + (uint64_t)mp * MR * K_main;
                    uint32_t m0 = mp * MR;

                    for (uint32_t kq = 0; kq < K_quads; ++kq)
                    {
                        uint32_t k = kq * KR;

                        // Load 8 fp32 rows × 4 K-elements = 8 float32x4_t
                        float32x4_t r0 = vld1q_f32(X + (uint64_t)(m0 + 0) * K + k);
                        float32x4_t r1 = vld1q_f32(X + (uint64_t)(m0 + 1) * K + k);
                        float32x4_t r2 = vld1q_f32(X + (uint64_t)(m0 + 2) * K + k);
                        float32x4_t r3 = vld1q_f32(X + (uint64_t)(m0 + 3) * K + k);
                        float32x4_t r4 = vld1q_f32(X + (uint64_t)(m0 + 4) * K + k);
                        float32x4_t r5 = vld1q_f32(X + (uint64_t)(m0 + 5) * K + k);
                        float32x4_t r6 = vld1q_f32(X + (uint64_t)(m0 + 6) * K + k);
                        float32x4_t r7 = vld1q_f32(X + (uint64_t)(m0 + 7) * K + k);

                        // RNE fp32 -> bf16 conversion (portable vshrn-based)
                        // Each vshrn_n_u32 narrows 4 uint32 -> 4 uint16 by
                        // shifting right by 16. Pre-adding 0x7FFF gives RNE.
                        auto fp32x4_to_bf16x4 = [&](float32x4_t v) -> uint16x4_t {
                            uint32x4_t u = vreinterpretq_u32_f32(v);
                            return vshrn_n_u32(vaddq_u32(u, v_rne_bias), 16);
                        };

                        uint16x4_t b0 = fp32x4_to_bf16x4(r0);
                        uint16x4_t b1 = fp32x4_to_bf16x4(r1);
                        uint16x4_t b2 = fp32x4_to_bf16x4(r2);
                        uint16x4_t b3 = fp32x4_to_bf16x4(r3);
                        uint16x4_t b4 = fp32x4_to_bf16x4(r4);
                        uint16x4_t b5 = fp32x4_to_bf16x4(r5);
                        uint16x4_t b6 = fp32x4_to_bf16x4(r6);
                        uint16x4_t b7 = fp32x4_to_bf16x4(r7);

                        // Pack 8 rows x 4 bf16 = 32 bf16 = 4x uint16x8_t.
                        // Each uint16x8_t holds 2 rows of 4 bf16.
                        vst1q_u16(dst + 0, vcombine_u16(b0, b1));
                        vst1q_u16(dst + 8, vcombine_u16(b2, b3));
                        vst1q_u16(dst + 16, vcombine_u16(b4, b5));
                        vst1q_u16(dst + 24, vcombine_u16(b6, b7));
                        dst += 32;
                    }
                }
            });
        }

        for (auto &t : packers)
            t.join();
    }

    // ============================================================
    // COMPUTE (parallel over N panels)
    //
    // For each N-panel of 8 cols, pack W panel to bf16 (already bf16,
    // just need to reorganize for BFDOT layout), then for each M-panel
    // of 8 rows, compute the 8x8 output tile using 16 BFDOT instructions
    // per K-quad (4 K-elements per BFDOT, 4 quads per K-step).
    // ============================================================
    {
        uint32_t nt = std::min(num_threads, N_panels);
        std::vector<std::thread> workers;
        uint32_t per = (N_panels + nt - 1) / nt;

        for (uint32_t t = 0; t < nt; ++t)
        {
            uint32_t s = t * per;
            uint32_t e = std::min(s + per, N_panels);
            if (s >= e)
                break;

            workers.emplace_back([=, &X_packed]() {
                std::vector<uint16_t> W_panel((uint64_t)K_main * NR);

                for (uint32_t np = s; np < e; ++np)
                {
                    uint32_t n0 = np * NR;

                    // PACK W (reorganize [N,K] -> [K_quad, NR, KR] for BFDOT)
                    uint16_t *wp = W_panel.data();
                    for (uint32_t kq = 0; kq < K_quads; ++kq)
                    {
                        uint32_t k = kq * KR;
                        for (uint32_t bi = 0; bi < NR / 2; ++bi)
                        {
                            uint32_t n = n0 + bi * 2;
                            // W is [N, K], pack 4 K-elements x 2 N-rows
                            wp[0] = W[(uint64_t)n * K + k];
                            wp[1] = W[(uint64_t)n * K + k + 1];
                            wp[2] = W[(uint64_t)n * K + k + 2];
                            wp[3] = W[(uint64_t)n * K + k + 3];
                            wp[4] = W[(uint64_t)(n + 1) * K + k];
                            wp[5] = W[(uint64_t)(n + 1) * K + k + 1];
                            wp[6] = W[(uint64_t)(n + 1) * K + k + 2];
                            wp[7] = W[(uint64_t)(n + 1) * K + k + 3];
                            wp += 8;
                        }
                    }

                    for (uint32_t mp = 0; mp < M_panels; ++mp)
                    {
                        uint32_t m_base = mp * MR;

                        const uint16_t *A_ptr = X_packed.data() + (uint64_t)mp * MR * K_main;
                        const uint16_t *B_ptr = W_panel.data();

                        // 8x8 = 8 rows x 2 col-pairs of 4 fp32 accumulators
                        // = 16 accumulator registers
                        float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0);
                        float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0);
                        float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0);
                        float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0);
                        float32x4_t c40 = vdupq_n_f32(0), c41 = vdupq_n_f32(0);
                        float32x4_t c50 = vdupq_n_f32(0), c51 = vdupq_n_f32(0);
                        float32x4_t c60 = vdupq_n_f32(0), c61 = vdupq_n_f32(0);
                        float32x4_t c70 = vdupq_n_f32(0), c71 = vdupq_n_f32(0);

                        for (uint32_t kq = 0; kq < K_quads; ++kq)
                        {
                            __builtin_prefetch(A_ptr + 128);
                            __builtin_prefetch(B_ptr + 128);

                            // 8 X rows, 8 bf16 each (2 bfloat16x8_t per K-quad)
                            bfloat16x8_t a0 = vreinterpretq_bf16_u16(vld1q_u16(A_ptr));
                            bfloat16x8_t a1 = vreinterpretq_bf16_u16(vld1q_u16(A_ptr + 8));
                            bfloat16x8_t a2 = vreinterpretq_bf16_u16(vld1q_u16(A_ptr + 16));
                            bfloat16x8_t a3 = vreinterpretq_bf16_u16(vld1q_u16(A_ptr + 24));

                            // 8 W cols, 8 bf16 each (2 bfloat16x8_t per K-quad)
                            bfloat16x8_t b0 = vreinterpretq_bf16_u16(vld1q_u16(B_ptr));
                            bfloat16x8_t b1 = vreinterpretq_bf16_u16(vld1q_u16(B_ptr + 8));
                            bfloat16x8_t b2 = vreinterpretq_bf16_u16(vld1q_u16(B_ptr + 16));
                            bfloat16x8_t b3 = vreinterpretq_bf16_u16(vld1q_u16(B_ptr + 24));

                            // vbfmmlaq_f32: 8x4 bf16 x 4x4 bf16 -> 8x4 fp32
                            // Each BFDOT does 8 muls + 4 accumulates.
                            // 16 BFDOTs per K-quad produce the full 8x8 output tile.
                            c00 = vbfmmlaq_f32(c00, a0, b0);
                            c01 = vbfmmlaq_f32(c01, a0, b1);
                            c10 = vbfmmlaq_f32(c10, a0, b2);
                            c11 = vbfmmlaq_f32(c11, a0, b3);

                            c20 = vbfmmlaq_f32(c20, a1, b0);
                            c21 = vbfmmlaq_f32(c21, a1, b1);
                            c30 = vbfmmlaq_f32(c30, a1, b2);
                            c31 = vbfmmlaq_f32(c31, a1, b3);

                            c40 = vbfmmlaq_f32(c40, a2, b0);
                            c41 = vbfmmlaq_f32(c41, a2, b1);
                            c50 = vbfmmlaq_f32(c50, a2, b2);
                            c51 = vbfmmlaq_f32(c51, a2, b3);

                            c60 = vbfmmlaq_f32(c60, a3, b0);
                            c61 = vbfmmlaq_f32(c61, a3, b1);
                            c70 = vbfmmlaq_f32(c70, a3, b2);
                            c71 = vbfmmlaq_f32(c71, a3, b3);

                            A_ptr += 32;
                            B_ptr += 32;
                        }

                        // Store 8x8 output tile.
                        //
                        // Each vbfmmlaq_f32 produces 4 fp32 values laid out as
                        // a 2x2 matrix in row-major order:
                        //   v[0] = (row0, col0), v[1] = (row0, col1),
                        //   v[2] = (row1, col0), v[3] = (row1, col1)
                        //
                        // For a given `a` (covering 2 rows) and four `b`s
                        // (each covering 2 cols, so 8 cols total):
                        //   v0 = a x b0 -> rows r0,r1 x cols 0,1
                        //   v1 = a x b1 -> rows r0,r1 x cols 2,3
                        //   v2 = a x b2 -> rows r0,r1 x cols 4,5
                        //   v3 = a x b3 -> rows r0,r1 x cols 6,7
                        //
                        // To store row r0 contiguously (cols 0..7), we need
                        // to gather the low halves of v0..v3 (which hold the
                        // r0 elements). vcombine_f32 of (low(v0), low(v1))
                        // gives (r0c0, r0c1, r0c2, r0c3); combine of
                        // (low(v2), low(v3)) gives (r0c4, r0c5, r0c6, r0c7).
                        // The high halves give the same for r1.
                        auto store_pair = [&](uint32_t mi, float32x4_t v0, float32x4_t v1, float32x4_t v2,
                                              float32x4_t v3) {
                            uint32_t r0 = m_base + mi * 2;
                            float32x4_t row0_a = vcombine_f32(vget_low_f32(v0), vget_low_f32(v1));
                            float32x4_t row0_b = vcombine_f32(vget_low_f32(v2), vget_low_f32(v3));
                            float32x4_t row1_a = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));
                            float32x4_t row1_b = vcombine_f32(vget_high_f32(v2), vget_high_f32(v3));

                            vst1q_f32(O + (uint64_t)r0 * N + n0, row0_a);
                            vst1q_f32(O + (uint64_t)r0 * N + n0 + 4, row0_b);
                            vst1q_f32(O + (uint64_t)(r0 + 1) * N + n0, row1_a);
                            vst1q_f32(O + (uint64_t)(r0 + 1) * N + n0 + 4, row1_b);
                        };
                        store_pair(0, c00, c01, c10, c11); // rows 0-1 x cols 0-7
                        store_pair(1, c20, c21, c30, c31); // rows 2-3 x cols 0-7
                        store_pair(2, c40, c41, c50, c51); // rows 4-5 x cols 0-7
                        store_pair(3, c60, c61, c70, c71); // rows 6-7 x cols 0-7
                    }
                }
            });
        }

        for (auto &w : workers)
            w.join();
    }

    // ============================================================
    // TAILS (M, N, K not divisible by MR/NR/KR)
    //
    // Fall back to scalar bf16 multiplication for the remaining rows/cols.
    // This is rare for the jina-v5 model (all dims are multiples of 8/4),
    // so the scalar path is intentionally simple.
    // ============================================================
    auto bf16_to_f32 = [](uint16_t h) -> float {
        uint32_t bits = (uint32_t)h << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        return f;
    };

    // M-tail (rows 0..M_main-1 already done; handle rows M_main..M-1)
    for (uint32_t m = M_main; m < M; ++m)
    {
        uint32_t b = m / S, s = m % S;
        for (uint32_t n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            const uint16_t *w_row = W + (uint64_t)n * K;
            const float *x_row = X + (uint64_t)b * S * K + (uint64_t)s * K;
            for (uint32_t k = 0; k < K; ++k)
            {
                sum += x_row[k] * bf16_to_f32(w_row[k]);
            }
            O[(uint64_t)b * S * N + (uint64_t)s * N + n] = sum;
        }
    }

    // N-tail (cols N_main..N-1, rows 0..M_main-1)
    for (uint32_t m = 0; m < M_main; ++m)
    {
        uint32_t b = m / S, s = m % S;
        for (uint32_t n = N_main; n < N; ++n)
        {
            float sum = 0.0f;
            const uint16_t *w_row = W + (uint64_t)n * K;
            const float *x_row = X + (uint64_t)b * S * K + (uint64_t)s * K;
            for (uint32_t k = 0; k < K; ++k)
            {
                sum += x_row[k] * bf16_to_f32(w_row[k]);
            }
            O[(uint64_t)b * S * N + (uint64_t)s * N + n] = sum;
        }
    }

    // K-tail (K_main..K-1, for the M_main x N_main inner tile)
    if (K_main < K)
    {
        for (uint32_t m = 0; m < M_main; ++m)
        {
            uint32_t b = m / S, s = m % S;
            for (uint32_t n = 0; n < N_main; ++n)
            {
                float sum = O[(uint64_t)b * S * N + (uint64_t)s * N + n];
                const uint16_t *w_row = W + (uint64_t)n * K;
                const float *x_row = X + (uint64_t)b * S * K + (uint64_t)s * K;
                for (uint32_t k = K_main; k < K; ++k)
                {
                    sum += x_row[k] * bf16_to_f32(w_row[k]);
                }
                O[(uint64_t)b * S * N + (uint64_t)s * N + n] = sum;
            }
        }
    }
}

inline LogicalId refFactoryBF16TransposedGEMM_v7(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId w_cast = graph.cast(inputs[1], DType::FLOAT32);
    int32_t perm[] = {1, 0};
    LogicalId w_t = graph.contiguous(graph.permute(w_cast, graph.constant({2}, perm, DType::INT32)));
    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, (int32_t)w_shape[1], (int32_t)w_shape[0]};
    return graph.dot(inputs[0], graph.reshape(w_t, graph.constant({3}, s3, DType::INT32)));
}

REGISTER_KERNEL("BF16_Transposed_GEMM_NEON_v7", 2, 2, matchBF16TransposedGEMM_v7, runBF16TransposedGEMM_v7,
                refFactoryBF16TransposedGEMM_v7, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::BF16}, {{1, 256, 512}, {128, 512}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON_TODOREMOVETHISCHECK && __ARM_FEATURE_BF16
