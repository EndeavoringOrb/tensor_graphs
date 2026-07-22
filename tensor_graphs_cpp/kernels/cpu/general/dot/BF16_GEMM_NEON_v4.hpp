// File: tensor_graphs_cpp/kernels/cpu/general/dot/BF16_GEMM_NEON_v4.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

#if defined(TG_HAS_NEON_TODOREMOVETHISCHECK)
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>

inline bool matchBF16GEMM_NEON_v4(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    auto sX = inputs[0].getShape();
    auto sW = inputs[1].getShape();
    auto sO = output.getShape();
    if (sX.size() != 3 || sW.size() != 3 || sO.size() != 3)
        return false;
    if (sW[0] != 1)
        return false;
    if (sX[2] != sW[1] || sO[2] != sW[2])
        return false;
    return isContiguous(output);
}

inline void runBF16GEMM_NEON_v4(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const uint16_t *W = static_cast<const uint16_t *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t B = ctx.inViews[0].getShape()[0];
    const uint32_t M_in = ctx.inViews[0].getShape()[1];
    const uint32_t K = ctx.inViews[0].getShape()[2];
    const uint32_t N = ctx.inViews[1].getShape()[2];
    const uint32_t M = B * M_in;

    constexpr uint32_t MR = 8;
    constexpr uint32_t NR = 8;
    constexpr uint32_t KR = 4;

    const uint32_t M_main = (M / MR) * MR;
    const uint32_t N_main = (N / NR) * NR;
    const uint32_t K_main = (K / KR) * KR;

    auto scalar_compute = [&](uint32_t m_s, uint32_t m_e, uint32_t n_s, uint32_t n_e,
                              uint32_t k_s, uint32_t k_e, bool accumulate) {
        for (uint32_t m = m_s; m < m_e; ++m) {
            for (uint32_t n = n_s; n < n_e; ++n) {
                float sum = accumulate ? Out[m * N + n] : 0.0f;
                for (uint32_t k = k_s; k < k_e; ++k) {
                    uint32_t bits = (uint32_t)W[k * N + n] << 16;
                    float wf; std::memcpy(&wf, &bits, 4);
                    sum += X[m * K + k] * wf;
                }
                Out[m * N + n] = sum;
            }
        }
    };

    if (M_main == 0 || N_main == 0 || K_main == 0) {
        scalar_compute(0, M, 0, N, 0, K, false);
        return;
    }

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;

    const uint32_t M_panels = M_main / MR;
    const uint32_t N_panels = N_main / NR;
    const uint32_t K_quads  = K_main / KR;

    // ====== Pack X into BFMMLA-friendly layout (once) ======
    // For each M-panel (8 rows): [K_quads][4 a-vecs][8 bf16]
    //   a-vec[ai] = [X(m+2ai,k..k+3), X(m+2ai+1,k..k+3)]   (truncate f32->bf16)
    std::vector<uint16_t> X_packed((uint64_t)M_main * K_main);
    {
        uint32_t nt = std::min(num_threads, M_panels);
        std::vector<std::thread> packers;
        uint32_t per = (M_panels + nt - 1) / nt;
        for (uint32_t t = 0; t < nt; ++t) {
            uint32_t s = t * per;
            uint32_t e = std::min(s + per, M_panels);
            if (s >= e) break;
            packers.emplace_back([=, &X_packed]() {
                for (uint32_t mp = s; mp < e; ++mp) {
                    uint16_t* dst = X_packed.data() + (uint64_t)mp * MR * K_main;
                    uint32_t m0 = mp * MR;
                    for (uint32_t kq = 0; kq < K_quads; ++kq) {
                        uint32_t k = kq * KR;
                        for (uint32_t ai = 0; ai < MR / 2; ++ai) {
                            float32x4_t r0 = vld1q_f32(X + (uint64_t)(m0 + ai*2)     * K + k);
                            float32x4_t r1 = vld1q_f32(X + (uint64_t)(m0 + ai*2 + 1) * K + k);
                            uint16x4_t b0 = vshrn_n_u32(vreinterpretq_u32_f32(r0), 16);
                            uint16x4_t b1 = vshrn_n_u32(vreinterpretq_u32_f32(r1), 16);
                            vst1q_u16(dst, vcombine_u16(b0, b1));
                            dst += 8;
                        }
                    }
                }
            });
        }
        for (auto& p : packers) p.join();
    }

    // ====== Compute (parallel over N-panels) ======
    // Each thread keeps the current 8-col W panel hot in L1, then sweeps all M.
    {
        uint32_t nt = std::min(num_threads, N_panels);
        std::vector<std::thread> workers;
        uint32_t per = (N_panels + nt - 1) / nt;

        for (uint32_t t = 0; t < nt; ++t) {
            uint32_t s = t * per;
            uint32_t e = std::min(s + per, N_panels);
            if (s >= e) break;
            workers.emplace_back([=, &X_packed]() {
                std::vector<uint16_t> W_panel((uint64_t)K_main * NR); // ~48 KB, fits L1

                for (uint32_t np = s; np < e; ++np) {
                    uint32_t n0 = np * NR;

                    // Pack W panel: [K_quads][4 b-vecs][8 bf16]
                    // b-vec[bi] = [W(k..k+3, n+2bi), W(k..k+3, n+2bi+1)]
                    uint16_t* wp = W_panel.data();
                    for (uint32_t kq = 0; kq < K_quads; ++kq) {
                        uint32_t k = kq * KR;
                        for (uint32_t bi = 0; bi < NR / 2; ++bi) {
                            uint32_t n = n0 + bi * 2;
                            wp[0] = W[(uint64_t)k     * N + n];
                            wp[1] = W[(uint64_t)(k+1) * N + n];
                            wp[2] = W[(uint64_t)(k+2) * N + n];
                            wp[3] = W[(uint64_t)(k+3) * N + n];
                            wp[4] = W[(uint64_t)k     * N + n + 1];
                            wp[5] = W[(uint64_t)(k+1) * N + n + 1];
                            wp[6] = W[(uint64_t)(k+2) * N + n + 1];
                            wp[7] = W[(uint64_t)(k+3) * N + n + 1];
                            wp += 8;
                        }
                    }

                    for (uint32_t mp = 0; mp < M_panels; ++mp) {
                        uint32_t m_base = mp * MR;
                        const uint16_t* A_ptr = X_packed.data() + (uint64_t)mp * MR * K_main;
                        const uint16_t* B_ptr = W_panel.data();

                        // 16 fp32x4 accumulators -> 8x8 output tile (each holds 2x2 chunk)
                        float32x4_t c00=vdupq_n_f32(0),c01=vdupq_n_f32(0),c02=vdupq_n_f32(0),c03=vdupq_n_f32(0);
                        float32x4_t c10=vdupq_n_f32(0),c11=vdupq_n_f32(0),c12=vdupq_n_f32(0),c13=vdupq_n_f32(0);
                        float32x4_t c20=vdupq_n_f32(0),c21=vdupq_n_f32(0),c22=vdupq_n_f32(0),c23=vdupq_n_f32(0);
                        float32x4_t c30=vdupq_n_f32(0),c31=vdupq_n_f32(0),c32=vdupq_n_f32(0),c33=vdupq_n_f32(0);

                        for (uint32_t kq = 0; kq < K_quads; ++kq) {
                            __builtin_prefetch(A_ptr + 128);
                            __builtin_prefetch(B_ptr + 128);

                            bfloat16x8_t a0 = vreinterpretq_bf16_u16(vld1q_u16(A_ptr));
                            bfloat16x8_t a1 = vreinterpretq_bf16_u16(vld1q_u16(A_ptr + 8));
                            bfloat16x8_t a2 = vreinterpretq_bf16_u16(vld1q_u16(A_ptr + 16));
                            bfloat16x8_t a3 = vreinterpretq_bf16_u16(vld1q_u16(A_ptr + 24));

                            bfloat16x8_t b0 = vreinterpretq_bf16_u16(vld1q_u16(B_ptr));
                            bfloat16x8_t b1 = vreinterpretq_bf16_u16(vld1q_u16(B_ptr + 8));
                            bfloat16x8_t b2 = vreinterpretq_bf16_u16(vld1q_u16(B_ptr + 16));
                            bfloat16x8_t b3 = vreinterpretq_bf16_u16(vld1q_u16(B_ptr + 24));

                            c00 = vbfmmlaq_f32(c00, a0, b0); c01 = vbfmmlaq_f32(c01, a0, b1);
                            c02 = vbfmmlaq_f32(c02, a0, b2); c03 = vbfmmlaq_f32(c03, a0, b3);
                            c10 = vbfmmlaq_f32(c10, a1, b0); c11 = vbfmmlaq_f32(c11, a1, b1);
                            c12 = vbfmmlaq_f32(c12, a1, b2); c13 = vbfmmlaq_f32(c13, a1, b3);
                            c20 = vbfmmlaq_f32(c20, a2, b0); c21 = vbfmmlaq_f32(c21, a2, b1);
                            c22 = vbfmmlaq_f32(c22, a2, b2); c23 = vbfmmlaq_f32(c23, a2, b3);
                            c30 = vbfmmlaq_f32(c30, a3, b0); c31 = vbfmmlaq_f32(c31, a3, b1);
                            c32 = vbfmmlaq_f32(c32, a3, b2); c33 = vbfmmlaq_f32(c33, a3, b3);

                            A_ptr += 32;
                            B_ptr += 32;
                        }

                        // Each cXY holds [C[m,n], C[m,n+1], C[m+1,n], C[m+1,n+1]]
                        // Reassemble into row-major fp32x4 stores.
                        auto store_pair = [&](uint32_t mi, float32x4_t v0, float32x4_t v1,
                                                          float32x4_t v2, float32x4_t v3) {
                            uint32_t r0 = m_base + mi * 2;
                            float32x4_t row0_a = vcombine_f32(vget_low_f32(v0),  vget_low_f32(v1));
                            float32x4_t row0_b = vcombine_f32(vget_low_f32(v2),  vget_low_f32(v3));
                            float32x4_t row1_a = vcombine_f32(vget_high_f32(v0), vget_high_f32(v1));
                            float32x4_t row1_b = vcombine_f32(vget_high_f32(v2), vget_high_f32(v3));
                            vst1q_f32(Out + (uint64_t)r0     * N + n0,     row0_a);
                            vst1q_f32(Out + (uint64_t)r0     * N + n0 + 4, row0_b);
                            vst1q_f32(Out + (uint64_t)(r0+1) * N + n0,     row1_a);
                            vst1q_f32(Out + (uint64_t)(r0+1) * N + n0 + 4, row1_b);
                        };

                        store_pair(0, c00, c01, c02, c03);
                        store_pair(1, c10, c11, c12, c13);
                        store_pair(2, c20, c21, c22, c23);
                        store_pair(3, c30, c31, c32, c33);
                    }
                }
            });
        }
        for (auto& w : workers) w.join();
    }

    // Tails (no-op for the benchmark shape, all divisible by 8/8/4)
    if (K_main < K)              scalar_compute(0, M_main, 0, N_main, K_main, K, true);
    if (M_main < M)              scalar_compute(M_main, M, 0, N, 0, K, false);
    if (N_main < N)              scalar_compute(0, M_main, N_main, N, 0, K, false);
}

inline LogicalId refFactoryBF16GEMM_NEON_v4(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId w_f32 = graph.cast(inputs[1], DType::FLOAT32);
    return graph.dot(inputs[0], w_f32);
}

REGISTER_KERNEL("BF16_GEMM_NEON_v4", 2, 2, matchBF16GEMM_NEON_v4, runBF16GEMM_NEON_v4, refFactoryBF16GEMM_NEON_v4, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::BF16}, {{1, 8, 64}, {1, 64, 1024}}, {true, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif