// File: tensor_graphs_cpp/kernels/cpu/general/dot/BF16_GEMM_NEON_v3.hpp
#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

inline bool matchBF16GEMM_NEON_v3(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

inline void runBF16GEMM_NEON_v3(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const uint16_t *W = static_cast<const uint16_t *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t B = ctx.inViews[0].getShape()[0];
    const uint32_t M = ctx.inViews[0].getShape()[1];
    const uint32_t K = ctx.inViews[0].getShape()[2];
    const uint32_t N = ctx.inViews[1].getShape()[2];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    uint32_t total_M = B * M;
    num_threads = std::min(num_threads, (total_M + 3) / 4);

    std::vector<std::thread> workers;
    uint32_t m_block = (total_M + num_threads - 1) / num_threads;
    m_block = (m_block + 3) & ~3; // Align to 4

    // L2 Cache blocking dimension for N. 256 columns of W (K=3072) is
    // exactly 1.57 MB, fitting perfectly inside L2/SLC cache alongside X.
    const uint32_t N_BLOCK = 256;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]() {
            uint32_t m_start = t * m_block;
            if (m_start >= total_M)
                return;
            uint32_t m_end = std::min(m_start + m_block, total_M);
            uint32_t m_rem = m_end & ~3;

            // Thread-local panel buffer for W. Allocated once per thread to avoid OS
            // mmap overhead. Layout is packed as [N_BLOCK/16, K, 16] to enable 100%
            // contiguous SIMD access in the inner loop.
            std::vector<uint16_t> W_panel(K * N_BLOCK);

            for (uint32_t n_outer = 0; n_outer < N; n_outer += N_BLOCK)
            {
                uint32_t n_curr_block = std::min(N_BLOCK, N - n_outer);
                uint32_t n_curr_rem = n_curr_block & ~15;

                // 1. On-the-fly panel packing.
                // Outer loop K, inner loop N_chunk ensures W is read sequentially along
                // rows. Eliminates 55KB stride jumps, achieving 100% cache line
                // utilization and 0 TLB misses.
                if (n_curr_rem > 0)
                {
                    uint16_t *panel_base = W_panel.data();
                    uint32_t K_16 = K * 16;
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        const uint16_t *w_row = W + k * N + n_outer;
                        uint16_t *dst_row = panel_base + k * 16;
                        for (uint32_t n_chunk = 0; n_chunk < n_curr_rem; n_chunk += 16)
                        {
                            uint16x8_t w_lo = vld1q_u16(w_row + n_chunk);
                            uint16x8_t w_hi = vld1q_u16(w_row + n_chunk + 8);
                            vst1q_u16(dst_row, w_lo);
                            vst1q_u16(dst_row + 8, w_hi);
                            dst_row += K_16;
                        }
                    }
                }

                // 2. Main GEMM Micro-Kernel for 4x16 tiles
                for (uint32_t m = m_start; m < m_rem; m += 4)
                {
                    for (uint32_t n_chunk = 0; n_chunk < n_curr_rem; n_chunk += 16)
                    {
                        float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0), c02 = vdupq_n_f32(0),
                                    c03 = vdupq_n_f32(0);
                        float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0), c12 = vdupq_n_f32(0),
                                    c13 = vdupq_n_f32(0);
                        float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0), c22 = vdupq_n_f32(0),
                                    c23 = vdupq_n_f32(0);
                        float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0), c32 = vdupq_n_f32(0),
                                    c33 = vdupq_n_f32(0);

                        const uint16_t *w_ptr = W_panel.data() + (n_chunk / 16) * K * 16;
                        const float *x0_ptr = X + (m + 0) * K;
                        const float *x1_ptr = X + (m + 1) * K;
                        const float *x2_ptr = X + (m + 2) * K;
                        const float *x3_ptr = X + (m + 3) * K;

                        uint32_t k_rem = K & ~1;
                        for (uint32_t k = 0; k < k_rem; k += 2)
                        {
                            // Unroll k by 2. Consecutive vld1q_u16 fuse into AArch64 'ldp'
                            // (load pair).
                            uint16x8_t w_q0_0 = vld1q_u16(w_ptr + 0);
                            uint16x8_t w_q1_0 = vld1q_u16(w_ptr + 8);

                            float32x4_t w0_0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w_q0_0), 16));
                            float32x4_t w1_0 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w_q0_0), 16));
                            float32x4_t w2_0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w_q1_0), 16));
                            float32x4_t w3_0 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w_q1_0), 16));

                            float32x4_t x0_0 = vdupq_n_f32(x0_ptr[k]);
                            float32x4_t x1_0 = vdupq_n_f32(x1_ptr[k]);
                            float32x4_t x2_0 = vdupq_n_f32(x2_ptr[k]);
                            float32x4_t x3_0 = vdupq_n_f32(x3_ptr[k]);

                            uint16x8_t w_q0_1 = vld1q_u16(w_ptr + 16);
                            uint16x8_t w_q1_1 = vld1q_u16(w_ptr + 24);
                            w_ptr += 32;

                            float32x4_t w0_1 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w_q0_1), 16));
                            float32x4_t w1_1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w_q0_1), 16));
                            float32x4_t w2_1 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w_q1_1), 16));
                            float32x4_t w3_1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w_q1_1), 16));

                            float32x4_t x0_1 = vdupq_n_f32(x0_ptr[k + 1]);
                            float32x4_t x1_1 = vdupq_n_f32(x1_ptr[k + 1]);
                            float32x4_t x2_1 = vdupq_n_f32(x2_ptr[k + 1]);
                            float32x4_t x3_1 = vdupq_n_f32(x3_ptr[k + 1]);

                            // FMAs for k + 0
                            c00 = vfmaq_f32(c00, x0_0, w0_0);
                            c01 = vfmaq_f32(c01, x0_0, w1_0);
                            c02 = vfmaq_f32(c02, x0_0, w2_0);
                            c03 = vfmaq_f32(c03, x0_0, w3_0);

                            c10 = vfmaq_f32(c10, x1_0, w0_0);
                            c11 = vfmaq_f32(c11, x1_0, w1_0);
                            c12 = vfmaq_f32(c12, x1_0, w2_0);
                            c13 = vfmaq_f32(c13, x1_0, w3_0);

                            c20 = vfmaq_f32(c20, x2_0, w0_0);
                            c21 = vfmaq_f32(c21, x2_0, w1_0);
                            c22 = vfmaq_f32(c22, x2_0, w2_0);
                            c23 = vfmaq_f32(c23, x2_0, w3_0);

                            c30 = vfmaq_f32(c30, x3_0, w0_0);
                            c31 = vfmaq_f32(c31, x3_0, w1_0);
                            c32 = vfmaq_f32(c32, x3_0, w2_0);
                            c33 = vfmaq_f32(c33, x3_0, w3_0);

                            // FMAs for k + 1
                            c00 = vfmaq_f32(c00, x0_1, w0_1);
                            c01 = vfmaq_f32(c01, x0_1, w1_1);
                            c02 = vfmaq_f32(c02, x0_1, w2_1);
                            c03 = vfmaq_f32(c03, x0_1, w3_1);

                            c10 = vfmaq_f32(c10, x1_1, w0_1);
                            c11 = vfmaq_f32(c11, x1_1, w1_1);
                            c12 = vfmaq_f32(c12, x1_1, w2_1);
                            c13 = vfmaq_f32(c13, x1_1, w3_1);

                            c20 = vfmaq_f32(c20, x2_1, w0_1);
                            c21 = vfmaq_f32(c21, x2_1, w1_1);
                            c22 = vfmaq_f32(c22, x2_1, w2_1);
                            c23 = vfmaq_f32(c23, x2_1, w3_1);

                            c30 = vfmaq_f32(c30, x3_1, w0_1);
                            c31 = vfmaq_f32(c31, x3_1, w1_1);
                            c32 = vfmaq_f32(c32, x3_1, w2_1);
                            c33 = vfmaq_f32(c33, x3_1, w3_1);
                        }

                        // Handle odd K remainder if any
                        for (uint32_t k = k_rem; k < K; ++k)
                        {
                            uint16x8_t w_q0 = vld1q_u16(w_ptr + 0);
                            uint16x8_t w_q1 = vld1q_u16(w_ptr + 8);
                            w_ptr += 16;

                            float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w_q0), 16));
                            float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w_q0), 16));
                            float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w_q1), 16));
                            float32x4_t w3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w_q1), 16));

                            float32x4_t x0 = vdupq_n_f32(x0_ptr[k]);
                            float32x4_t x1 = vdupq_n_f32(x1_ptr[k]);
                            float32x4_t x2 = vdupq_n_f32(x2_ptr[k]);
                            float32x4_t x3 = vdupq_n_f32(x3_ptr[k]);

                            c00 = vfmaq_f32(c00, x0, w0);
                            c01 = vfmaq_f32(c01, x0, w1);
                            c02 = vfmaq_f32(c02, x0, w2);
                            c03 = vfmaq_f32(c03, x0, w3);

                            c10 = vfmaq_f32(c10, x1, w0);
                            c11 = vfmaq_f32(c11, x1, w1);
                            c12 = vfmaq_f32(c12, x1, w2);
                            c13 = vfmaq_f32(c13, x1, w3);

                            c20 = vfmaq_f32(c20, x2, w0);
                            c21 = vfmaq_f32(c21, x2, w1);
                            c22 = vfmaq_f32(c22, x2, w2);
                            c23 = vfmaq_f32(c23, x2, w3);

                            c30 = vfmaq_f32(c30, x3, w0);
                            c31 = vfmaq_f32(c31, x3, w1);
                            c32 = vfmaq_f32(c32, x3, w2);
                            c33 = vfmaq_f32(c33, x3, w3);
                        }

                        uint32_t n_actual = n_outer + n_chunk;
                        float *out_ptr = Out + m * N + n_actual;
                        vst1q_f32(out_ptr + 0 * N + 0, c00);
                        vst1q_f32(out_ptr + 0 * N + 4, c01);
                        vst1q_f32(out_ptr + 0 * N + 8, c02);
                        vst1q_f32(out_ptr + 0 * N + 12, c03);

                        vst1q_f32(out_ptr + 1 * N + 0, c10);
                        vst1q_f32(out_ptr + 1 * N + 4, c11);
                        vst1q_f32(out_ptr + 1 * N + 8, c12);
                        vst1q_f32(out_ptr + 1 * N + 12, c13);

                        vst1q_f32(out_ptr + 2 * N + 0, c20);
                        vst1q_f32(out_ptr + 2 * N + 4, c21);
                        vst1q_f32(out_ptr + 2 * N + 8, c22);
                        vst1q_f32(out_ptr + 2 * N + 12, c23);

                        vst1q_f32(out_ptr + 3 * N + 0, c30);
                        vst1q_f32(out_ptr + 3 * N + 4, c31);
                        vst1q_f32(out_ptr + 3 * N + 8, c32);
                        vst1q_f32(out_ptr + 3 * N + 12, c33);
                    }

                    // Handle remaining N columns in the current block (if N_BLOCK is not
                    // a multiple of 16)
                    for (uint32_t n_chunk = n_curr_rem; n_chunk < n_curr_block; ++n_chunk)
                    {
                        uint32_t n_actual = n_outer + n_chunk;
                        float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                        for (uint32_t k = 0; k < K; ++k)
                        {
                            uint32_t bits = (uint32_t)W[k * N + n_actual] << 16;
                            float wf;
                            std::memcpy(&wf, &bits, 4);
                            sum0 += X[(m + 0) * K + k] * wf;
                            sum1 += X[(m + 1) * K + k] * wf;
                            sum2 += X[(m + 2) * K + k] * wf;
                            sum3 += X[(m + 3) * K + k] * wf;
                        }
                        Out[(m + 0) * N + n_actual] = sum0;
                        Out[(m + 1) * N + n_actual] = sum1;
                        Out[(m + 2) * N + n_actual] = sum2;
                        Out[(m + 3) * N + n_actual] = sum3;
                    }
                }

                // Handle remaining M rows for the current N block
                for (uint32_t m = m_rem; m < m_end; ++m)
                {
                    for (uint32_t n_chunk = 0; n_chunk < n_curr_rem; n_chunk += 4)
                    {
                        float32x4_t c0 = vdupq_n_f32(0);
                        uint32_t n_actual = n_outer + n_chunk;
                        for (uint32_t k = 0; k < K; ++k)
                        {
                            float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W + k * N + n_actual), 16));
                            float32x4_t x0 = vdupq_n_f32(X[m * K + k]);
                            c0 = vfmaq_f32(c0, x0, w0);
                        }
                        vst1q_f32(Out + m * N + n_actual, c0);
                    }
                    for (uint32_t n_chunk = n_curr_rem; n_chunk < n_curr_block; ++n_chunk)
                    {
                        uint32_t n_actual = n_outer + n_chunk;
                        float sum0 = 0;
                        for (uint32_t k = 0; k < K; ++k)
                        {
                            uint32_t bits = (uint32_t)W[k * N + n_actual] << 16;
                            float wf;
                            std::memcpy(&wf, &bits, 4);
                            sum0 += X[m * K + k] * wf;
                        }
                        Out[m * N + n_actual] = sum0;
                    }
                }
            }
        });
    }
    for (auto &worker : workers)
        worker.join();
}

inline LogicalId refFactoryBF16GEMM_NEON_v3(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId w_f32 = graph.cast(inputs[1], DType::FLOAT32);
    return graph.dot(inputs[0], w_f32);
}

REGISTER_KERNEL("BF16_GEMM_NEON_v3", 2, 2, matchBF16GEMM_NEON_v3, runBF16GEMM_NEON_v3, refFactoryBF16GEMM_NEON_v3,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::BF16},
                {{1, 8, 64}, {1, 64, 1024}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif