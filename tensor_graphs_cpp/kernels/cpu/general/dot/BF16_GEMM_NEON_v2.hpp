// File: tensor_graphs_cpp/kernels/cpu/general/dot/BF16_GEMM_NEON_v2.hpp
#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

inline bool matchBF16GEMM_NEON_v2(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

inline void runBF16GEMM_NEON_v2(const KernelContext &ctx)
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

    // We parallelize over M slices. Since X is [B, M, K], we can treat B*M as a
    // single dimension
    uint32_t total_M = B * M;
    num_threads = std::min(num_threads, (total_M + 3) / 4);

    std::vector<std::thread> workers;
    uint32_t m_block = (total_M + num_threads - 1) / num_threads;
    m_block = (m_block + 3) & ~3; // Align to 4

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]() {
            uint32_t m_start = t * m_block;
            if (m_start >= total_M)
                return;
            uint32_t m_end = std::min(m_start + m_block, total_M);

            uint32_t n_rem = N & ~15;
            uint32_t m_rem = m_end & ~3;

            // Transpose/pack a vertical slice of W to avoid extreme TLB/cache
            // thrashing in the inner loop.
            std::vector<uint16_t> w_packed(K * 16);

            // Re-order loops: Iterate N -> pack W slice -> iterate M -> iterate K
            // (contiguous memory loads)
            for (uint32_t n = 0; n < n_rem; n += 16)
            {
                // Pack 16 columns of W across all K sequentially to stay in L1 cache
                // bounds
                uint16_t *wp_out = w_packed.data();
                for (uint32_t k = 0; k < K; ++k)
                {
                    const uint16_t *wp_in = W + k * N + n;
                    uint16x8_t w0 = vld1q_u16(wp_in + 0);
                    uint16x8_t w1 = vld1q_u16(wp_in + 8);
                    vst1q_u16(wp_out + 0, w0);
                    vst1q_u16(wp_out + 8, w1);
                    wp_out += 16;
                }

                for (uint32_t m = m_start; m < m_rem; m += 4)
                {
                    float32x4_t c00 = vdupq_n_f32(0), c01 = vdupq_n_f32(0), c02 = vdupq_n_f32(0), c03 = vdupq_n_f32(0);
                    float32x4_t c10 = vdupq_n_f32(0), c11 = vdupq_n_f32(0), c12 = vdupq_n_f32(0), c13 = vdupq_n_f32(0);
                    float32x4_t c20 = vdupq_n_f32(0), c21 = vdupq_n_f32(0), c22 = vdupq_n_f32(0), c23 = vdupq_n_f32(0);
                    float32x4_t c30 = vdupq_n_f32(0), c31 = vdupq_n_f32(0), c32 = vdupq_n_f32(0), c33 = vdupq_n_f32(0);

                    const float *x0_ptr = X + (m + 0) * K;
                    const float *x1_ptr = X + (m + 1) * K;
                    const float *x2_ptr = X + (m + 2) * K;
                    const float *x3_ptr = X + (m + 3) * K;
                    const uint16_t *wp = w_packed.data();

                    for (uint32_t k = 0; k < K; ++k)
                    {
                        // All accesses in inner loop are now contiguous and completely
                        // prefetchable
                        float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(wp + 0), 16));
                        float32x4_t w1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(wp + 4), 16));
                        float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(wp + 8), 16));
                        float32x4_t w3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(wp + 12), 16));
                        wp += 16;

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

                    float *out_ptr0 = Out + (m + 0) * N + n;
                    float *out_ptr1 = Out + (m + 1) * N + n;
                    float *out_ptr2 = Out + (m + 2) * N + n;
                    float *out_ptr3 = Out + (m + 3) * N + n;

                    vst1q_f32(out_ptr0 + 0, c00);
                    vst1q_f32(out_ptr0 + 4, c01);
                    vst1q_f32(out_ptr0 + 8, c02);
                    vst1q_f32(out_ptr0 + 12, c03);

                    vst1q_f32(out_ptr1 + 0, c10);
                    vst1q_f32(out_ptr1 + 4, c11);
                    vst1q_f32(out_ptr1 + 8, c12);
                    vst1q_f32(out_ptr1 + 12, c13);

                    vst1q_f32(out_ptr2 + 0, c20);
                    vst1q_f32(out_ptr2 + 4, c21);
                    vst1q_f32(out_ptr2 + 8, c22);
                    vst1q_f32(out_ptr2 + 12, c23);

                    vst1q_f32(out_ptr3 + 0, c30);
                    vst1q_f32(out_ptr3 + 4, c31);
                    vst1q_f32(out_ptr3 + 8, c32);
                    vst1q_f32(out_ptr3 + 12, c33);
                }
            }

            // Handle remaining N for main M block (unlikely required given shape
            // dimensions)
            for (uint32_t m = m_start; m < m_rem; m += 4)
            {
                for (uint32_t n = n_rem; n < N; ++n)
                {
                    float sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        uint32_t bits = (uint32_t)W[k * N + n] << 16;
                        float wf;
                        std::memcpy(&wf, &bits, 4);
                        sum0 += X[(m + 0) * K + k] * wf;
                        sum1 += X[(m + 1) * K + k] * wf;
                        sum2 += X[(m + 2) * K + k] * wf;
                        sum3 += X[(m + 3) * K + k] * wf;
                    }
                    Out[(m + 0) * N + n] = sum0;
                    Out[(m + 1) * N + n] = sum1;
                    Out[(m + 2) * N + n] = sum2;
                    Out[(m + 3) * N + n] = sum3;
                }
            }

            // Handle remaining M
            for (uint32_t m = m_rem; m < m_end; ++m)
            {
                for (uint32_t n = 0; n < n_rem; n += 4)
                {
                    float32x4_t c0 = vdupq_n_f32(0);
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(W + k * N + n), 16));
                        float32x4_t x0 = vdupq_n_f32(X[m * K + k]);
                        c0 = vfmaq_f32(c0, x0, w0);
                    }
                    vst1q_f32(Out + m * N + n, c0);
                }
                for (uint32_t n = n_rem; n < N; ++n)
                {
                    float sum0 = 0;
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        uint32_t bits = (uint32_t)W[k * N + n] << 16;
                        float wf;
                        std::memcpy(&wf, &bits, 4);
                        sum0 += X[m * K + k] * wf;
                    }
                    Out[m * N + n] = sum0;
                }
            }
        });
    }
    for (auto &worker : workers)
        worker.join();
}

inline LogicalId refFactoryBF16GEMM_NEON_v2(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // inputs[0] is X [B, M, K] (F32)
    // inputs[1] is W [1, K, N] (BF16)
    LogicalId w_f32 = graph.cast(inputs[1], DType::FLOAT32);
    return graph.dot(inputs[0], w_f32);
}

REGISTER_KERNEL("BF16_GEMM_NEON_v2", 2, 2, matchBF16GEMM_NEON_v2, runBF16GEMM_NEON_v2, refFactoryBF16GEMM_NEON_v2,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::BF16},
                {{1, 8, 64}, {1, 64, 1024}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif