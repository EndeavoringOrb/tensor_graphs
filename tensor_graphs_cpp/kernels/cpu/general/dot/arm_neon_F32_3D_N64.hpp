#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "core/graph.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>

inline bool matchDotF32_3D_N64(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &s0 = inputs[0].getShape();
    const auto &s1 = inputs[1].getShape();
    const auto &so = output.getShape();

    if (s0.size() != 3 || s1.size() != 3 || so.size() != 3)
        return false;
    if (s0[0] != s1[0] || s0[2] != s1[1])
        return false;
    if (so[0] != s0[0] || so[1] != s0[1] || so[2] != s1[2])
        return false;

    // Specifically target head dimension N = 64
    if (so[2] != 64)
        return false;

    // Both B and Out must have stride-1 on the last dimension for contiguous vector loads
    if (inputs[1].strides[2] != 1 || output.strides[2] != 1)
        return false;

    return true;
}

inline void runDotF32_3D_N64(const KernelContext &ctx)
{
    const float *A_ptr = static_cast<const float *>(ctx.inputs[0]);
    const float *B_ptr = static_cast<const float *>(ctx.inputs[1]);
    float *Out_ptr = static_cast<float *>(ctx.outputs[0]);

    const auto &viewA = ctx.inViews[0];
    const auto &viewB = ctx.inViews[1];
    const auto &viewOut = ctx.outViews[0];

    const uint32_t B_count = viewA.getShape()[0];
    const uint32_t M = viewA.getShape()[1];
    const uint32_t K = viewA.getShape()[2];

    const int64_t strideA_B = viewA.strides[0];
    const int64_t strideA_M = viewA.strides[1];
    const int64_t strideA_K = viewA.strides[2];

    const int64_t strideB_B = viewB.strides[0];
    const int64_t strideB_K = viewB.strides[1];

    const int64_t strideO_B = viewOut.strides[0];
    const int64_t strideO_M = viewOut.strides[1];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    std::vector<std::thread> workers;

    uint32_t total_rows = B_count * M;
    uint32_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t start_row = t * rows_per_thread;
            uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);
            if (start_row >= end_row) return;

            uint32_t b_start = start_row / M;
            uint32_t b_end = (end_row - 1) / M;

            // K_BLK = 128 perfectly sizes B block (32 KB) + A block (30 KB) + C block (15 KB)
            // protecting L1d cache (96 KB) from sweeps leading to conflict evictions.
            const uint32_t K_BLK = 128; 
            const uint32_t M_BLK = 60; // Multiple of 6 ensures unrolled M-loop avoids tail logic entirely

            for (uint32_t b = b_start; b <= b_end; ++b) {
                uint32_t m_start_b = (b == b_start) ? (start_row % M) : 0;
                uint32_t m_end_b = (b == b_end) ? ((end_row - 1) % M + 1) : M;

                const float* a_ptr_base = A_ptr + b * strideA_B;
                const float* b_ptr_base = B_ptr + b * strideB_B;
                float* o_ptr_base = Out_ptr + b * strideO_B;

                for (uint32_t m_blk_start = m_start_b; m_blk_start < m_end_b; m_blk_start += M_BLK) {
                    uint32_t m_blk_end = std::min(m_blk_start + M_BLK, m_end_b);

                    for (uint32_t k_start = 0; k_start < K; k_start += K_BLK) {
                        uint32_t k_end = std::min(k_start + K_BLK, K);
                        bool is_first_k = (k_start == 0);

                        uint32_t m = m_blk_start;
                        for (; m + 5 < m_blk_end; m += 6) {
                            const float* a0_ptr = a_ptr_base + (m + 0) * strideA_M;
                            const float* a1_ptr = a_ptr_base + (m + 1) * strideA_M;
                            const float* a2_ptr = a_ptr_base + (m + 2) * strideA_M;
                            const float* a3_ptr = a_ptr_base + (m + 3) * strideA_M;
                            const float* a4_ptr = a_ptr_base + (m + 4) * strideA_M;
                            const float* a5_ptr = a_ptr_base + (m + 5) * strideA_M;

                            float* o0_ptr = o_ptr_base + (m + 0) * strideO_M;
                            float* o1_ptr = o_ptr_base + (m + 1) * strideO_M;
                            float* o2_ptr = o_ptr_base + (m + 2) * strideO_M;
                            float* o3_ptr = o_ptr_base + (m + 3) * strideO_M;
                            float* o4_ptr = o_ptr_base + (m + 4) * strideO_M;
                            float* o5_ptr = o_ptr_base + (m + 5) * strideO_M;

                            for (uint32_t n = 0; n < 64; n += 16) {
                                float32x4_t c00, c01, c02, c03;
                                float32x4_t c10, c11, c12, c13;
                                float32x4_t c20, c21, c22, c23;
                                float32x4_t c30, c31, c32, c33;
                                float32x4_t c40, c41, c42, c43;
                                float32x4_t c50, c51, c52, c53;

                                if (is_first_k) {
                                    c00 = vdupq_n_f32(0); c01 = vdupq_n_f32(0); c02 = vdupq_n_f32(0); c03 = vdupq_n_f32(0);
                                    c10 = vdupq_n_f32(0); c11 = vdupq_n_f32(0); c12 = vdupq_n_f32(0); c13 = vdupq_n_f32(0);
                                    c20 = vdupq_n_f32(0); c21 = vdupq_n_f32(0); c22 = vdupq_n_f32(0); c23 = vdupq_n_f32(0);
                                    c30 = vdupq_n_f32(0); c31 = vdupq_n_f32(0); c32 = vdupq_n_f32(0); c33 = vdupq_n_f32(0);
                                    c40 = vdupq_n_f32(0); c41 = vdupq_n_f32(0); c42 = vdupq_n_f32(0); c43 = vdupq_n_f32(0);
                                    c50 = vdupq_n_f32(0); c51 = vdupq_n_f32(0); c52 = vdupq_n_f32(0); c53 = vdupq_n_f32(0);
                                } else {
                                    c00 = vld1q_f32(o0_ptr + n + 0); c01 = vld1q_f32(o0_ptr + n + 4); c02 = vld1q_f32(o0_ptr + n + 8); c03 = vld1q_f32(o0_ptr + n + 12);
                                    c10 = vld1q_f32(o1_ptr + n + 0); c11 = vld1q_f32(o1_ptr + n + 4); c12 = vld1q_f32(o1_ptr + n + 8); c13 = vld1q_f32(o1_ptr + n + 12);
                                    c20 = vld1q_f32(o2_ptr + n + 0); c21 = vld1q_f32(o2_ptr + n + 4); c22 = vld1q_f32(o2_ptr + n + 8); c23 = vld1q_f32(o2_ptr + n + 12);
                                    c30 = vld1q_f32(o3_ptr + n + 0); c31 = vld1q_f32(o3_ptr + n + 4); c32 = vld1q_f32(o3_ptr + n + 8); c33 = vld1q_f32(o3_ptr + n + 12);
                                    c40 = vld1q_f32(o4_ptr + n + 0); c41 = vld1q_f32(o4_ptr + n + 4); c42 = vld1q_f32(o4_ptr + n + 8); c43 = vld1q_f32(o4_ptr + n + 12);
                                    c50 = vld1q_f32(o5_ptr + n + 0); c51 = vld1q_f32(o5_ptr + n + 4); c52 = vld1q_f32(o5_ptr + n + 8); c53 = vld1q_f32(o5_ptr + n + 12);
                                }

                                const float* p_a0 = a0_ptr + k_start * strideA_K;
                                const float* p_a1 = a1_ptr + k_start * strideA_K;
                                const float* p_a2 = a2_ptr + k_start * strideA_K;
                                const float* p_a3 = a3_ptr + k_start * strideA_K;
                                const float* p_a4 = a4_ptr + k_start * strideA_K;
                                const float* p_a5 = a5_ptr + k_start * strideA_K;
                                const float* p_b = b_ptr_base + k_start * strideB_K + n;

                                uint32_t k = k_start;
                                for (; k + 1 < k_end; k += 2) {
                                    // Step 1
                                    float32x4_t b0 = vld1q_f32(p_b + 0);
                                    float32x4_t b1 = vld1q_f32(p_b + 4);
                                    float32x4_t b2 = vld1q_f32(p_b + 8);
                                    float32x4_t b3 = vld1q_f32(p_b + 12);

                                    float32x4_t a0 = vdupq_n_f32(*p_a0);
                                    c00 = vfmaq_f32(c00, b0, a0); c01 = vfmaq_f32(c01, b1, a0); c02 = vfmaq_f32(c02, b2, a0); c03 = vfmaq_f32(c03, b3, a0);

                                    float32x4_t a1 = vdupq_n_f32(*p_a1);
                                    c10 = vfmaq_f32(c10, b0, a1); c11 = vfmaq_f32(c11, b1, a1); c12 = vfmaq_f32(c12, b2, a1); c13 = vfmaq_f32(c13, b3, a1);

                                    float32x4_t a2 = vdupq_n_f32(*p_a2);
                                    c20 = vfmaq_f32(c20, b0, a2); c21 = vfmaq_f32(c21, b1, a2); c22 = vfmaq_f32(c22, b2, a2); c23 = vfmaq_f32(c23, b3, a2);

                                    float32x4_t a3 = vdupq_n_f32(*p_a3);
                                    c30 = vfmaq_f32(c30, b0, a3); c31 = vfmaq_f32(c31, b1, a3); c32 = vfmaq_f32(c32, b2, a3); c33 = vfmaq_f32(c33, b3, a3);

                                    float32x4_t a4 = vdupq_n_f32(*p_a4);
                                    c40 = vfmaq_f32(c40, b0, a4); c41 = vfmaq_f32(c41, b1, a4); c42 = vfmaq_f32(c42, b2, a4); c43 = vfmaq_f32(c43, b3, a4);

                                    float32x4_t a5 = vdupq_n_f32(*p_a5);
                                    c50 = vfmaq_f32(c50, b0, a5); c51 = vfmaq_f32(c51, b1, a5); c52 = vfmaq_f32(c52, b2, a5); c53 = vfmaq_f32(c53, b3, a5);

                                    p_b += strideB_K;
                                    p_a0 += strideA_K; p_a1 += strideA_K; p_a2 += strideA_K; p_a3 += strideA_K; p_a4 += strideA_K; p_a5 += strideA_K;

                                    // Step 2
                                    b0 = vld1q_f32(p_b + 0);
                                    b1 = vld1q_f32(p_b + 4);
                                    b2 = vld1q_f32(p_b + 8);
                                    b3 = vld1q_f32(p_b + 12);

                                    a0 = vdupq_n_f32(*p_a0);
                                    c00 = vfmaq_f32(c00, b0, a0); c01 = vfmaq_f32(c01, b1, a0); c02 = vfmaq_f32(c02, b2, a0); c03 = vfmaq_f32(c03, b3, a0);

                                    a1 = vdupq_n_f32(*p_a1);
                                    c10 = vfmaq_f32(c10, b0, a1); c11 = vfmaq_f32(c11, b1, a1); c12 = vfmaq_f32(c12, b2, a1); c13 = vfmaq_f32(c13, b3, a1);

                                    a2 = vdupq_n_f32(*p_a2);
                                    c20 = vfmaq_f32(c20, b0, a2); c21 = vfmaq_f32(c21, b1, a2); c22 = vfmaq_f32(c22, b2, a2); c23 = vfmaq_f32(c23, b3, a2);

                                    a3 = vdupq_n_f32(*p_a3);
                                    c30 = vfmaq_f32(c30, b0, a3); c31 = vfmaq_f32(c31, b1, a3); c32 = vfmaq_f32(c32, b2, a3); c33 = vfmaq_f32(c33, b3, a3);

                                    a4 = vdupq_n_f32(*p_a4);
                                    c40 = vfmaq_f32(c40, b0, a4); c41 = vfmaq_f32(c41, b1, a4); c42 = vfmaq_f32(c42, b2, a4); c43 = vfmaq_f32(c43, b3, a4);

                                    a5 = vdupq_n_f32(*p_a5);
                                    c50 = vfmaq_f32(c50, b0, a5); c51 = vfmaq_f32(c51, b1, a5); c52 = vfmaq_f32(c52, b2, a5); c53 = vfmaq_f32(c53, b3, a5);

                                    p_b += strideB_K;
                                    p_a0 += strideA_K; p_a1 += strideA_K; p_a2 += strideA_K; p_a3 += strideA_K; p_a4 += strideA_K; p_a5 += strideA_K;
                                }

                                for (; k < k_end; ++k) {
                                    float32x4_t b0 = vld1q_f32(p_b + 0);
                                    float32x4_t b1 = vld1q_f32(p_b + 4);
                                    float32x4_t b2 = vld1q_f32(p_b + 8);
                                    float32x4_t b3 = vld1q_f32(p_b + 12);

                                    float32x4_t a0 = vdupq_n_f32(*p_a0);
                                    c00 = vfmaq_f32(c00, b0, a0); c01 = vfmaq_f32(c01, b1, a0); c02 = vfmaq_f32(c02, b2, a0); c03 = vfmaq_f32(c03, b3, a0);

                                    float32x4_t a1 = vdupq_n_f32(*p_a1);
                                    c10 = vfmaq_f32(c10, b0, a1); c11 = vfmaq_f32(c11, b1, a1); c12 = vfmaq_f32(c12, b2, a1); c13 = vfmaq_f32(c13, b3, a1);

                                    float32x4_t a2 = vdupq_n_f32(*p_a2);
                                    c20 = vfmaq_f32(c20, b0, a2); c21 = vfmaq_f32(c21, b1, a2); c22 = vfmaq_f32(c22, b2, a2); c23 = vfmaq_f32(c23, b3, a2);

                                    float32x4_t a3 = vdupq_n_f32(*p_a3);
                                    c30 = vfmaq_f32(c30, b0, a3); c31 = vfmaq_f32(c31, b1, a3); c32 = vfmaq_f32(c32, b2, a3); c33 = vfmaq_f32(c33, b3, a3);

                                    float32x4_t a4 = vdupq_n_f32(*p_a4);
                                    c40 = vfmaq_f32(c40, b0, a4); c41 = vfmaq_f32(c41, b1, a4); c42 = vfmaq_f32(c42, b2, a4); c43 = vfmaq_f32(c43, b3, a4);

                                    float32x4_t a5 = vdupq_n_f32(*p_a5);
                                    c50 = vfmaq_f32(c50, b0, a5); c51 = vfmaq_f32(c51, b1, a5); c52 = vfmaq_f32(c52, b2, a5); c53 = vfmaq_f32(c53, b3, a5);

                                    p_b += strideB_K;
                                    p_a0 += strideA_K; p_a1 += strideA_K; p_a2 += strideA_K; p_a3 += strideA_K; p_a4 += strideA_K; p_a5 += strideA_K;
                                }

                                vst1q_f32(o0_ptr + n + 0, c00); vst1q_f32(o0_ptr + n + 4, c01); vst1q_f32(o0_ptr + n + 8, c02); vst1q_f32(o0_ptr + n + 12, c03);
                                vst1q_f32(o1_ptr + n + 0, c10); vst1q_f32(o1_ptr + n + 4, c11); vst1q_f32(o1_ptr + n + 8, c12); vst1q_f32(o1_ptr + n + 12, c13);
                                vst1q_f32(o2_ptr + n + 0, c20); vst1q_f32(o2_ptr + n + 4, c21); vst1q_f32(o2_ptr + n + 8, c22); vst1q_f32(o2_ptr + n + 12, c23);
                                vst1q_f32(o3_ptr + n + 0, c30); vst1q_f32(o3_ptr + n + 4, c31); vst1q_f32(o3_ptr + n + 8, c32); vst1q_f32(o3_ptr + n + 12, c33);
                                vst1q_f32(o4_ptr + n + 0, c40); vst1q_f32(o4_ptr + n + 4, c41); vst1q_f32(o4_ptr + n + 8, c42); vst1q_f32(o4_ptr + n + 12, c43);
                                vst1q_f32(o5_ptr + n + 0, c50); vst1q_f32(o5_ptr + n + 4, c51); vst1q_f32(o5_ptr + n + 8, c52); vst1q_f32(o5_ptr + n + 12, c53);
                            }
                        }

                        // M-Tail Logic
                        for (; m < m_blk_end; ++m) {
                            const float* a_ptr = a_ptr_base + m * strideA_M;
                            float* o_ptr = o_ptr_base + m * strideO_M;

                            for (uint32_t n = 0; n < 64; n += 16) {
                                float32x4_t c0, c1, c2, c3;
                                if (is_first_k) {
                                    c0 = vdupq_n_f32(0); c1 = vdupq_n_f32(0); c2 = vdupq_n_f32(0); c3 = vdupq_n_f32(0);
                                } else {
                                    c0 = vld1q_f32(o_ptr + n + 0); c1 = vld1q_f32(o_ptr + n + 4); 
                                    c2 = vld1q_f32(o_ptr + n + 8); c3 = vld1q_f32(o_ptr + n + 12);
                                }

                                const float* p_b = b_ptr_base + k_start * strideB_K + n;
                                const float* p_a = a_ptr + k_start * strideA_K;
                                uint32_t k = k_start;

                                for (; k + 1 < k_end; k += 2) {
                                    float32x4_t b0 = vld1q_f32(p_b + 0);
                                    float32x4_t b1 = vld1q_f32(p_b + 4);
                                    float32x4_t b2 = vld1q_f32(p_b + 8);
                                    float32x4_t b3 = vld1q_f32(p_b + 12);
                                    float32x4_t a0 = vdupq_n_f32(*p_a);
                                    c0 = vfmaq_f32(c0, b0, a0); c1 = vfmaq_f32(c1, b1, a0); c2 = vfmaq_f32(c2, b2, a0); c3 = vfmaq_f32(c3, b3, a0);
                                    p_b += strideB_K; p_a += strideA_K;

                                    b0 = vld1q_f32(p_b + 0);
                                    b1 = vld1q_f32(p_b + 4);
                                    b2 = vld1q_f32(p_b + 8);
                                    b3 = vld1q_f32(p_b + 12);
                                    a0 = vdupq_n_f32(*p_a);
                                    c0 = vfmaq_f32(c0, b0, a0); c1 = vfmaq_f32(c1, b1, a0); c2 = vfmaq_f32(c2, b2, a0); c3 = vfmaq_f32(c3, b3, a0);
                                    p_b += strideB_K; p_a += strideA_K;
                                }

                                for (; k < k_end; ++k) {
                                    float32x4_t b0 = vld1q_f32(p_b + 0);
                                    float32x4_t b1 = vld1q_f32(p_b + 4);
                                    float32x4_t b2 = vld1q_f32(p_b + 8);
                                    float32x4_t b3 = vld1q_f32(p_b + 12);
                                    float32x4_t a0 = vdupq_n_f32(*p_a);
                                    c0 = vfmaq_f32(c0, b0, a0); c1 = vfmaq_f32(c1, b1, a0); c2 = vfmaq_f32(c2, b2, a0); c3 = vfmaq_f32(c3, b3, a0);
                                    p_b += strideB_K; p_a += strideA_K;
                                }

                                vst1q_f32(o_ptr + n + 0, c0); vst1q_f32(o_ptr + n + 4, c1); 
                                vst1q_f32(o_ptr + n + 8, c2); vst1q_f32(o_ptr + n + 12, c3);
                            }
                        }
                    }
                }
            } });
    }

    for (auto &thread : workers)
        thread.join();
}

inline uint32_t refFactoryDotF32_3D_N64(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Dot 3D requires 2 inputs");

    return graph.dot(inputs[0], inputs[1]);
}

REGISTER_KERNEL(
    "Dot_F32_3D_CPU_Neon_N64",
    2,
    matchDotF32_3D_N64,
    runDotF32_3D_N64,
    refFactoryDotF32_3D_N64,
    {Backend::CPU},
    {DType::FLOAT32, DType::FLOAT32},
    {{1, 8, 8}, {1, 8, 64}},
    {true, true},
    {{Backend::CPU}, {Backend::CPU}});

#endif // TG_HAS_NEON