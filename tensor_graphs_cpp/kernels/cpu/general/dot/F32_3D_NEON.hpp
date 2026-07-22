#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h> // ARM SIMD Intrinsics
#include <thread>
#include <vector>
#include <algorithm>

/**
 * Optimized ARM NEON Dot Product for Snapdragon X Elite
 * Implementation: 2D Blocked K-outer Loop with 4x16 Register Microkernel.
 * Parallelization: Distributed across a 2D grid of M and N to eliminate memory contention.
 */
inline bool matchDotF32_3D_Neon(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

    if (!isContiguous(output))
        return false;

    return true;
}

inline void runDotF32_3D_Neon(const KernelContext &ctx)
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
    const uint32_t N = viewB.getShape()[2];

    const int64_t strideA_B = viewA.strides[0];
    const int64_t strideA_M = viewA.strides[1];
    const int64_t strideA_K = viewA.strides[2];

    const int64_t strideB_B = viewB.strides[0];
    const int64_t strideB_K = viewB.strides[1];
    const int64_t strideB_N = viewB.strides[2];

    const int64_t strideO_B = viewOut.strides[0];
    const int64_t strideO_M = viewOut.strides[1];
    const int64_t strideO_N = viewOut.strides[2];

    uint32_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;

    // Fast path uses aggressive NEON register caching but requires contiguous inner dimension
    bool is_fast_path = (strideB_N == 1 && strideO_N == 1);

    if (is_fast_path)
    {
        // Grid dimensioning to perfectly balance M x N workloads across all 12 cores
        uint32_t n_split = 1;
        uint32_t m_split = 1;

        if (N > M)
        {
            n_split = std::min(num_threads, (N + 63) / 64);
            m_split = std::max(1u, num_threads / n_split);
        }
        else
        {
            m_split = std::min(num_threads, (M + 3) / 4);
            n_split = std::max(1u, num_threads / m_split);
        }

        uint32_t actual_threads = m_split * n_split;
        uint32_t m_chunk = (M + m_split - 1) / m_split;
        uint32_t n_chunk = (N + n_split - 1) / n_split;
        n_chunk = ((n_chunk + 15) / 16) * 16; // strict 16-alignment for wide N SIMD

        for (uint32_t t = 0; t < actual_threads; ++t)
        {
            uint32_t t_m = t / n_split;
            uint32_t t_n = t % n_split;

            workers.emplace_back([=]()
                                 {
                uint32_t m_start = t_m * m_chunk;
                uint32_t m_end = std::min(m_start + m_chunk, M);
                uint32_t n_start = t_n * n_chunk;
                uint32_t n_end = std::min(n_start + n_chunk, N);
                
                if (m_start >= m_end || n_start >= n_end) return;
                
                const uint32_t K_BLK = 256; // Tiling factor to protect L1 Data Cache & TLB

                for (uint32_t b = 0; b < B_count; ++b) {
                    const float* a_ptr_base = A_ptr + b * strideA_B;
                    const float* b_ptr_base = B_ptr + b * strideB_B;
                    float* o_ptr_base = Out_ptr + b * strideO_B;

                    for (uint32_t k_start = 0; k_start < K; k_start += K_BLK) {
                        uint32_t k_end = std::min(k_start + K_BLK, K);
                        bool is_first_k = (k_start == 0);

                        // M-Block = 4
                        uint32_t m = m_start;
                        for (; m + 3 < m_end; m += 4) {
                            const float* a0_ptr = a_ptr_base + (m + 0) * strideA_M;
                            const float* a1_ptr = a_ptr_base + (m + 1) * strideA_M;
                            const float* a2_ptr = a_ptr_base + (m + 2) * strideA_M;
                            const float* a3_ptr = a_ptr_base + (m + 3) * strideA_M;

                            float* o0_ptr = o_ptr_base + (m + 0) * strideO_M;
                            float* o1_ptr = o_ptr_base + (m + 1) * strideO_M;
                            float* o2_ptr = o_ptr_base + (m + 2) * strideO_M;
                            float* o3_ptr = o_ptr_base + (m + 3) * strideO_M;

                            uint32_t n = n_start;
                            
                            // Highly tuned 4x16 Inner Microkernel (Uses 24 / 32 NEON registers entirely)
                            for (; n + 15 < n_end; n += 16) {
                                float32x4_t c00, c01, c02, c03;
                                float32x4_t c10, c11, c12, c13;
                                float32x4_t c20, c21, c22, c23;
                                float32x4_t c30, c31, c32, c33;

                                if (is_first_k) {
                                    c00 = vdupq_n_f32(0); c01 = vdupq_n_f32(0); c02 = vdupq_n_f32(0); c03 = vdupq_n_f32(0);
                                    c10 = vdupq_n_f32(0); c11 = vdupq_n_f32(0); c12 = vdupq_n_f32(0); c13 = vdupq_n_f32(0);
                                    c20 = vdupq_n_f32(0); c21 = vdupq_n_f32(0); c22 = vdupq_n_f32(0); c23 = vdupq_n_f32(0);
                                    c30 = vdupq_n_f32(0); c31 = vdupq_n_f32(0); c32 = vdupq_n_f32(0); c33 = vdupq_n_f32(0);
                                } else {
                                    c00 = vld1q_f32(o0_ptr + n + 0); c01 = vld1q_f32(o0_ptr + n + 4); c02 = vld1q_f32(o0_ptr + n + 8); c03 = vld1q_f32(o0_ptr + n + 12);
                                    c10 = vld1q_f32(o1_ptr + n + 0); c11 = vld1q_f32(o1_ptr + n + 4); c12 = vld1q_f32(o1_ptr + n + 8); c13 = vld1q_f32(o1_ptr + n + 12);
                                    c20 = vld1q_f32(o2_ptr + n + 0); c21 = vld1q_f32(o2_ptr + n + 4); c22 = vld1q_f32(o2_ptr + n + 8); c23 = vld1q_f32(o2_ptr + n + 12);
                                    c30 = vld1q_f32(o3_ptr + n + 0); c31 = vld1q_f32(o3_ptr + n + 4); c32 = vld1q_f32(o3_ptr + n + 8); c33 = vld1q_f32(o3_ptr + n + 12);
                                }

                                for (uint32_t k = k_start; k < k_end; ++k) {
                                    const float* b_ptr = b_ptr_base + k * strideB_K + n;
                                    float32x4_t b0 = vld1q_f32(b_ptr + 0);
                                    float32x4_t b1 = vld1q_f32(b_ptr + 4);
                                    float32x4_t b2 = vld1q_f32(b_ptr + 8);
                                    float32x4_t b3 = vld1q_f32(b_ptr + 12);

                                    float32x4_t a0 = vdupq_n_f32(a0_ptr[k * strideA_K]);
                                    c00 = vfmaq_f32(c00, b0, a0); c01 = vfmaq_f32(c01, b1, a0); c02 = vfmaq_f32(c02, b2, a0); c03 = vfmaq_f32(c03, b3, a0);

                                    float32x4_t a1 = vdupq_n_f32(a1_ptr[k * strideA_K]);
                                    c10 = vfmaq_f32(c10, b0, a1); c11 = vfmaq_f32(c11, b1, a1); c12 = vfmaq_f32(c12, b2, a1); c13 = vfmaq_f32(c13, b3, a1);

                                    float32x4_t a2 = vdupq_n_f32(a2_ptr[k * strideA_K]);
                                    c20 = vfmaq_f32(c20, b0, a2); c21 = vfmaq_f32(c21, b1, a2); c22 = vfmaq_f32(c22, b2, a2); c23 = vfmaq_f32(c23, b3, a2);

                                    float32x4_t a3 = vdupq_n_f32(a3_ptr[k * strideA_K]);
                                    c30 = vfmaq_f32(c30, b0, a3); c31 = vfmaq_f32(c31, b1, a3); c32 = vfmaq_f32(c32, b2, a3); c33 = vfmaq_f32(c33, b3, a3);
                                }

                                vst1q_f32(o0_ptr + n + 0, c00); vst1q_f32(o0_ptr + n + 4, c01); vst1q_f32(o0_ptr + n + 8, c02); vst1q_f32(o0_ptr + n + 12, c03);
                                vst1q_f32(o1_ptr + n + 0, c10); vst1q_f32(o1_ptr + n + 4, c11); vst1q_f32(o1_ptr + n + 8, c12); vst1q_f32(o1_ptr + n + 12, c13);
                                vst1q_f32(o2_ptr + n + 0, c20); vst1q_f32(o2_ptr + n + 4, c21); vst1q_f32(o2_ptr + n + 8, c22); vst1q_f32(o2_ptr + n + 12, c23);
                                vst1q_f32(o3_ptr + n + 0, c30); vst1q_f32(o3_ptr + n + 4, c31); vst1q_f32(o3_ptr + n + 8, c32); vst1q_f32(o3_ptr + n + 12, c33);
                            }

                            // Tails for N when M-Block = 4
                            for (; n + 3 < n_end; n += 4) {
                                float32x4_t c0, c1, c2, c3;
                                if (is_first_k) {
                                    c0 = vdupq_n_f32(0); c1 = vdupq_n_f32(0); c2 = vdupq_n_f32(0); c3 = vdupq_n_f32(0);
                                } else {
                                    c0 = vld1q_f32(o0_ptr + n); c1 = vld1q_f32(o1_ptr + n); c2 = vld1q_f32(o2_ptr + n); c3 = vld1q_f32(o3_ptr + n);
                                }
                                for (uint32_t k = k_start; k < k_end; ++k) {
                                    float32x4_t b0 = vld1q_f32(b_ptr_base + k * strideB_K + n);
                                    c0 = vfmaq_f32(c0, b0, vdupq_n_f32(a0_ptr[k * strideA_K]));
                                    c1 = vfmaq_f32(c1, b0, vdupq_n_f32(a1_ptr[k * strideA_K]));
                                    c2 = vfmaq_f32(c2, b0, vdupq_n_f32(a2_ptr[k * strideA_K]));
                                    c3 = vfmaq_f32(c3, b0, vdupq_n_f32(a3_ptr[k * strideA_K]));
                                }
                                vst1q_f32(o0_ptr + n, c0); vst1q_f32(o1_ptr + n, c1); vst1q_f32(o2_ptr + n, c2); vst1q_f32(o3_ptr + n, c3);
                            }

                            for (; n < n_end; ++n) {
                                float c0 = is_first_k ? 0 : o0_ptr[n];
                                float c1 = is_first_k ? 0 : o1_ptr[n];
                                float c2 = is_first_k ? 0 : o2_ptr[n];
                                float c3 = is_first_k ? 0 : o3_ptr[n];
                                for (uint32_t k = k_start; k < k_end; ++k) {
                                    float b0 = b_ptr_base[k * strideB_K + n];
                                    c0 += a0_ptr[k * strideA_K] * b0;
                                    c1 += a1_ptr[k * strideA_K] * b0;
                                    c2 += a2_ptr[k * strideA_K] * b0;
                                    c3 += a3_ptr[k * strideA_K] * b0;
                                }
                                o0_ptr[n] = c0; o1_ptr[n] = c1; o2_ptr[n] = c2; o3_ptr[n] = c3;
                            }
                        }

                        // Tail for M (when M is not a clean multiple of 4)
                        for (; m < m_end; ++m) {
                            const float* a_ptr = a_ptr_base + m * strideA_M;
                            float* o_ptr = o_ptr_base + m * strideO_M;
                            
                            uint32_t n = n_start;
                            for (; n + 15 < n_end; n += 16) {
                                float32x4_t c0, c1, c2, c3;
                                if (is_first_k) {
                                    c0 = vdupq_n_f32(0); c1 = vdupq_n_f32(0); c2 = vdupq_n_f32(0); c3 = vdupq_n_f32(0);
                                } else {
                                    c0 = vld1q_f32(o_ptr + n + 0); c1 = vld1q_f32(o_ptr + n + 4); c2 = vld1q_f32(o_ptr + n + 8); c3 = vld1q_f32(o_ptr + n + 12);
                                }
                                for (uint32_t k = k_start; k < k_end; ++k) {
                                    float32x4_t a0 = vdupq_n_f32(a_ptr[k * strideA_K]);
                                    const float* b_ptr = b_ptr_base + k * strideB_K + n;
                                    c0 = vfmaq_f32(c0, vld1q_f32(b_ptr + 0), a0);
                                    c1 = vfmaq_f32(c1, vld1q_f32(b_ptr + 4), a0);
                                    c2 = vfmaq_f32(c2, vld1q_f32(b_ptr + 8), a0);
                                    c3 = vfmaq_f32(c3, vld1q_f32(b_ptr + 12), a0);
                                }
                                vst1q_f32(o_ptr + n + 0, c0); vst1q_f32(o_ptr + n + 4, c1); vst1q_f32(o_ptr + n + 8, c2); vst1q_f32(o_ptr + n + 12, c3);
                            }
                            for (; n + 3 < n_end; n += 4) {
                                float32x4_t c0 = is_first_k ? vdupq_n_f32(0) : vld1q_f32(o_ptr + n);
                                for (uint32_t k = k_start; k < k_end; ++k) {
                                    c0 = vfmaq_f32(c0, vld1q_f32(b_ptr_base + k * strideB_K + n), vdupq_n_f32(a_ptr[k * strideA_K]));
                                }
                                vst1q_f32(o_ptr + n, c0);
                            }
                            for (; n < n_end; ++n) {
                                float c0 = is_first_k ? 0 : o_ptr[n];
                                for (uint32_t k = k_start; k < k_end; ++k) {
                                    c0 += a_ptr[k * strideA_K] * b_ptr_base[k * strideB_K + n];
                                }
                                o_ptr[n] = c0;
                            }
                        }
                    }
                } });
        }
    }
    else
    {
        // Slow Fallback (Preserves Original Correctness for odd/non-contiguous matrices)
        uint32_t total_rows = B_count * M;
        uint32_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;

        for (uint32_t t = 0; t < num_threads; ++t)
        {
            workers.emplace_back([=]()
                                 {
                uint32_t start_row = t * rows_per_thread;
                uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);

                for (uint32_t row_idx = start_row; row_idx < end_row; ++row_idx) {
                    uint32_t b = row_idx / M;
                    uint32_t m = row_idx % M;

                    const float* rowA = A_ptr + (b * strideA_B) + (m * strideA_M);
                    const float* batchB = B_ptr + (b * strideB_B);
                    float* rowOut = Out_ptr + (b * strideO_B) + (m * strideO_M);

                    for (uint32_t n = 0; n < N; ++n) rowOut[n * strideO_N] = 0.0f;

                    for (uint32_t k = 0; k < K; ++k) {
                        float valA = rowA[k * strideA_K];
                        const float* rowB = batchB + (k * strideB_K);
                        for (uint32_t n = 0; n < N; ++n) {
                            rowOut[n * strideO_N] += valA * rowB[n * strideB_N];
                        }
                    }
                } });
        }
    }

    for (auto &thread : workers)
        thread.join();
}

inline LogicalId refFactoryDotF32_3D_Neon(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Dot 3D requires 2 inputs");

    return graph.dot(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Dot_F32_3D_CPU_Neon", 2, 2, matchDotF32_3D_Neon, runDotF32_3D_Neon, refFactoryDotF32_3D_Neon, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32}, {{1, 8, 8}, {1, 8, 8}}, {true, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON