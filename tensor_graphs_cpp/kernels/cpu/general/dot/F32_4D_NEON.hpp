#pragma once
#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <vector>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

inline bool matchDotF32_4D_Neon(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &s0 = inputs[0].getShape();
    const auto &s1 = inputs[1].getShape();
    const auto &so = output.getShape();

    if (s0.size() != 4 || s1.size() != 4 || so.size() != 4)
        return false;
    if (s0[0] != s1[0] || s0[1] != s1[1] || s0[3] != s1[2])
        return false;
    if (so[0] != s0[0] || so[1] != s0[1] || so[2] != s0[2] || so[3] != s1[3])
        return false;

    return isContiguous(output);
}

inline void runDotF32_4D_Neon(const KernelContext &ctx)
{
    const float *A_ptr = static_cast<const float *>(ctx.inputs[0]);
    const float *B_ptr = static_cast<const float *>(ctx.inputs[1]);
    float *Out_ptr = static_cast<float *>(ctx.outputs[0]);

    const auto &viewA = ctx.inViews[0];
    const auto &viewB = ctx.inViews[1];
    const auto &viewOut = ctx.outViews[0];

    const uint32_t B_count = viewA.getShape()[0];
    const uint32_t H = viewA.getShape()[1];
    const uint32_t M = viewA.getShape()[2];
    const uint32_t K = viewA.getShape()[3];
    const uint32_t N = viewB.getShape()[3];

    const int64_t strideA_B = viewA.strides[0];
    const int64_t strideA_H = viewA.strides[1];
    const int64_t strideA_M = viewA.strides[2];
    const int64_t strideA_K = viewA.strides[3];

    const int64_t strideB_B = viewB.strides[0];
    const int64_t strideB_H = viewB.strides[1];
    const int64_t strideB_K = viewB.strides[2];
    const int64_t strideB_N = viewB.strides[3];

    const int64_t strideO_B = viewOut.strides[0];
    const int64_t strideO_H = viewOut.strides[1];
    const int64_t strideO_M = viewOut.strides[2];
    const int64_t strideO_N = viewOut.strides[3];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    uint32_t total_rows = B_count * H * M;
    num_threads = std::min(num_threads, total_rows);

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
        uint32_t start_row = t * rows_per_thread;
        uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);

        const uint32_t N16 = N & ~15u;
        const uint32_t N4 = N & ~3u;

        for (uint32_t row_idx = start_row; row_idx < end_row; ++row_idx)
        {
            uint32_t b = row_idx / (H * M);
            uint32_t rem = row_idx % (H * M);
            uint32_t h = rem / M;
            uint32_t m = rem % M;

            const float *rowA = A_ptr + (b * strideA_B) + (h * strideA_H) + (m * strideA_M);
            const float *matrixB = B_ptr + (b * strideB_B) + (h * strideB_H);
            float *rowOut = Out_ptr + (b * strideO_B) + (h * strideO_H) + (m * strideO_M);

            // Zero initialize output row
            for (uint32_t n = 0; n < N; ++n)
                rowOut[n * strideO_N] = 0.0f;

            for (uint32_t k = 0; k < K; ++k)
            {
                float valA = rowA[k * strideA_K];
                const float *rowB = matrixB + (k * strideB_K);

#if defined(TG_HAS_NEON)
                float32x4_t va = vdupq_n_f32(valA);
                uint32_t n = 0;

                if (strideB_N == 1 && strideO_N == 1)
                {
                    for (; n < N16; n += 16)
                    {
                        float32x4_t vo0 = vld1q_f32(rowOut + n);
                        float32x4_t vo1 = vld1q_f32(rowOut + n + 4);
                        float32x4_t vo2 = vld1q_f32(rowOut + n + 8);
                        float32x4_t vo3 = vld1q_f32(rowOut + n + 12);

                        float32x4_t vb0 = vld1q_f32(rowB + n);
                        float32x4_t vb1 = vld1q_f32(rowB + n + 4);
                        float32x4_t vb2 = vld1q_f32(rowB + n + 8);
                        float32x4_t vb3 = vld1q_f32(rowB + n + 12);

                        vst1q_f32(rowOut + n, vfmaq_f32(vo0, va, vb0));
                        vst1q_f32(rowOut + n + 4, vfmaq_f32(vo1, va, vb1));
                        vst1q_f32(rowOut + n + 8, vfmaq_f32(vo2, va, vb2));
                        vst1q_f32(rowOut + n + 12, vfmaq_f32(vo3, va, vb3));
                    }
                    for (; n < N4; n += 4)
                    {
                        float32x4_t vo = vld1q_f32(rowOut + n);
                        float32x4_t vb = vld1q_f32(rowB + n);
                        vst1q_f32(rowOut + n, vfmaq_f32(vo, va, vb));
                    }
                }
                for (; n < N; ++n)
                {
                    rowOut[n * strideO_N] += valA * rowB[n * strideB_N];
                }
#else
                for (uint32_t n = 0; n < N; ++n)
                {
                    rowOut[n * strideO_N] += valA * rowB[n * strideB_N];
                }
#endif
            }
        }
    });
}

inline LogicalId refFactoryDotF32_4D_Neon(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.dot(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Dot_F32_4D_CPU_Neon", 2, 2, matchDotF32_4D_Neon, runDotF32_4D_Neon, refFactoryDotF32_4D_Neon, {},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 32, 128, 128}, {1, 32, 128, 128}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});