#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>

inline bool matchDotF32_3D_TransposedB_Optimized(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

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
    if (!isContiguous(inputs[0]))
        return false;

    if (inputs[1].strides[1] != 1)
        return false;

    return true;
}

inline void runDotF32_3D_TransposedB_Optimized(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                   const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *A_ptr = static_cast<const float *>(inputs[0]);
    const float *B_ptr = static_cast<const float *>(inputs[1]);
    float *Out_ptr = static_cast<float *>(outputs[0]);

    const auto &viewA = inViews[0];
    const auto &viewB = inViews[1];
    const auto &viewOut = outViews[0];

    const uint32_t B_count = viewA.getShape()[0];
    const uint32_t M = viewA.getShape()[1];
    const uint32_t K = viewA.getShape()[2];
    const uint32_t N = viewB.getShape()[2];

    const int64_t strideA_B = viewA.strides[0];
    const int64_t strideA_M = viewA.strides[1];

    const int64_t strideB_B = viewB.strides[0];
    const int64_t strideB_N = viewB.strides[2]; 

    const int64_t strideO_B = viewOut.strides[0];
    const int64_t strideO_M = viewOut.strides[1];
    const int64_t strideO_N = viewOut.strides[2];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;
    std::vector<std::thread> workers;
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

                for (uint32_t n = 0; n < N; ++n) {
                    const float* rowB = batchB + (n * strideB_N);
                    
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    uint32_t k = 0;
                    for (; k + 4 <= K; k += 4) {
                        float32x4_t vA = vld1q_f32(rowA + k);
                        float32x4_t vB = vld1q_f32(rowB + k);
                        sum_vec = vfmaq_f32(sum_vec, vA, vB);
                    }
                    float sum = vaddvq_f32(sum_vec);
                    for (; k < K; ++k) {
                        sum += rowA[k] * rowB[k];
                    }
                    rowOut[n * strideO_N] = sum;
                }
            } });
    }
    for (auto &thread : workers)
        thread.join();
}

inline uint32_t refFactoryDotF32_3D_TransposedB_Optimized(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.dot(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Dot_F32_3D_TransposedB_Optimized", 2, matchDotF32_3D_TransposedB_Optimized, runDotF32_3D_TransposedB_Optimized, refFactoryDotF32_3D_TransposedB_Optimized, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 8, 8}, {1, 8, 8}}, {false, false}, {{Backend::CPU}, {Backend::CPU}});
#endif