// File: tensor_graphs_cpp/kernels/cpu/general/dot/partial_arm_neon_F32_3D.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "core/graph.hpp"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <thread>
#include <vector>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

/**
 * Fused Scatter-Dot kernel for partial recomputation paths.
 *
 * Updated to take 12 inputs to allow the FusionRule to match the pattern
 * without needing to inspect constant values during planning.
 */

inline bool matchScatterDotF32_3D_Optimized(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Signature: [cache, A, B, starts, ends, steps, startsA, endsA, stepsA, startsB, endsB, stepsB] (12 inputs)
    if (inputs.size() != 12)
        return false;

    // Check main tensor dtypes
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 ||
        inputs[2].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Check coordinate tensor dtypes
    for (int i = 3; i < 12; ++i)
    {
        if (inputs[i].dtype != DType::INT32)
            return false;
    }

    const auto &shape_A = inputs[1].getShape();
    const auto &shape_B = inputs[2].getShape();

    if (shape_A.size() != 3 || shape_B.size() != 3)
        return false;

    // SIMD requirement: Inner-most dimensions must be contiguous
    if (inputs[1].strides.back() != 1 || inputs[2].strides.back() != 1)
        return false;

    return true;
}

inline void runScatterDotF32_3D_Optimized(const std::vector<const void *> &inputs,
                                          const std::vector<void *> &outputs,
                                          const std::vector<TensorView> &inViews,
                                          const std::vector<TensorView> &outViews)
{
    // We only actually need the Scatter coordinates (3, 4, 5) for execution logic,
    // as the SIMD loop bounds are derived from the view shapes of A and B.
    const float *A_ptr = static_cast<const float *>(inputs[1]);
    const float *B_ptr = static_cast<const float *>(inputs[2]);
    const int32_t *starts = static_cast<const int32_t *>(inputs[3]);
    const int32_t *steps = static_cast<const int32_t *>(inputs[5]);

    float *out_cache_ptr = static_cast<float *>(outputs[0]);

    const TensorView &view_cache = inViews[0];
    const TensorView &view_A = inViews[1];
    const TensorView &view_B = inViews[2];

    int32_t start_b = starts[0], step_b = steps[0];
    int32_t start_m = starts[1], step_m = steps[1];
    int32_t start_n = starts[2], step_n = steps[2];

    uint32_t slice_B_cnt = view_A.getShape()[0];
    uint32_t slice_M = view_A.getShape()[1];
    uint32_t slice_N = view_B.getShape()[2];
    uint32_t K = view_A.getShape()[2];

    const int64_t stride_A_B = view_A.strides[0];
    const int64_t stride_A_M = view_A.strides[1];
    const int64_t stride_B_B = view_B.strides[0];
    const int64_t stride_B_K = view_B.strides[1];
    const int64_t stride_C_B = view_cache.strides[0];
    const int64_t stride_C_M = view_cache.strides[1];
    const int64_t stride_C_N = view_cache.strides[2];

    uint32_t total_work = slice_B_cnt * slice_M;
    uint32_t num_threads = std::thread::hardware_concurrency();
    uint32_t rows_per_thread = (total_work + num_threads - 1) / (num_threads ? num_threads : 1);

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t start_flat = t * rows_per_thread;
            uint32_t end_flat   = std::min(start_flat + rows_per_thread, total_work);

            for (uint32_t flat = start_flat; flat < end_flat; ++flat) {
                uint32_t b_idx = flat / slice_M;
                uint32_t m_idx = flat % slice_M;

                int32_t b_global = start_b + b_idx * step_b;
                int32_t m_global = start_m + m_idx * step_m;

                float *row_out = out_cache_ptr + (b_global * stride_C_B) + (m_global * stride_C_M);
                const float *rowA = A_ptr + (b_idx * stride_A_B) + (m_idx * stride_A_M);

                for (uint32_t k = 0; k < K; ++k) {
                    float32x4_t vA = vdupq_n_f32(rowA[k]);
                    const float *k_B = B_ptr + (b_idx * stride_B_B) + (k * stride_B_K);

                    uint32_t n = 0;
                    for (; n + 4 <= slice_N; n += 4) {
                        uint32_t col = start_n + n * step_n;
                        float32x4_t vB = vld1q_f32(k_B + n);
                        float32x4_t vOut = vld1q_f32(row_out + col * stride_C_N);
                        vOut = vfmaq_f32(vOut, vA, vB);
                        vst1q_f32(row_out + col * stride_C_N, vOut);
                    }
                    for (; n < slice_N; ++n) {
                        row_out[(start_n + n * step_n) * stride_C_N] += rowA[k] * k_B[n];
                    }
                }
            } });
    }
    for (auto &th : workers)
        th.join();
}

inline uint32_t refFactoryScatterDotF32_3D_Optimized(const std::vector<uint32_t> &inIds, Graph &graph)
{
    // inIds: [cache, A, B, sS, eS, tS, sA, eA, tA, sB, eB, tB]
    uint32_t sliceA = graph.slice(inIds[1], inIds[6], inIds[7], inIds[8]);
    uint32_t sliceB = graph.slice(inIds[2], inIds[9], inIds[10], inIds[11]);

    uint32_t contigA = graph.contiguous(sliceA);
    uint32_t contigB = graph.contiguous(sliceB);

    uint32_t dot = graph.dot(contigA, contigB);
    uint32_t contigDot = graph.contiguous(dot);

    return graph.scatter(inIds[0], contigDot, inIds[3], inIds[4], inIds[5]);
}

REGISTER_KERNEL(
    "Scatter_Dot_F32_3D_CPU_Optimized",
    12,
    matchScatterDotF32_3D_Optimized,
    runScatterDotF32_3D_Optimized,
    refFactoryScatterDotF32_3D_Optimized,
    {Backend::CPU},
    {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32,
     DType::INT32, DType::INT32, DType::INT32,
     DType::INT32, DType::INT32, DType::INT32,
     DType::INT32, DType::INT32, DType::INT32},
    {{1, 4, 8}, {1, 4, 8}, {1, 8, 8}, {3}, {3}, {3}, {3}, {3}, {3}, {3}, {3}, {3}},
    {false, false, false, false, false, false, false, false, false, false, false, false},
    {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});

#endif // TG_HAS_NEON