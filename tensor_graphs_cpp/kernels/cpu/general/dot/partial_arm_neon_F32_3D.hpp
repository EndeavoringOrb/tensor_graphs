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

    return true;
}

inline void runScatterDotF32_3D_Optimized(const std::vector<const void *> &inputs,
                                          const std::vector<void *> &outputs,
                                          const std::vector<TensorView> &inViews,
                                          const std::vector<TensorView> &outViews)
{
    const float *target_ptr = static_cast<const float *>(inputs[0]);
    const float *A_ptr = static_cast<const float *>(inputs[1]);
    const float *B_ptr = static_cast<const float *>(inputs[2]);

    const int32_t *starts_raw = static_cast<const int32_t *>(inputs[3]);
    const int32_t *ends_raw = static_cast<const int32_t *>(inputs[4]);
    const int32_t *steps_raw = static_cast<const int32_t *>(inputs[5]);
    const int32_t *startsA_raw = static_cast<const int32_t *>(inputs[6]);
    const int32_t *endsA_raw = static_cast<const int32_t *>(inputs[7]);
    const int32_t *stepsA_raw = static_cast<const int32_t *>(inputs[8]);
    const int32_t *startsB_raw = static_cast<const int32_t *>(inputs[9]);
    const int32_t *endsB_raw = static_cast<const int32_t *>(inputs[10]);
    const int32_t *stepsB_raw = static_cast<const int32_t *>(inputs[11]);

    float *out_cache_ptr = static_cast<float *>(outputs[0]);

    const TensorView &view_cache = inViews[0];
    const TensorView &view_A = inViews[1];
    const TensorView &view_B = inViews[2];

    if (target_ptr != out_cache_ptr)
    {
        uint64_t n_target = countElements(outViews[0].getShape());
        for (uint64_t i = 0; i < n_target; ++i)
        {
            out_cache_ptr[getStridedIndex(i, outViews[0].getShape(), outViews[0].strides)] =
                target_ptr[getStridedIndex(i, outViews[0].getShape(), inViews[0].strides)];
        }
    }

    int32_t starts[3], ends[3], steps[3];
    int32_t startsA[3], endsA[3], stepsA[3];
    int32_t startsB[3], endsB[3], stepsB[3];

    for (int i = 0; i < 3; ++i)
    {
        starts[i] = (inViews[3].getShape().empty() || i >= inViews[3].getShape()[0]) ? 0 : starts_raw[i];
        ends[i] = (inViews[4].getShape().empty() || i >= inViews[4].getShape()[0]) ? view_cache.getShape()[i] : ends_raw[i];
        steps[i] = (inViews[5].getShape().empty() || i >= inViews[5].getShape()[0]) ? 1 : steps_raw[i];

        startsA[i] = (inViews[6].getShape().empty() || i >= inViews[6].getShape()[0]) ? 0 : startsA_raw[i];
        endsA[i] = (inViews[7].getShape().empty() || i >= inViews[7].getShape()[0]) ? view_A.getShape()[i] : endsA_raw[i];
        stepsA[i] = (inViews[8].getShape().empty() || i >= inViews[8].getShape()[0]) ? 1 : stepsA_raw[i];

        startsB[i] = (inViews[9].getShape().empty() || i >= inViews[9].getShape()[0]) ? 0 : startsB_raw[i];
        endsB[i] = (inViews[10].getShape().empty() || i >= inViews[10].getShape()[0]) ? view_B.getShape()[i] : endsB_raw[i];
        stepsB[i] = (inViews[11].getShape().empty() || i >= inViews[11].getShape()[0]) ? 1 : stepsB_raw[i];
    }

    auto get_dim = [](int32_t s, int32_t e, int32_t st, uint32_t dim_len) -> uint32_t
    {
        if (s < 0)
            s += dim_len;
        if (e < 0)
            e += dim_len;
        s = std::max(0, std::min(s, (int32_t)dim_len));
        e = std::max(0, std::min(e, (int32_t)dim_len));
        if (st <= 0 || s >= e)
            return 0;
        return (e - s + st - 1) / st;
    };

    uint32_t slice_B_cnt = get_dim(starts[0], ends[0], steps[0], view_cache.getShape()[0]);
    uint32_t slice_M = get_dim(starts[1], ends[1], steps[1], view_cache.getShape()[1]);
    uint32_t slice_N = get_dim(starts[2], ends[2], steps[2], view_cache.getShape()[2]);

    uint32_t K = get_dim(startsA[2], endsA[2], stepsA[2], view_A.getShape()[2]);

    if (slice_B_cnt == 0 || slice_M == 0 || slice_N == 0 || K == 0)
        return;

    int32_t start_b = starts[0];
    if (start_b < 0)
        start_b += view_cache.getShape()[0];
    int32_t start_m = starts[1];
    if (start_m < 0)
        start_m += view_cache.getShape()[1];
    int32_t start_n = starts[2];
    if (start_n < 0)
        start_n += view_cache.getShape()[2];
    int32_t step_b = steps[0], step_m = steps[1], step_n = steps[2];

    int32_t startA_b = startsA[0];
    if (startA_b < 0)
        startA_b += view_A.getShape()[0];
    int32_t startA_m = startsA[1];
    if (startA_m < 0)
        startA_m += view_A.getShape()[1];
    int32_t startA_k = startsA[2];
    if (startA_k < 0)
        startA_k += view_A.getShape()[2];

    int32_t startB_b = startsB[0];
    if (startB_b < 0)
        startB_b += view_B.getShape()[0];
    int32_t startB_k = startsB[1];
    if (startB_k < 0)
        startB_k += view_B.getShape()[1];
    int32_t startB_n = startsB[2];
    if (startB_n < 0)
        startB_n += view_B.getShape()[2];

    const int64_t stride_A_B = view_A.strides[0] * stepsA[0];
    const int64_t stride_A_M = view_A.strides[1] * stepsA[1];
    const int64_t stride_A_K = view_A.strides[2] * stepsA[2];

    const int64_t stride_B_B = view_B.strides[0] * stepsB[0];
    const int64_t stride_B_K = view_B.strides[1] * stepsB[1];
    const int64_t stride_B_N = view_B.strides[2] * stepsB[2];

    const int64_t stride_C_B = view_cache.strides[0];
    const int64_t stride_C_M = view_cache.strides[1];
    const int64_t stride_C_N = view_cache.strides[2];

    A_ptr += (startA_b * view_A.strides[0] + startA_m * view_A.strides[1] + startA_k * view_A.strides[2]);
    B_ptr += (startB_b * view_B.strides[0] + startB_k * view_B.strides[1] + startB_n * view_B.strides[2]);

    bool can_simd = (stride_C_N == 1 && step_n == 1 && stride_B_N == 1 && stride_A_K == 1);

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

                if (can_simd) {
                    uint32_t n = 0;
                    float32x4_t vZero = vdupq_n_f32(0.0f);
                    for (; n + 4 <= slice_N; n += 4) {
                        uint32_t col = start_n + n;
                        vst1q_f32(row_out + col, vZero);
                    }
                    for (; n < slice_N; ++n) {
                        row_out[start_n + n] = 0.0f;
                    }
                } else {
                    for (uint32_t n = 0; n < slice_N; ++n) {
                        row_out[(start_n + n * step_n) * stride_C_N] = 0.0f;
                    }
                }

                for (uint32_t k = 0; k < K; ++k) {
                    float a_val = rowA[k * stride_A_K];
                    const float *k_B = B_ptr + (b_idx * stride_B_B) + (k * stride_B_K);

                    if (can_simd) {
                        float32x4_t vA = vdupq_n_f32(a_val);
                        uint32_t n = 0;
                        for (; n + 4 <= slice_N; n += 4) {
                            uint32_t col = start_n + n;
                            float32x4_t vB = vld1q_f32(k_B + n);
                            float32x4_t vOut = vld1q_f32(row_out + col);
                            vOut = vfmaq_f32(vOut, vA, vB);
                            vst1q_f32(row_out + col, vOut);
                        }
                        for (; n < slice_N; ++n) {
                            row_out[start_n + n] += a_val * k_B[n];
                        }
                    } else {
                        for (uint32_t n = 0; n < slice_N; ++n) {
                            row_out[(start_n + n * step_n) * stride_C_N] += a_val * k_B[n * stride_B_N];
                        }
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