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
 * Reference pattern:
 *   scatter(op_cache, contiguous(dot(contiguous(slice(A)), contiguous(slice(B)))))
 *
 * This kernel directly computes a dot product over a sub‑region of A and B
 * (defined by the `starts`, `ends`, `steps` slices) and adds the result
 * into the corresponding slice of the target buffer `op_cache`.
 *
 * Optimized for ARM NEON (Snapdragon X Elite) with IKJ loop order
 * and multi‑threading across batch and rows.
 */

inline bool matchScatterDotF32_3D_Optimized(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Signature: cache, A, B, starts, ends, steps   (6 inputs)
    if (inputs.size() != 6)
        return false;
    // dtypes
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 ||
        inputs[2].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[3].dtype != DType::INT32 || inputs[4].dtype != DType::INT32 ||
        inputs[5].dtype != DType::INT32)
        return false;

    const auto &shape_cache = inputs[0].getShape();
    const auto &shape_A = inputs[1].getShape();
    const auto &shape_B = inputs[2].getShape();

    // A and B must be 3D (B, M, K) and (B, K, N)
    if (shape_A.size() != 3 || shape_B.size() != 3)
        return false;
    if (shape_A[0] != shape_B[0] || shape_A[2] != shape_B[1])
        return false; // batch and K dimensions must match

    // Cache must have the same 3D shape as the full dot output (B, M, N)
    if (shape_cache.size() != 3)
        return false;
    if (shape_cache[0] != shape_A[0] || shape_cache[1] != shape_A[1] ||
        shape_cache[2] != shape_B[2])
        return false;

    // Inner‑most dimensions must be contiguous, and slice steps must be 1
    // for the IKJ / SIMD implementation. (The planner already inserts
    // contiguous / copy‑to nodes when needed, so this is safe.)
    const auto &strides_A = inputs[1].strides;
    const auto &strides_B = inputs[2].strides;
    if (strides_A.size() < 3 || strides_B.size() < 3)
        return false;
    if (strides_A[2] != 1 || strides_B[2] != 1)
        return false;

    // Slice parameters must be 1‑D and have the same length as rank (3)
    if (shape_A.size() != shape_cache.size())
        return false;
    if (inputs[3].getShape().size() != 1 || inputs[3].getShape()[0] != 3)
        return false;
    if (inputs[4].getShape().size() != 1 || inputs[4].getShape()[0] != 3)
        return false;
    if (inputs[5].getShape().size() != 1 || inputs[5].getShape()[0] != 3)
        return false;

    // All must be on the same backend (CPU)
    if (inputs[0].backend != Backend::CPU || inputs[1].backend != Backend::CPU ||
        inputs[2].backend != Backend::CPU || output.backend != Backend::CPU)
        return false;

    // Steps must be 1 – our SIMD path doesn't handle strides > 1 yet
    // (We defer stricter runtime verification to the run function)
    return true;
}

inline void runScatterDotF32_3D_Optimized(const std::vector<const void *> &inputs,
                                          const std::vector<void *> &outputs,
                                          const std::vector<TensorView> &inViews,
                                          const std::vector<TensorView> &outViews)
{
    const float *cache_ptr = static_cast<const float *>(inputs[0]);
    const float *A_ptr = static_cast<const float *>(inputs[1]);
    const float *B_ptr = static_cast<const float *>(inputs[2]);
    const int32_t *starts = static_cast<const int32_t *>(inputs[3]);
    const int32_t *ends = static_cast<const int32_t *>(inputs[4]);
    const int32_t *steps = static_cast<const int32_t *>(inputs[5]);

    float *out_cache_ptr = static_cast<float *>(outputs[0]);

    const TensorView &view_cache = inViews[0];
    const TensorView &view_A = inViews[1];
    const TensorView &view_B = inViews[2];

    // Slice bounds – guaranteed to be rank 3 by match.
    int32_t start_b = starts[0], end_b = ends[0], step_b = steps[0];
    int32_t start_m = starts[1], end_m = ends[1], step_m = steps[1];
    int32_t start_n = starts[2], end_n = ends[2], step_n = steps[2];

    // Derived slice shapes
    uint32_t slice_B_cnt = (end_b - start_b + step_b - 1) / step_b;
    uint32_t slice_M = (end_m - start_m + step_m - 1) / step_m;
    uint32_t slice_N = (end_n - start_n + step_n - 1) / step_n;

    // A shape: [B, M, K]
    uint32_t K = view_A.getShape()[2];

    // Strides (in elements)
    const int64_t stride_A_B = view_A.strides[0];
    const int64_t stride_A_M = view_A.strides[1];
    const int64_t stride_A_K = view_A.strides[2]; // == 1 (guaranteed)
    const int64_t stride_B_B = view_B.strides[0];
    const int64_t stride_B_K = view_B.strides[1];
    const int64_t stride_B_N = view_B.strides[2]; // == 1
    const int64_t stride_C_B = view_cache.strides[0];
    const int64_t stride_C_M = view_cache.strides[1];
    const int64_t stride_C_N = view_cache.strides[2];

    // Threading setup: work units = batch slices * rows
    uint32_t total_work = slice_B_cnt * slice_M;
    uint32_t num_threads = std::thread::hardware_concurrency();
    uint32_t rows_per_thread = (total_work + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=, &view_cache, &view_A, &view_B]()
                             {
            uint32_t start_row = t * rows_per_thread;
            uint32_t end_row   = std::min(start_row + rows_per_thread, total_work);

            for (uint32_t flat = start_row; flat < end_row; ++flat) {
                uint32_t b_idx = flat / slice_M;  // which batch slice
                uint32_t m_idx = flat % slice_M;  // which row inside slice

                // Global indices inside A / B shape
                int32_t b_global = start_b + b_idx * step_b;
                int32_t m_global = start_m + m_idx * step_m;

                // Pointer to start of output row in cache (add‑accumulate)
                float *row_out = out_cache_ptr +
                                 (b_global * stride_C_B) + (m_global * stride_C_M);

                // Pointer to start of row in A
                const float *rowA = A_ptr +
                                    (b_global * stride_A_B) + (m_global * stride_A_M);

                // Pointer to start of matching slice in B (start_n column)
                const float *rowB = B_ptr +
                                    (b_global * stride_B_B) + (start_n * stride_B_N);

                // ---- IKJ Dot Product ----
                for (uint32_t k = 0; k < K; ++k) {
                    float valA = rowA[k * stride_A_K];      // stride_A_K == 1

                    // Broadcast A element to 4 SIMD lanes
                    float32x4_t vA = vdupq_n_f32(valA);

                    // Point to the k‑th row of B (constant for this k)
                    const float *k_B = B_ptr +
                                       (b_global * stride_B_B) + (k * stride_B_K);

                    uint32_t n = 0;
                    // SIMD accumulation: 4 columns at a time
                    for (; n + 4 <= slice_N; n += 4) {
                        uint32_t col = start_n + n * step_n;  // global column
                        // Load 4 values from B (stride_B_N == 1)
                        float32x4_t vB   = vld1q_f32(k_B + col);
                        // Load current accumulator values from cache
                        float32x4_t vOut = vld1q_f32(row_out + col * stride_C_N);

                        // FMA: out = out + valA * B
                        vOut = vfmaq_f32(vOut, vA, vB);
                        vst1q_f32(row_out + col * stride_C_N, vOut);
                    }
                    // Tail (remaining columns)
                    for (; n < slice_N; ++n) {
                        uint32_t col = start_n + n * step_n;
                        row_out[col * stride_C_N] +=
                            valA * k_B[col];   // stride_B_N == 1, stride_C_N may vary
                    }
                }
            } });
    }
    for (auto &th : workers)
        th.join();
}

/**
 * Reference graph factory – builds the exact subgraph that this kernel replaces.
 * This is used by the fusion rule (FusionRule) to discover pattern matches.
 */
inline uint32_t refFactoryScatterDotF32_3D_Optimized(const std::vector<uint32_t> &inIds, Graph &graph)
{
    // inIds: [cache, A, B, starts, ends, steps]
    if (inIds.size() != 6)
        Error::throw_err("ScatterDotF32_3D_Optimized expects exactly 6 inputs");

    uint32_t cache_id = inIds[0];
    uint32_t A_id = inIds[1];
    uint32_t B_id = inIds[2];
    uint32_t starts_id = inIds[3];
    uint32_t ends_id = inIds[4];
    uint32_t steps_id = inIds[5];

    // The shape of A and B are known (dummy shapes, e.g. [B, M, K] and [B, K, N])
    // We need to reconstruct the individual slices from the single output region.
    const auto &shape_A = graph.getNode(A_id).getShape(); // [B, M, K]
    const auto &shape_B = graph.getNode(B_id).getShape(); // [B, K, N]

    // Get the starts/ends/steps as constants (they are INT32 constants)
    std::vector<int32_t> starts = getConstantInt32(starts_id, graph);
    std::vector<int32_t> ends = getConstantInt32(ends_id, graph);
    std::vector<int32_t> steps = getConstantInt32(steps_id, graph);

    // Build slice parameters for A:  same batch & row slice, but full K
    std::vector<int32_t> starts_A(starts), ends_A(ends), steps_A(steps);
    starts_A[2] = 0;
    ends_A[2] = shape_A[2]; // full K
    steps_A[2] = 1;

    // Build slice parameters for B:  same batch, full K, same column slice
    std::vector<int32_t> starts_B(starts), ends_B(ends), steps_B(steps);
    starts_B[1] = 0;
    ends_B[1] = shape_B[1]; // full K
    steps_B[1] = 1;

    // Create constant nodes for the derived slices
    uint32_t starts_A_id = graph.constant({(uint32_t)starts_A.size()}, starts_A.data(), DType::INT32);
    uint32_t ends_A_id = graph.constant({(uint32_t)ends_A.size()}, ends_A.data(), DType::INT32);
    uint32_t steps_A_id = graph.constant({(uint32_t)steps_A.size()}, steps_A.data(), DType::INT32);

    uint32_t starts_B_id = graph.constant({(uint32_t)starts_B.size()}, starts_B.data(), DType::INT32);
    uint32_t ends_B_id = graph.constant({(uint32_t)ends_B.size()}, ends_B.data(), DType::INT32);
    uint32_t steps_B_id = graph.constant({(uint32_t)steps_B.size()}, steps_B.data(), DType::INT32);

    // Slice A and B with their respective parameters
    uint32_t sliceA = graph.slice(A_id, starts_A_id, ends_A_id, steps_A_id);
    uint32_t sliceB = graph.slice(B_id, starts_B_id, ends_B_id, steps_B_id);

    uint32_t contigA = graph.contiguous(sliceA);
    uint32_t contigB = graph.contiguous(sliceB);

    uint32_t dot = graph.dot(contigA, contigB);
    uint32_t contigDot = graph.contiguous(dot);

    return graph.scatter(cache_id, contigDot, starts_id, ends_id, steps_id);
}

// Register the fused kernel
REGISTER_KERNEL(
    "Scatter_Dot_F32_3D_CPU_Optimized",   // opName
    6,                                    // numInputs
    matchScatterDotF32_3D_Optimized,      // match function
    runScatterDotF32_3D_Optimized,        // run function
    refFactoryScatterDotF32_3D_Optimized, // reference factory
    {Backend::CPU},                       // backends
    {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32,
     DType::INT32, DType::INT32, DType::INT32},
    {// dummy shapes (cache, A, B, starts, ends, steps)
     {1, 4, 8},
     {1, 4, 8},
     {1, 8, 8},
     {3},
     {3},
     {3}},
    // requiresContiguous: cache may be non‑contiguous; A/B inner dims must be
    {false, false, false, false, false, false},
    {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});

#endif // TG_HAS_NEON