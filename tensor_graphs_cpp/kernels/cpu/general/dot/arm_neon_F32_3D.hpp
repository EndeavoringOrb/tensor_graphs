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
 * Implementation: IKJ loop order to maximize cache hits and SIMD throughput.
 * Parallelization: Distributed across Batch (B) and Row (M) dimensions.
 */
inline bool matchDotF32_3D_Optimized(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

    // For SIMD optimization, we require the inner-most dimension (N) to be contiguous
    if (!isContiguous(output))
        return false;

    return true;
}

inline void runDotF32_3D_Optimized(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
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
    const int64_t strideA_K = viewA.strides[2];

    const int64_t strideB_B = viewB.strides[0];
    const int64_t strideB_K = viewB.strides[1];
    const int64_t strideB_N = viewB.strides[2];

    const int64_t strideO_B = viewOut.strides[0];
    const int64_t strideO_M = viewOut.strides[1];
    const int64_t strideO_N = viewOut.strides[2];

    // --- Multithreading Logic ---
    // Distribute work based on total available cores (12 on Snapdragon X Elite)
    uint32_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;

    // Total work units = Batch * Rows
    uint32_t total_rows = B_count * M;
    uint32_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=, &viewA, &viewB, &viewOut]()
                             {
            uint32_t start_row = t * rows_per_thread;
            uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);

            for (uint32_t row_idx = start_row; row_idx < end_row; ++row_idx) {
                // Decompose linear row index back into batch (b) and row (m)
                uint32_t b = row_idx / M;
                uint32_t m = row_idx % M;

                const float* rowA = A_ptr + (b * strideA_B) + (m * strideA_M);
                const float* batchB = B_ptr + (b * strideB_B);
                float* rowOut = Out_ptr + (b * strideO_B) + (m * strideO_M);

                // Initialize output row to zero
                for (uint32_t n = 0; n < N; ++n) rowOut[n * strideO_N] = 0.0f;

                // IKJ Loop Order: This is the key for performance.
                // We fix one element of A and multiply it across a whole row of B.
                for (uint32_t k = 0; k < K; ++k) {
                    float valA = rowA[k * strideA_K];
                    float32x4_t vA = vdupq_n_f32(valA); // Broadcast A[b,m,k] to all 4 slots
                    
                    const float* rowB = batchB + (k * strideB_K);
                    uint32_t n = 0;

                    // SIMD loop: Process 4 columns of B at a time
                    for (; n + 4 <= N; n += 4) {
                        // Load 4 elements of B
                        // Note: Assuming strideB_N is 1 for maximum speed. 
                        // If B is not contiguous, this needs vld1q_f32 replacement.
                        float32x4_t vB = vld1q_f32(rowB + (n * strideB_N));
                        float32x4_t vOut = vld1q_f32(rowOut + (n * strideO_N));
                        
                        // Fused Multiply-Add: Out = Out + (A * B)
                        vOut = vfmaq_f32(vOut, vA, vB);
                        
                        vst1q_f32(rowOut + (n * strideO_N), vOut);
                    }

                    // Tail loop for remaining N elements
                    for (; n < N; ++n) {
                        rowOut[n * strideO_N] += valA * rowB[n * strideB_N];
                    }
                }
            } });
    }

    for (auto &thread : workers)
        thread.join();
}

inline uint32_t refFactoryDotF32_3D_Optimized(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Dot 3D requires 2 inputs");

    return graph.dot(inputs[0], inputs[1]);
}

// Register as a high-performance kernel instead of a reference kernel
REGISTER_KERNEL("Dot_F32_3D_CPU_Optimized", 2, matchDotF32_3D_Optimized, runDotF32_3D_Optimized, refFactoryDotF32_3D_Optimized, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 8, 8}, {1, 8, 8}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});

#endif // TG_HAS_NEON
