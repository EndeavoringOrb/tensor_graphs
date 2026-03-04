#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

/**
 * KERNEL: DOT F32 3D (Non-Contiguous / Stride-Aware)
 * Performs: Out[b, m, n] = sum_k (A[b, m, k] * B[b, k, n])
 *
 * This implementation uses TensorView strides to allow zero-copy execution
 * on permuted or sliced tensors.
 */

inline bool matchDotF32_3D_NonContig(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    const auto &s0 = inputs[0].shape;
    const auto &s1 = inputs[1].shape;
    const auto &so = output.shape;

    // Rank Check
    if (s0.size() != 3 || s1.size() != 3 || so.size() != 3)
        return false;

    // Dimension Compatibility Checks
    // Batch dimensions must match
    if (s0[0] != s1[0] || so[0] != s0[0])
        return false;
    // Inner K-dimension must match: A[B, M, K] @ B[B, K, N]
    if (s0[2] != s1[1])
        return false;
    // Output dimensions: C[B, M, N]
    if (so[1] != s0[1] || so[2] != s1[2])
        return false;

    // NOTE: We do NOT check for isContiguous() here, as this kernel
    // is designed to handle arbitrary strides.
    return true;
}

inline void runDotF32_3D_NonContig(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                   const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *A_base = static_cast<const float *>(inputs[0]);
    const float *B_base = static_cast<const float *>(inputs[1]);
    float *C_base = static_cast<float *>(outputs[0]);

    const auto &viewA = inViews[0];
    const auto &viewB = inViews[1];
    const auto &viewC = outViews[0];

    uint32_t Batch = viewA.shape[0];
    uint32_t M = viewA.shape[1];
    uint32_t K = viewA.shape[2];
    uint32_t N = viewB.shape[2];

    // Initialize output to zero using strides
    // (Output is usually contiguous, but we handle the general case)
    for (uint32_t b = 0; b < Batch; ++b)
    {
        for (uint32_t m = 0; m < M; ++m)
        {
            for (uint32_t n = 0; n < N; ++n)
            {
                C_base[b * viewC.strides[0] + m * viewC.strides[1] + n * viewC.strides[2]] = 0.0f;
            }
        }
    }

    // Standard Triple-Loop MatMul with Batching
    // Reordered to B, M, K, N to improve cache locality for the inner loop over B
    for (uint32_t b = 0; b < Batch; ++b)
    {
        for (uint32_t m = 0; m < M; ++m)
        {
            for (uint32_t k = 0; k < K; ++k)
            {
                // Fetch A value once for the inner-most loop
                float a_val = A_base[b * viewA.strides[0] + m * viewA.strides[1] + k * viewA.strides[2]];

                // Pointers to the start of the row in B and C for this b, m, k
                // This minimizes offset math in the tightest loop
                for (uint32_t n = 0; n < N; ++n)
                {
                    float b_val = B_base[b * viewB.strides[0] + k * viewB.strides[1] + n * viewB.strides[2]];
                    C_base[b * viewC.strides[0] + m * viewC.strides[1] + n * viewC.strides[2]] += a_val * b_val;
                }
            }
        }
    }
}

// Register as a CPU kernel for the DOT operation.
// This will compete with the reference contiguous kernel; the planner will
// choose this if it allows it to skip a PERMUTE step.
// TODO: maybe move this to tensor_graphs_cpp/kernels/cpu/fused/dot and make the reference graph explicitly permute->dot? are there any other operations that are not contiguous?
REGISTER_KERNEL(OpType::DOT, Backend::CPU, matchDotF32_3D_NonContig, runDotF32_3D_NonContig);