#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

/**
 * KERNEL: DOT F32 3D (Batched MatMul)
 * Performs: Out[b, m, n] = sum_k (A[b, m, k] * B[b, k, n])
 */

inline bool matchDotF32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2) return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    const auto &s0 = inputs[0].shape;
    const auto &s1 = inputs[1].shape;
    const auto &so = output.shape;

    if (s0.size() != 3 || s1.size() != 3 || so.size() != 3) return false;
    
    // Batch and K-dim checks
    if (s0[0] != s1[0] || s0[2] != s1[1]) return false;
    // Output shape checks
    if (so[0] != s0[0] || so[1] != s0[1] || so[2] != s1[2]) return false;

    if (!inputs[0].view.isContiguous() || !inputs[1].view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

inline void runDotF32_3D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *A = static_cast<const float *>(inputs[0]);
    const float *B = static_cast<const float *>(inputs[1]);
    float *Out = static_cast<float *>(outputs[0]);

    uint32_t B_count = inViews[0].shape[0];
    uint32_t M = inViews[0].shape[1];
    uint32_t K = inViews[0].shape[2];
    uint32_t N = inViews[1].shape[2];

    // Initialize output to zero
    uint64_t totalOut = (uint64_t)B_count * M * N;
    for (uint64_t i = 0; i < totalOut; ++i) Out[i] = 0.0f;

    for (uint32_t b = 0; b < B_count; ++b) {
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t k = 0; k < K; ++k) {
                float a_val = A[b * M * K + m * K + k];
                for (uint32_t n = 0; n < N; ++n) {
                    Out[b * M * N + m * N + n] += a_val * B[b * K * N + k * N + n];
                }
            }
        }
    }
}

REGISTER_REF_KERNEL(OpType::DOT, Backend::CPU, matchDotF32_3D, runDotF32_3D);