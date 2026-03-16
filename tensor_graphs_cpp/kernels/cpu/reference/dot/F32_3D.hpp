#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchDotF32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    const auto &s0 = inputs[0].shape;
    const auto &s1 = inputs[1].shape;
    const auto &so = output.shape;
    if (s0.size() != 3 || s1.size() != 3 || so.size() != 3)
        return false;
    // A: [B, M, K], B: [B, K, N], Out: [B, M, N]
    if (s0[0] != s1[0] || s0[2] != s1[1])
        return false;
    if (so[0] != s0[0] || so[1] != s0[1] || so[2] != s1[2])
        return false;
    return true;
}

inline void runDotF32_3D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *A = static_cast<const float *>(inputs[0]);
    const float *B = static_cast<const float *>(inputs[1]);
    float *Out = static_cast<float *>(outputs[0]);

    const auto &viewA = inViews[0];
    const auto &viewB = inViews[1];
    const auto &viewOut = outViews[0];

    uint32_t B_count = viewA.shape[0];
    uint32_t M = viewA.shape[1];
    uint32_t K = viewA.shape[2];
    uint32_t N = viewB.shape[2];

    // Fast Path: Check for contiguous tensors
    bool a_contig = viewA.isContiguous();
    bool b_contig = viewB.isContiguous();
    bool out_contig = viewOut.isContiguous();

    if (a_contig && b_contig && out_contig)
    {
        // Optimized contiguous loop
        // Access patterns:
        // A: moves by 1 along K (stride is 1 for last dim)
        // B: moves by N along K (stride is N for middle dim)

        for (uint32_t b = 0; b < B_count; ++b)
        {
            const float *a_batch = A + (size_t)b * M * K;
            const float *b_batch = B + (size_t)b * K * N;
            float *o_batch = Out + (size_t)b * M * N;

            for (uint32_t m = 0; m < M; ++m)
            {
                const float *a_row = a_batch + (size_t)m * K;

                for (uint32_t n = 0; n < N; ++n)
                {
                    // Prefetch or simple loop
                    const float *b_col = b_batch + n;
                    float sum = 0.0f;

                    // Compiler can auto-vectorize this easily (stride 1 for A)
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        sum += a_row[k] * b_col[(size_t)k * N];
                    }
                    o_batch[(size_t)m * N + n] = sum;
                }
            }
        }
    }
    else
    {
        // General Path: Handles non-contiguous (strided) data
        // Optimized by pre-computing offsets and incrementing pointers

        // Strides for the reduction dimension K
        // In A [B, M, K], K is index 2
        int64_t strideA_K = viewA.strides[2];
        // In B [B, K, N], K is index 1
        int64_t strideB_K = viewB.strides[1];

        for (uint32_t b = 0; b < B_count; ++b)
        {
            // Batch offsets
            size_t offset_A_batch = b * viewA.strides[0];
            size_t offset_B_batch = b * viewB.strides[0];
            size_t offset_O_batch = b * viewOut.strides[0];

            for (uint32_t m = 0; m < M; ++m)
            {
                // Row offset for A
                size_t offset_A_row = m * viewA.strides[1];

                for (uint32_t n = 0; n < N; ++n)
                {
                    // Col offset for B and Out
                    size_t offset_B_col = n * viewB.strides[2];
                    size_t offset_O_col = n * viewOut.strides[2];

                    // Compute start pointers for this specific dot product
                    const float *ptr_A = A + offset_A_batch + offset_A_row;
                    const float *ptr_B = B + offset_B_batch + offset_B_col;

                    float sum = 0.0f;

                    // Inner loop: simple pointer arithmetic, no expensive index recalculation
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        sum += (*ptr_A) * (*ptr_B);
                        ptr_A += strideA_K;
                        ptr_B += strideB_K;
                    }

                    *(Out + offset_O_batch + m * viewOut.strides[1] + offset_O_col) = sum;
                }
            }
        }
    }
}

REGISTER_REF_KERNEL(OpType::DOT, Backend::CPU, matchDotF32_3D, runDotF32_3D);