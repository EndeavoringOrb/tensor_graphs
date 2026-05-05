#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchDotF32_4D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    const auto &s0 = inputs[0].getShape();
    const auto &s1 = inputs[1].getShape();
    const auto &so = output.getShape();

    if (s0.size() != 4 || s1.size() != 4 || so.size() != 4)
        return false;

    // A: [B, H, M, K], B: [B, H, K, N], Out: [B, H, M, N]
    // Batch and Head dimensions must match
    if (s0[0] != s1[0] || s0[1] != s1[1])
        return false;
    // Reduction dimension K must match (A index 3, B index 2)
    if (s0[3] != s1[2])
        return false;

    // Output shape validation
    if (so[0] != s0[0] || so[1] != s0[1] || so[2] != s0[2] || so[3] != s1[3])
        return false;

    return true;
}

inline void runDotF32_4D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *A = static_cast<const float *>(inputs[0]);
    const float *B = static_cast<const float *>(inputs[1]);
    float *Out = static_cast<float *>(outputs[0]);

    const auto &viewA = inViews[0];
    const auto &viewB = inViews[1];
    const auto &viewOut = outViews[0];

    uint32_t B_count = viewA.getShape()[0];
    uint32_t H_count = viewA.getShape()[1];
    uint32_t M = viewA.getShape()[2];
    uint32_t K = viewA.getShape()[3];
    uint32_t N = viewB.getShape()[3];

    // Strides for the reduction dimension K
    // In A [B, H, M, K], K is index 3
    int64_t strideA_K = viewA.strides[3];
    // In B [B, H, K, N], K is index 2
    int64_t strideB_K = viewB.strides[2];

    for (uint32_t b = 0; b < B_count; ++b)
    {
        size_t offA_b = b * viewA.strides[0];
        size_t offB_b = b * viewB.strides[0];
        size_t offO_b = b * viewOut.strides[0];

        for (uint32_t h = 0; h < H_count; ++h)
        {
            size_t offA_h = h * viewA.strides[1];
            size_t offB_h = h * viewB.strides[1];
            size_t offO_h = h * viewOut.strides[1];

            for (uint32_t m = 0; m < M; ++m)
            {
                size_t offA_m = m * viewA.strides[2];
                size_t offO_m = m * viewOut.strides[2];

                for (uint32_t n = 0; n < N; ++n)
                {
                    size_t offB_n = n * viewB.strides[3];
                    size_t offO_n = n * viewOut.strides[3];

                    const float *ptr_A = A + offA_b + offA_h + offA_m;
                    const float *ptr_B = B + offB_b + offB_h + offB_n;

                    float sum = 0.0f;
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        sum += (*ptr_A) * (*ptr_B);
                        ptr_A += strideA_K;
                        ptr_B += strideB_K;
                    }

                    *(Out + offO_b + offO_h + offO_m + offO_n) = sum;
                }
            }
        }
    }
}

REGISTER_REF_KERNEL(
    OpType::DOT,
    2,
    matchDotF32_4D,
    runDotF32_4D,
    {Backend::CPU},
    {DType::FLOAT32, DType::FLOAT32},
    {{1, 1, 8, 8}, {1, 1, 8, 8}},
    {false, false},
    {{Backend::CPU}, {Backend::CPU}});