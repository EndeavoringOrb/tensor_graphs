#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchDotF32_4D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &sA = inputs[0].getShape();
    const auto &sB = inputs[1].getShape();
    const auto &sC = output.getShape();
    if (sA.size() != 4 || sB.size() != 4 || sC.size() != 4)
        return false;
    // A: [B, H, M, K], B: [B, H, K, N], C: [B, H, M, N]
    if (sA[0] != sB[0] || sA[1] != sB[1] || sA[3] != sB[2])
        return false;
    if (sC[0] != sA[0] || sC[1] != sA[1] || sC[2] != sA[2] || sC[3] != sB[3])
        return false;
    return true;
}

inline void runDotF32_4D(const KernelContext &ctx)
{
    const float *A = static_cast<const float *>(ctx.inputs[0]); // [B, H, M, K]
    const float *B = static_cast<const float *>(ctx.inputs[1]); // [B, H, K, N]
    float *C = static_cast<float *>(ctx.outputs[0]);            // [B, H, M, N]

    const auto &viewA = ctx.inViews[0];
    const auto &viewB = ctx.inViews[1];
    const auto &viewC = ctx.outViews[0];

    uint32_t B_count = viewA.getShape()[0];
    uint32_t H = viewA.getShape()[1];
    uint32_t M = viewA.getShape()[2];
    uint32_t K = viewA.getShape()[3];
    uint32_t N = viewB.getShape()[3];

    // Strides for the reduction dimension K
    // In A [B, H, M, K], K is index 3
    uint64_t strideA_K = viewA.strides[3];
    // In B [B, H, K, N], K is index 2
    uint64_t strideB_K = viewB.strides[2];

    for (uint32_t b = 0; b < B_count; ++b)
    {
        // Batch offsets
        uint64_t offset_A_batch = b * viewA.strides[0];
        uint64_t offset_B_batch = b * viewB.strides[0];
        uint64_t offset_C_batch = b * viewC.strides[0];

        for (uint32_t h = 0; h < H; ++h)
        {
            uint64_t offset_A_head = offset_A_batch + h * viewA.strides[1];
            uint64_t offset_B_head = offset_B_batch + h * viewB.strides[1];
            uint64_t offset_C_head = offset_C_batch + h * viewC.strides[1];

            for (uint32_t m = 0; m < M; ++m)
            {
                uint64_t offset_A_row = offset_A_head + m * viewA.strides[2];
                uint64_t offset_C_row = offset_C_head + m * viewC.strides[2];

                for (uint32_t n = 0; n < N; ++n)
                {
                    uint64_t offset_B_col = n * viewB.strides[3];
                    uint64_t offset_C_col = n * viewC.strides[3];

                    const float *ptr_A = A + offset_A_row;
                    const float *ptr_B = B + offset_B_head + offset_B_col;

                    float sum = 0.0f;

                    for (uint32_t k = 0; k < K; ++k)
                    {
                        sum += (*ptr_A) * (*ptr_B);
                        ptr_A += strideA_K;
                        ptr_B += strideB_K;
                    }

                    *(C + offset_C_row + offset_C_col) = sum;
                }
            }
        }
    }
}

REGISTER_REF_KERNEL(OpType::DOT, 2, 2, matchDotF32_4D, runDotF32_4D, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32}, {{1, 8, 8}, {1, 8, 8}},
                    {true, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
