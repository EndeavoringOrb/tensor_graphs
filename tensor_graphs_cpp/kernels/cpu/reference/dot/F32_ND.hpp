#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchDotF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &sA = inputs[0].getShape();
    const auto &sB = inputs[1].getShape();
    const auto &sC = output.getShape();
    if (sA.size() < 2 || sB.size() < 2 || sC.size() < 2)
        return false;
    if (sA.size() != sB.size() || sA.size() != sC.size())
        return false;

    size_t rank = sA.size();
    for (size_t i = 0; i < rank - 2; ++i)
    {
        if (sA[i] != sB[i] || sA[i] != sC[i])
            return false;
    }
    if (sA[rank - 1] != sB[rank - 2])
        return false;
    if (sC[rank - 2] != sA[rank - 2] || sC[rank - 1] != sB[rank - 1])
        return false;
    return true;
}

inline void runDotF32_ND(const KernelContext &ctx)
{
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    const float *B = static_cast<const float *>(ctx.inputs[1]);
    float *C = static_cast<float *>(ctx.outputs[0]);

    const auto &viewA = ctx.inViews[0];
    const auto &viewB = ctx.inViews[1];
    const auto &viewC = ctx.outViews[0];

    size_t rank = viewA.getShape().size();
    uint32_t M = viewA.getShape()[rank - 2];
    uint32_t K = viewA.getShape()[rank - 1];
    uint32_t N = viewB.getShape()[rank - 1];

    uint64_t outer_iters = 1;
    for (size_t i = 0; i < rank - 2; ++i)
    {
        outer_iters *= viewA.getShape()[i];
    }

    uint64_t strideA_M = viewA.strides[rank - 2];
    uint64_t strideA_K = viewA.strides[rank - 1];
    uint64_t strideB_K = viewB.strides[rank - 2];
    uint64_t strideB_N = viewB.strides[rank - 1];
    uint64_t strideC_M = viewC.strides[rank - 2];
    uint64_t strideC_N = viewC.strides[rank - 1];

    for (uint64_t o = 0; o < outer_iters; ++o)
    {
        uint64_t offsetA = 0, offsetB = 0, offsetC = 0;
        uint64_t temp = o;
        for (int i = static_cast<int>(rank) - 3; i >= 0; --i)
        {
            uint64_t coord = temp % viewA.getShape()[i];
            temp /= viewA.getShape()[i];
            offsetA += coord * viewA.strides[i];
            offsetB += coord * viewB.strides[i];
            offsetC += coord * viewC.strides[i];
        }

        for (uint32_t m = 0; m < M; ++m)
        {
            uint64_t offset_A_row = offsetA + m * strideA_M;
            uint64_t offset_C_row = offsetC + m * strideC_M;

            for (uint32_t n = 0; n < N; ++n)
            {
                uint64_t offset_B_col = offsetB + n * strideB_N;
                uint64_t offset_C_col = offset_C_row + n * strideC_N;

                const float *ptr_A = A + offset_A_row;
                const float *ptr_B = B + offset_B_col;

                float sum = 0.0f;
                for (uint32_t k = 0; k < K; ++k)
                {
                    sum += (*ptr_A) * (*ptr_B);
                    ptr_A += strideA_K;
                    ptr_B += strideB_K;
                }

                C[offset_C_col] = sum;
            }
        }
    }
}

REGISTER_REF_KERNEL(OpType::DOT, 2, 2, matchDotF32_ND, runDotF32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32}, {{1, 8, 8}, {1, 8, 8}},
                    {true, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
