#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchQwenOuterProd(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    auto sA = inputs[0].getShape();
    auto sB = inputs[1].getShape();
    if (sA.size() != 3 || sB.size() != 3)
        return false;
    if (sA[2] != 1 || sB[1] != 1 || sA[0] != sB[0])
        return false;
    if (output.getShape()[1] != sA[1] || output.getShape()[2] != sB[2])
        return false;
    return isContiguous(output);
}

inline void runQwenOuterProd(const KernelContext &ctx)
{
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    const float *B_vec = static_cast<const float *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const auto &vA = ctx.inViews[0];
    const auto &vB = ctx.inViews[1];

    uint32_t B_batch = vA.getShape()[0];
    uint32_t M = vA.getShape()[1];
    uint32_t N = vB.getShape()[2];

    uint64_t strA_B = vA.strides[0], strA_M = vA.strides[1];
    uint64_t strB_B = vB.strides[0], strB_N = vB.strides[2];
    uint64_t strO_B = ctx.outViews[0].strides[0], strO_M = ctx.outViews[0].strides[1],
             strO_N = ctx.outViews[0].strides[2];

    for (uint32_t b = 0; b < B_batch; ++b)
    {
        const float *b_row = B_vec + b * strB_B;
        for (uint32_t m = 0; m < M; ++m)
        {
            float a_val = A[b * strA_B + m * strA_M];
            float *out_row = Out + b * strO_B + m * strO_M;
            float32x4_t va = vdupq_n_f32(a_val);

            uint32_t n = 0;
            for (; n + 15 < N; n += 16)
            {
                float32x4_t vb0 = vld1q_f32(b_row + n * strB_N);
                float32x4_t vb1 = vld1q_f32(b_row + (n + 4) * strB_N);
                float32x4_t vb2 = vld1q_f32(b_row + (n + 8) * strB_N);
                float32x4_t vb3 = vld1q_f32(b_row + (n + 12) * strB_N);

                vst1q_f32(out_row + n * strO_N, vmulq_f32(va, vb0));
                vst1q_f32(out_row + (n + 4) * strO_N, vmulq_f32(va, vb1));
                vst1q_f32(out_row + (n + 8) * strO_N, vmulq_f32(va, vb2));
                vst1q_f32(out_row + (n + 12) * strO_N, vmulq_f32(va, vb3));
            }
            for (; n < N; ++n)
                out_row[n * strO_N] = a_val * b_row[n * strB_N];
        }
    }
}
inline LogicalId refQwenOuterProd(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.dot(inputs[0], inputs[1]);
}
REGISTER_KERNEL("Qwen_OuterProd_NEON", 2, 2, matchQwenOuterProd, runQwenOuterProd, refQwenOuterProd,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{32, 128, 1}, {32, 1, 128}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif