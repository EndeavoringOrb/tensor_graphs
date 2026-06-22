#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchQwenVecMat(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    auto sA = inputs[0].getShape();
    auto sB = inputs[1].getShape();
    if (sA.size() != 3 || sB.size() != 3)
        return false;
    if (sA[1] != 1 || sA[2] != sB[1] || sA[0] != sB[0])
        return false;
    if (output.getShape()[2] != sB[2])
        return false;
    return isContiguous(output);
}

inline void runQwenVecMat(const KernelContext &ctx)
{
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    const float *B = static_cast<const float *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const auto &vA = ctx.inViews[0];
    const auto &vB = ctx.inViews[1];

    uint32_t B_batch = vA.getShape()[0];
    uint32_t K = vA.getShape()[2];
    uint32_t N = vB.getShape()[2];

    uint64_t strA_B = vA.strides[0], strA_K = vA.strides[2];
    uint64_t strB_B = vB.strides[0], strB_K = vB.strides[1], strB_N = vB.strides[2];
    uint64_t strO_B = ctx.outViews[0].strides[0], strO_N = ctx.outViews[0].strides[2];

    for (uint32_t b = 0; b < B_batch; ++b)
    {
        float *out_row = Out + b * strO_B;

        uint32_t n = 0;
        float32x4_t v_zero = vdupq_n_f32(0.0f);
        for (; n + 15 < N; n += 16)
        {
            vst1q_f32(out_row + n, v_zero);
            vst1q_f32(out_row + n + 4, v_zero);
            vst1q_f32(out_row + n + 8, v_zero);
            vst1q_f32(out_row + n + 12, v_zero);
        }
        for (; n < N; ++n)
            out_row[n] = 0.0f;

        for (uint32_t k = 0; k < K; ++k)
        {
            float a_val = A[b * strA_B + k * strA_K];
            const float *b_row = B + b * strB_B + k * strB_K;
            float32x4_t va = vdupq_n_f32(a_val);

            n = 0;
            for (; n + 15 < N; n += 16)
            {
                float32x4_t vb0 = vld1q_f32(b_row + n);
                float32x4_t vb1 = vld1q_f32(b_row + n + 4);
                float32x4_t vb2 = vld1q_f32(b_row + n + 8);
                float32x4_t vb3 = vld1q_f32(b_row + n + 12);

                float32x4_t vo0 = vld1q_f32(out_row + n);
                float32x4_t vo1 = vld1q_f32(out_row + n + 4);
                float32x4_t vo2 = vld1q_f32(out_row + n + 8);
                float32x4_t vo3 = vld1q_f32(out_row + n + 12);

                vst1q_f32(out_row + n, vfmaq_f32(vo0, va, vb0));
                vst1q_f32(out_row + n + 4, vfmaq_f32(vo1, va, vb1));
                vst1q_f32(out_row + n + 8, vfmaq_f32(vo2, va, vb2));
                vst1q_f32(out_row + n + 12, vfmaq_f32(vo3, va, vb3));
            }
            for (; n < N; ++n)
                out_row[n] += a_val * b_row[n];
        }
    }
}

inline uint32_t refQwenVecMat(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.dot(inputs[0], inputs[1]);
}
REGISTER_KERNEL("Qwen_VecMat_NEON", 2, matchQwenVecMat, runQwenVecMat, refQwenVecMat, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{32, 1, 128}, {32, 128, 128}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});
#endif