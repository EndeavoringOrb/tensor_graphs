#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <thread>
#include <vector>
#include <algorithm>
#include <cmath>
#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchApplyRoPE(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 3)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || inputs[2].dtype != DType::FLOAT32)
        return false;

    auto sX = inputs[0].getShape();
    auto sCos = inputs[1].getShape();
    auto sSin = inputs[2].getShape();

    if (sX.size() != 3 || sCos.size() != 3 || sSin.size() != 3)
        return false;
    if (sX != sCos || sX != sSin || sX != output.getShape())
        return false;

    return isContiguous(output);
}

inline void runApplyRoPE(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *X = static_cast<const float *>(inputs[0]);
    const float *Cos = static_cast<const float *>(inputs[1]);
    const float *Sin = static_cast<const float *>(inputs[2]);
    float *Out = static_cast<float *>(outputs[0]);

    uint32_t B = inViews[0].getShape()[0];
    uint32_t S = inViews[0].getShape()[1];
    uint32_t D = inViews[0].getShape()[2];
    uint32_t half_D = D / 2;

    for (uint32_t b = 0; b < B; ++b)
    {
        for (uint32_t s = 0; s < S; ++s)
        {
            const float *x_row = X + b * S * D + s * D;
            const float *cos_row = Cos + b * S * D + s * D;
            const float *sin_row = Sin + b * S * D + s * D;
            float *out_row = Out + b * S * D + s * D;

            uint32_t d = 0;
            for (; d + 4 <= half_D; d += 4)
            {
                float32x4_t x1 = vld1q_f32(x_row + d);
                float32x4_t x2 = vld1q_f32(x_row + d + half_D);

                float32x4_t c1 = vld1q_f32(cos_row + d);
                float32x4_t c2 = vld1q_f32(cos_row + d + half_D);

                float32x4_t s1 = vld1q_f32(sin_row + d);
                float32x4_t s2 = vld1q_f32(sin_row + d + half_D);

                float32x4_t out1 = vmlsq_f32(vmulq_f32(x1, c1), x2, s1);
                float32x4_t out2 = vmlaq_f32(vmulq_f32(x2, c2), x1, s2);

                vst1q_f32(out_row + d, out1);
                vst1q_f32(out_row + d + half_D, out2);
            }

            for (; d < half_D; ++d)
            {
                float x1 = x_row[d];
                float x2 = x_row[d + half_D];

                float c1 = cos_row[d];
                float c2 = cos_row[d + half_D];

                float s1 = sin_row[d];
                float s2 = sin_row[d + half_D];

                out_row[d] = x1 * c1 - x2 * s1;
                out_row[d + half_D] = x2 * c2 + x1 * s2;
            }
        }
    }
}

inline uint32_t refFactoryApplyRoPE(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t x_id = inputs[0];
    uint32_t cos_id = inputs[1];
    uint32_t sin_id = inputs[2];

    auto sX = graph.getNode(x_id).getShape();
    uint32_t B = sX[0];
    uint32_t S = sX[1];
    uint32_t D = sX[2];

    int32_t starts1[] = {0, 0, 0};
    int32_t ends1[] = {(int32_t)B, (int32_t)S, (int32_t)D / 2};
    int32_t steps1[] = {1, 1, 1};

    uint32_t x1 = graph.slice(x_id,
                              graph.constant({3}, starts1, DType::INT32),
                              graph.constant({3}, ends1, DType::INT32),
                              graph.constant({3}, steps1, DType::INT32));
    x1 = graph.contiguous(x1);

    int32_t starts2[] = {0, 0, (int32_t)D / 2};
    int32_t ends2[] = {(int32_t)B, (int32_t)S, (int32_t)D};

    uint32_t x2 = graph.slice(x_id,
                              graph.constant({3}, starts2, DType::INT32),
                              graph.constant({3}, ends2, DType::INT32),
                              graph.constant({3}, steps1, DType::INT32));

    uint32_t neg_x2 = graph.neg(x2);

    int32_t axis = 2;
    uint32_t rotated = graph.concat({neg_x2, x1}, graph.constant({1}, &axis, DType::INT32));

    uint32_t term1 = graph.mul(x_id, cos_id);
    uint32_t term2 = graph.mul(rotated, sin_id);

    return graph.add(term1, term2);
}

REGISTER_KERNEL("Apply_RoPE_NEON", 3, matchApplyRoPE, runApplyRoPE, refFactoryApplyRoPE, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32}, {{1, 8, 256}, {1, 8, 256}, {1, 8, 256}}, {true, true, true}, {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}});
#endif