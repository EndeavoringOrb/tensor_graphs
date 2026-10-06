#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/softmax/ref.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>

#include <algorithm>
#include <cmath>

inline bool matchSoftmaxF32_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Softmax typically operates on the last dimension of a 3D tensor [Batch,
    // Seq, Hidden]
    if (inputs[0].getShape().size() != 3 || !isContiguous(output))
        return false;
    return true;
}

inline void runSoftmaxF32_NEON(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &shape = ctx.inViews[0].getShape();
    uint32_t outer_size = shape[0] * shape[1];
    uint32_t dim_size = shape[2];

    for (uint32_t i = 0; i < outer_size; ++i)
    {
        const float *r_in = in + i * dim_size;
        float *r_out = out + i * dim_size;

        // 1. Find Max for numerical stability
        float32x4_t v_max = vdupq_n_f32(-1e30f);
        uint32_t d = 0;
        for (; d + 4 <= dim_size; d += 4)
        {
            v_max = vmaxq_f32(v_max, vld1q_f32(r_in + d));
        }
        float max_val = vmaxvq_f32(v_max);
        for (; d < dim_size; ++d)
            max_val = std::max(max_val, r_in[d]);

        // 2. Compute Exp and Sum
        float sum_val = 0.0f;
        for (d = 0; d < dim_size; ++d)
        {
            float e = std::exp(r_in[d] - max_val);
            r_out[d] = e;
            sum_val += e;
        }

        // 3. Normalize
        float inv_sum = 1.0f / sum_val;
        float32x4_t v_inv_sum = vdupq_n_f32(inv_sum);
        for (d = 0; d + 4 <= dim_size; d += 4)
        {
            vst1q_f32(r_out + d, vmulq_f32(vld1q_f32(r_out + d), v_inv_sum));
        }
        for (; d < dim_size; ++d)
            r_out[d] *= inv_sum;
    }
}

/**
 * Unsimplified Ref Factory
 * This mirrors the exact structure of attention_output_atomic in main.cpp
 */


REGISTER_KERNEL("Softmax_NEON", 1, 1, matchSoftmaxF32_NEON, runSoftmaxF32_NEON, refFactorySoftmax, {0},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{4, 8, 8}}, {true},
                {{MemSpace(1, HandleType::CPP)}});

#endif