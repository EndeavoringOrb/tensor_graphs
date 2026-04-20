#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <cmath>
#include <algorithm>

inline bool matchSoftmaxF32_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    // Softmax typically operates on the last dimension of a 3D tensor [Batch, Seq, Hidden]
    if (inputs[0].getShape().size() != 3 || !isContiguous(inputs[0]) || !isContiguous(output))
        return false;
    return true;
}

inline void runSoftmaxF32_NEON(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                               const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    float *out = static_cast<float *>(outputs[0]);

    const auto &shape = inViews[0].getShape();
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
inline uint32_t refFactorySoftmax(const std::vector<uint32_t> &inputs, Graph &g)
{
    uint32_t x = inputs[0]; // The scores tensor [Heads, Seq, Seq]
    auto shape = g.getNode(x).getShape();
    uint32_t H = shape[0];
    uint32_t S = shape[1];

    // --- Part 1: Safe Softmax Shift (Max reduction) ---
    int32_t axis_val = -1;
    uint32_t axis_node = g.constant({1}, &axis_val, DType::INT32);
    uint32_t max_scores = g.max(x, axis_node);

    // repeat_3d_axis(max_scores, seq_len, 2)
    int32_t s_rep_val = (int32_t)S;
    uint32_t s_rep_node = g.constant({1}, &s_rep_val, DType::INT32);
    int32_t ax2_val = 2;
    uint32_t ax2_node = g.constant({1}, &ax2_val, DType::INT32);
    uint32_t max_expanded = g.repeat(max_scores, s_rep_node, ax2_node);

    uint32_t shifted_scores = g.add(x, g.neg(max_expanded));

    // --- Part 2: Exponentiate (expand_scalar_to_3d for e_node) ---
    float e_val = 2.718281828459045f;
    uint32_t e_scalar = g.constant({1}, &e_val, DType::FLOAT32);
    int32_t shape_3d_const[] = {1, 1, 1};
    uint32_t e_reshaped = g.reshape(e_scalar, g.constant({3}, shape_3d_const, DType::INT32));

    uint32_t e_node = e_reshaped;
    if (H > 1)
    {
        int32_t h_rep = (int32_t)H;
        int32_t ax0 = 0;
        e_node = g.repeat(e_node, g.constant({1}, &h_rep, DType::INT32), g.constant({1}, &ax0, DType::INT32));
    }
    if (S > 1)
    {
        int32_t s_rep = (int32_t)S;
        int32_t ax1 = 1;
        e_node = g.repeat(e_node, g.constant({1}, &s_rep, DType::INT32), g.constant({1}, &ax1, DType::INT32));
    }
    if (S > 1)
    { // Third dimension expansion
        int32_t s_rep = (int32_t)S;
        int32_t ax2 = 2;
        e_node = g.repeat(e_node, g.constant({1}, &s_rep, DType::INT32), g.constant({1}, &ax2, DType::INT32));
    }

    uint32_t exp_scores = g.pow(e_node, shifted_scores);

    // --- Part 3: Normalize (Sum reduction) ---
    uint32_t sum_exp = g.sum(exp_scores, g.constant({1}, &axis_val, DType::INT32));

    // repeat_3d_axis(sum_exp, seq_len, 2)
    uint32_t sum_exp_expanded = g.repeat(sum_exp, s_rep_node, ax2_node);

    return g.div(exp_scores, sum_exp_expanded);
}

REGISTER_KERNEL("Softmax_NEON", 1, matchSoftmaxF32_NEON, runSoftmaxF32_NEON, refFactorySoftmax, {Backend::CPU}, {DType::FLOAT32}, {{4, 8, 8}}, {true}, {{Backend::CPU}});

#endif