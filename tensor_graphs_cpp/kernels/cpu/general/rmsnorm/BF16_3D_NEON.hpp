#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchRMSNormBF16_3D_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::BF16 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1)
        return false;
    if (inputs[0].getShape()[2] != inputs[1].getShape()[0])
        return false;
    if (output.getShape() != inputs[0].getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runRMSNormBF16_3D_NEON(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *x = static_cast<const float *>(inputs[0]);
    const uint16_t *w = static_cast<const uint16_t *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint32_t B = inViews[0].getShape()[0];
    uint32_t S = inViews[0].getShape()[1];
    uint32_t D = inViews[0].getShape()[2];

    float eps = 1e-6f;

    // Fast path to avoid heap allocation for common hidden sizes (covers your D=640)
    float w_f32_buf[1024];
    float *w_f32 = w_f32_buf;
    std::vector<float> w_f32_vec;
    if (D > 1024)
    {
        w_f32_vec.resize(D);
        w_f32 = w_f32_vec.data();
    }

    // Pre-calculate Weight conversion + bias once across all loops
    uint32_t d_w = 0;
    float32x4_t ones = vdupq_n_f32(1.0f);
    for (; d_w + 7 < D; d_w += 8)
    {
        uint16x8_t w_u16 = vld1q_u16(w + d_w);

        // Logical shift left 16 maps bfloat16 into correct float32 bit placements
        uint32x4_t w_low = vshll_n_u16(vget_low_u16(w_u16), 16);
        uint32x4_t w_high = vshll_n_u16(vget_high_u16(w_u16), 16);

        float32x4_t f_low = vreinterpretq_f32_u32(w_low);
        float32x4_t f_high = vreinterpretq_f32_u32(w_high);

        f_low = vaddq_f32(f_low, ones);
        f_high = vaddq_f32(f_high, ones);

        vst1q_f32(w_f32 + d_w, f_low);
        vst1q_f32(w_f32 + d_w + 4, f_high);
    }

    int64_t num_rows = static_cast<int64_t>(B) * S;

#pragma omp parallel for
    for (int64_t i = 0; i < num_rows; ++i)
    {
        const float *x_row = x + i * D;
        float *out_row = out + i * D;

        float sum_sq = 0.0f;
        uint32_t d2 = 0;

        float32x4_t v_sum0 = vdupq_n_f32(0.0f);
        float32x4_t v_sum1 = vdupq_n_f32(0.0f);
        float32x4_t v_sum2 = vdupq_n_f32(0.0f);
        float32x4_t v_sum3 = vdupq_n_f32(0.0f);

        for (; d2 + 15 < D; d2 += 16)
        {
            float32x4_t v_x0 = vld1q_f32(x_row + d2);
            float32x4_t v_x1 = vld1q_f32(x_row + d2 + 4);
            float32x4_t v_x2 = vld1q_f32(x_row + d2 + 8);
            float32x4_t v_x3 = vld1q_f32(x_row + d2 + 12);

            v_sum0 = vfmaq_f32(v_sum0, v_x0, v_x0);
            v_sum1 = vfmaq_f32(v_sum1, v_x1, v_x1);
            v_sum2 = vfmaq_f32(v_sum2, v_x2, v_x2);
            v_sum3 = vfmaq_f32(v_sum3, v_x3, v_x3);
        }
        v_sum0 = vaddq_f32(v_sum0, v_sum1);
        v_sum2 = vaddq_f32(v_sum2, v_sum3);
        v_sum0 = vaddq_f32(v_sum0, v_sum2);

        sum_sq += vaddvq_f32(v_sum0); // Native Aarch64 horizontal aggregate reduce

        float mean_sq = sum_sq / D;
        float inv_std = 1.0f / std::sqrt(mean_sq + eps);

        uint32_t d3 = 0;
        float32x4_t v_inv_std = vdupq_n_f32(inv_std);

        for (; d3 + 15 < D; d3 += 16)
        {
            float32x4_t v_x0 = vld1q_f32(x_row + d3);
            float32x4_t v_x1 = vld1q_f32(x_row + d3 + 4);
            float32x4_t v_x2 = vld1q_f32(x_row + d3 + 8);
            float32x4_t v_x3 = vld1q_f32(x_row + d3 + 12);

            float32x4_t v_w0 = vld1q_f32(w_f32 + d3);
            float32x4_t v_w1 = vld1q_f32(w_f32 + d3 + 4);
            float32x4_t v_w2 = vld1q_f32(w_f32 + d3 + 8);
            float32x4_t v_w3 = vld1q_f32(w_f32 + d3 + 12);

            float32x4_t v_scale0 = vmulq_f32(v_inv_std, v_w0);
            float32x4_t v_scale1 = vmulq_f32(v_inv_std, v_w1);
            float32x4_t v_scale2 = vmulq_f32(v_inv_std, v_w2);
            float32x4_t v_scale3 = vmulq_f32(v_inv_std, v_w3);

            float32x4_t v_out0 = vmulq_f32(v_x0, v_scale0);
            float32x4_t v_out1 = vmulq_f32(v_x1, v_scale1);
            float32x4_t v_out2 = vmulq_f32(v_x2, v_scale2);
            float32x4_t v_out3 = vmulq_f32(v_x3, v_scale3);

            vst1q_f32(out_row + d3, v_out0);
            vst1q_f32(out_row + d3 + 4, v_out1);
            vst1q_f32(out_row + d3 + 8, v_out2);
            vst1q_f32(out_row + d3 + 12, v_out3);
        }
    }
}

inline uint32_t ref_rms_bf16_broadcast_scalar_NEON(Graph &g, uint32_t scalar_id, uint32_t B, uint32_t S, uint32_t D, bool full_d = false)
{
    int32_t shape_3d[] = {1, 1, 1};
    uint32_t out = g.reshape(scalar_id, g.constant({3}, shape_3d, DType::INT32));

    if (B > 1)
    {
        int32_t b_rep = (int32_t)B;
        int32_t b_ax = 0;
        out = g.repeat(out, g.constant({1}, &b_rep, DType::INT32), g.constant({1}, &b_ax, DType::INT32));
    }
    if (S > 1)
    {
        int32_t s_rep = (int32_t)S;
        int32_t s_ax = 1;
        out = g.repeat(out, g.constant({1}, &s_rep, DType::INT32), g.constant({1}, &s_ax, DType::INT32));
    }
    if (full_d && D > 1)
    {
        int32_t d_rep = (int32_t)D;
        int32_t d_ax = 2;
        out = g.repeat(out, g.constant({1}, &d_rep, DType::INT32), g.constant({1}, &d_ax, DType::INT32));
    }
    return out;
}

inline uint32_t refFactoryRMSNormBF16_NEON(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t x_id = inputs[0];
    uint32_t weight_id_bf16 = inputs[1];

    uint32_t weight_id = graph.cast(weight_id_bf16, DType::FLOAT32);

    auto shapeX = graph.getNode(x_id).getShape();
    uint32_t B = shapeX[0];
    uint32_t S = shapeX[1];
    uint32_t D = shapeX[2];

    uint32_t x_sq = graph.mul(x_id, x_id);
    int32_t axis_val = -1;
    uint32_t axis_node = graph.constant({1}, &axis_val, DType::INT32);
    uint32_t sum_sq = graph.sum(x_sq, axis_node);

    float d_float = (float)D;
    uint32_t n_node = ref_rms_bf16_broadcast_scalar_NEON(graph, graph.constant({1}, &d_float, DType::FLOAT32), B, S, 1, false);
    uint32_t mean_sq = graph.div(sum_sq, n_node);

    float eps = 1e-6f;
    uint32_t eps_expanded = ref_rms_bf16_broadcast_scalar_NEON(graph, graph.constant({1}, &eps, DType::FLOAT32), B, S, 1, false);
    uint32_t mean_sq_plus_eps = graph.add(mean_sq, eps_expanded);

    float half_val = 0.5f;
    uint32_t sqrt_exp = ref_rms_bf16_broadcast_scalar_NEON(graph, graph.constant({1}, &half_val, DType::FLOAT32), B, S, 1, false);
    uint32_t std_dev = graph.pow(mean_sq_plus_eps, sqrt_exp);

    float one_val = 1.0f;
    uint32_t one_node = ref_rms_bf16_broadcast_scalar_NEON(graph, graph.constant({1}, &one_val, DType::FLOAT32), B, S, 1, false);
    uint32_t inv_std = graph.div(one_node, std_dev);

    int32_t d_rep = (int32_t)D;
    int32_t d_ax = 2;
    uint32_t inv_std_expanded = graph.repeat(inv_std, graph.constant({1}, &d_rep, DType::INT32), graph.constant({1}, &d_ax, DType::INT32));
    uint32_t x_norm = graph.mul(x_id, inv_std_expanded);

    int32_t reshape_dims[] = {1, 1, (int32_t)D};
    uint32_t w_reshaped = graph.reshape(weight_id, graph.constant({3}, reshape_dims, DType::INT32));

    uint32_t w_expanded = w_reshaped;
    if (B > 1)
    {
        int32_t b_rep = (int32_t)B;
        int32_t b_ax = 0;
        w_expanded = graph.repeat(w_expanded, graph.constant({1}, &b_rep, DType::INT32), graph.constant({1}, &b_ax, DType::INT32));
    }
    if (S > 1)
    {
        int32_t s_rep = (int32_t)S;
        int32_t s_ax = 1;
        w_expanded = graph.repeat(w_expanded, graph.constant({1}, &s_rep, DType::INT32), graph.constant({1}, &s_ax, DType::INT32));
    }

    uint32_t one_full = ref_rms_bf16_broadcast_scalar_NEON(graph, graph.constant({1}, &one_val, DType::FLOAT32), B, S, D, true);
    uint32_t scale = graph.add(w_expanded, one_full);

    return graph.mul(x_norm, scale);
}

REGISTER_KERNEL("RMSNorm_BF16_B1_NEON", 2, matchRMSNormBF16_3D_NEON, runRMSNormBF16_3D_NEON, refFactoryRMSNormBF16_NEON, {Backend::CPU}, {DType::FLOAT32, DType::BF16}, {{1, 8, 640}, {640}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});

#endif