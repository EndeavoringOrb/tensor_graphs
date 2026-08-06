#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

/**
 * FUSED KERNEL: Flux 4D RMSNorm (NEON + Threaded)
 *
 * Matches the pattern in FluxTransformer::rms_norm_atomic:
 * sq = x * x
 * sum_sq = sum(sq, axis=-1)
 * mean_sq = sum_sq / head_dim
 * std = sqrt(mean_sq + 1e-6)
 * inv_std = 1.0 / std
 * out = (x * inv_std) * weight
 */

inline bool matchFluxRMSNormF32_4D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // inputs: [x, weight]
    // x: [Batch, Heads, Seq, HeadDim], weight: [HeadDim]
    if (inputs[0].getShape().size() != 4 || inputs[1].getShape().size() != 1)
        return false;

    // The last dimension of x must match the weight dimension
    if (inputs[0].getShape()[3] != inputs[1].getShape()[0])
        return false;

    return isContiguous(output);
}

inline void runFluxRMSNormF32_4D(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    const float *w = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t B = ctx.inViews[0].getShape()[0];
    const uint32_t H = ctx.inViews[0].getShape()[1];
    const uint32_t S = ctx.inViews[0].getShape()[2];
    const uint32_t D = ctx.inViews[0].getShape()[3];
    const float eps = 1e-6f;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    uint32_t total_rows = B * H * S;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
        uint32_t start_row = t * rows_per_thread;
        uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);

        for (uint32_t r = start_row; r < end_row; ++r)
        {
            const float *row_x = x + r * D;
            float *row_out = out + r * D;

            // 1. Sum of squares using NEON
            float32x4_t v_sum_sq = vdupq_n_f32(0.0f);
            uint32_t d = 0;
            for (; d + 4 <= D; d += 4)
            {
                float32x4_t v_x = vld1q_f32(row_x + d);
                v_sum_sq = vfmaq_f32(v_sum_sq, v_x, v_x);
            }
            float sum_sq = vaddvq_f32(v_sum_sq);
            for (; d < D; ++d)
                sum_sq += row_x[d] * row_x[d];

            // 2. Inverse Standard Deviation
            float inv_std = 1.0f / std::sqrt((sum_sq / (float)D) + eps);
            float32x4_t v_inv_std = vdupq_n_f32(inv_std);

            // 3. Normalize and scale by weight
            d = 0;
            for (; d + 4 <= D; d += 4)
            {
                float32x4_t v_x = vld1q_f32(row_x + d);
                float32x4_t v_w = vld1q_f32(w + d);
                float32x4_t v_norm = vmulq_f32(v_x, v_inv_std);
                vst1q_f32(row_out + d, vmulq_f32(v_norm, v_w));
            }
            for (; d < D; ++d)
            {
                row_out[d] = row_x[d] * inv_std * w[d];
            }
        }
    });
}

/**
 * Reference Factory
 * Replicates the exact subgraph structure of FluxTransformer::rms_norm_atomic
 */
inline LogicalId refFactoryFluxRMSNorm4D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x_id = inputs[0];
    LogicalId weight_id = inputs[1];
    auto shape = g.getNode(x_id).getShape();
    uint32_t B = shape[0], H = shape[1], S = shape[2], D = shape[3];

    // sq = x * x
    LogicalId x_sq = g.mul(x_id, x_id);

    // sum_sq = sum(sq, axis=-1)
    int32_t axis_val = -1;
    LogicalId axis_node = g.constant({1}, &axis_val, DType::INT32);
    LogicalId sum_sq = g.sum(x_sq, axis_node);

    auto expand_to_4d_broadcast = [&](float val, uint32_t last_d) {
        int32_t sh[] = {1, 1, 1, 1};
        LogicalId out = g.reshape(g.constant({1}, &val, DType::FLOAT32), g.constant({4}, sh, DType::INT32));
        if (B > 1)
        {
            int32_t r = B, a = 0;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        if (H > 1)
        {
            int32_t r = H, a = 1;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        if (S > 1)
        {
            int32_t r = S, a = 2;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        if (last_d > 1)
        {
            int32_t r = last_d, a = 3;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        return out;
    };

    // mean_sq = sum_sq / HeadDim
    LogicalId head_dim_const = expand_to_4d_broadcast((float)D, 1);
    LogicalId mean_sq = g.div(sum_sq, head_dim_const);

    // std = pow(mean_sq + 1e-6, 0.5)
    LogicalId eps_node = expand_to_4d_broadcast(1e-6f, 1);
    LogicalId half_node = expand_to_4d_broadcast(0.5f, 1);
    LogicalId std_dev = g.pow(g.add(mean_sq, eps_node), half_node);

    // inv_std = 1.0 / std_dev (repeated across D)
    LogicalId one_node = expand_to_4d_broadcast(1.0f, 1);
    LogicalId inv_std_scalar = g.div(one_node, std_dev);
    int32_t r_d = D, a_d = 3;
    LogicalId inv_std =
        g.repeat(inv_std_scalar, g.constant({1}, &r_d, DType::INT32), g.constant({1}, &a_d, DType::INT32));

    // x_norm = x * inv_std
    LogicalId x_norm = g.mul(x_id, inv_std);

    // w_exp = reshape and repeat weight to match [B, H, S, D]
    int32_t sh_w[] = {1, 1, 1, (int32_t)D};
    LogicalId w_exp = g.reshape(weight_id, g.constant({4}, sh_w, DType::INT32));
    if (H > 1)
    {
        int32_t r = H, a = 1;
        w_exp = g.repeat(w_exp, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }
    if (S > 1)
    {
        int32_t r = S, a = 2;
        w_exp = g.repeat(w_exp, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }
    // Batch repeat usually handled by strides/0-stride view, but for ref pattern
    // compatibility:
    if (B > 1)
    {
        int32_t r = B, a = 0;
        w_exp = g.repeat(w_exp, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }

    return g.mul(x_norm, w_exp);
}

REGISTER_KERNEL("FluxRMSNorm_F32_4D", 2, 2, matchFluxRMSNormF32_4D, runFluxRMSNormF32_4D, refFactoryFluxRMSNorm4D, {0},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 24, 512, 128}, {128}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif