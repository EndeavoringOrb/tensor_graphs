#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/flux_rms_norm_4d/ref.hpp"
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


REGISTER_KERNEL("FluxRMSNorm_F32_4D", 2, 2, matchFluxRMSNormF32_4D, runFluxRMSNormF32_4D, refFactoryFluxRMSNorm4D, {0},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 24, 512, 128}, {128}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif