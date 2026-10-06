#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/gemma_rms_norm/ref.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchGemmaRMSNormF32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // inputs: [x, weight]
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1)
        return false;
    if (inputs[0].getShape()[2] != inputs[1].getShape()[0])
        return false;
    return isContiguous(output);
}

inline void runGemmaRMSNormF32_3D(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    const float *w = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t B = ctx.inViews[0].getShape()[0];
    const uint32_t S = ctx.inViews[0].getShape()[1];
    const uint32_t D = ctx.inViews[0].getShape()[2];
    const float eps = 1e-6f;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    uint32_t total_rows = B * S;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
        uint32_t start_row = t * rows_per_thread;
        uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);

        for (uint32_t r = start_row; r < end_row; ++r)
        {
            const float *row_x = x + r * D;
            float *row_out = out + r * D;

            // 1. Calculate Sum of Squares
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

            // 2. Calculate Inverse RMS
            float inv_std = 1.0f / std::sqrt((sum_sq / (float)D) + eps);
            float32x4_t v_inv_std = vdupq_n_f32(inv_std);
            float32x4_t v_one = vdupq_n_f32(1.0f);

            // 3. Normalize and Scale: x * inv_std * (w + 1)
            d = 0;
            for (; d + 4 <= D; d += 4)
            {
                float32x4_t v_x = vld1q_f32(row_x + d);
                float32x4_t v_w = vld1q_f32(w + d);
                float32x4_t v_scale = vaddq_f32(v_w, v_one); // (w + 1)
                float32x4_t v_norm = vmulq_f32(v_x, v_inv_std);
                vst1q_f32(row_out + d, vmulq_f32(v_norm, v_scale));
            }
            for (; d < D; ++d)
            {
                row_out[d] = row_x[d] * inv_std * (w[d] + 1.0f);
            }
        }
    });
}

// Mirror decomposition in models/gemma-3-270m.hpp


REGISTER_KERNEL("GemmaRMSNorm_F32_3D", 2, 2, matchGemmaRMSNormF32_3D, runGemmaRMSNormF32_3D, refFactoryGemmaRMSNorm,
                {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 8, 2048}, {2048}}, {true, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif