#pragma once
#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

inline bool matchKreaModulatedRMSNormV2(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1 || inputs[2].getShape().size() != 3 ||
        inputs[3].getShape().size() != 3)
        return false;

    const auto &sX = inputs[0].getShape();
    if (sX[2] != inputs[1].getShape()[0])
        return false;
    if (sX != inputs[2].getShape() || sX != inputs[3].getShape() || sX != output.getShape())
        return false;

    return isContiguous(output);
}

inline void runKreaModulatedRMSNormV2(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    const float *w = static_cast<const float *>(ctx.inputs[1]);
    const float *scale = static_cast<const float *>(ctx.inputs[2]);
    const float *shift = static_cast<const float *>(ctx.inputs[3]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t S = ctx.inViews[0].getShape()[1];
    const uint32_t D = ctx.inViews[0].getShape()[2];
    const float eps = 1e-6f;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    num_threads = std::min(num_threads, std::max(1u, S));

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t s_per_thread = (S + num_threads - 1) / num_threads;
        uint32_t s_start = t * s_per_thread;
        uint32_t s_end = std::min(s_start + s_per_thread, S);

        const uint32_t D4 = D & ~3u;

        for (uint32_t s = s_start; s < s_end; ++s)
        {
            const float *row_x = x + static_cast<uint64_t>(s) * D;
            const float *row_scale = scale + static_cast<uint64_t>(s) * D;
            const float *row_shift = shift + static_cast<uint64_t>(s) * D;
            float *row_out = out + static_cast<uint64_t>(s) * D;

            float sum_sq = 0.0f;
#if defined(TG_HAS_NEON)
            float32x4_t v_sum_sq = vdupq_n_f32(0.0f);
            for (uint32_t d = 0; d < D4; d += 4)
            {
                float32x4_t vx = vld1q_f32(row_x + d);
                v_sum_sq = vfmaq_f32(v_sum_sq, vx, vx);
            }
            sum_sq = vaddvq_f32(v_sum_sq);
#else
            for (uint32_t d = 0; d < D4; d += 4)
            {
                sum_sq += row_x[d] * row_x[d] + row_x[d + 1] * row_x[d + 1] + row_x[d + 2] * row_x[d + 2] +
                          row_x[d + 3] * row_x[d + 3];
            }
#endif
            for (uint32_t d = D4; d < D; ++d)
            {
                sum_sq += row_x[d] * row_x[d];
            }

            float mean_sq = sum_sq / static_cast<float>(D);
            float inv_std = 1.0f / std::sqrt(mean_sq + eps);

#if defined(TG_HAS_NEON)
            float32x4_t v_inv = vdupq_n_f32(inv_std);
            float32x4_t v_one = vdupq_n_f32(1.0f);

            for (uint32_t d = 0; d < D4; d += 4)
            {
                float32x4_t vx = vld1q_f32(row_x + d);
                float32x4_t vw = vld1q_f32(w + d);
                float32x4_t vscale = vld1q_f32(row_scale + d);
                float32x4_t vshift = vld1q_f32(row_shift + d);

                float32x4_t w_plus_one = vaddq_f32(vw, v_one);
                float32x4_t x_scaled = vmulq_f32(vmulq_f32(vx, v_inv), w_plus_one);
                float32x4_t one_plus_scale = vaddq_f32(v_one, vscale);
                float32x4_t res = vfmaq_f32(vshift, one_plus_scale, x_scaled);
                vst1q_f32(row_out + d, res);
            }
#else
            for (uint32_t d = 0; d < D4; d += 4)
            {
                for (int k = 0; k < 4; ++k)
                {
                    uint32_t idx = d + k;
                    float x_scaled = row_x[idx] * inv_std * (w[idx] + 1.0f);
                    row_out[idx] = (1.0f + row_scale[idx]) * x_scaled + row_shift[idx];
                }
            }
#endif
            for (uint32_t d = D4; d < D; ++d)
            {
                float x_scaled = row_x[d] * inv_std * (w[d] + 1.0f);
                row_out[d] = (1.0f + row_scale[d]) * x_scaled + row_shift[d];
            }
        }
    });
}

inline LogicalId refFactoryKreaModulatedRMSNormV2(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    LogicalId w = inputs[1];
    LogicalId scale = inputs[2];
    LogicalId shift = inputs[3];

    auto shape = g.getNode(x).getShape();
    uint32_t B = shape[0];
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    LogicalId x_sq = g.mul(x, x);
    LogicalId sum_sq = g.sum(x_sq, -1);
    LogicalId mean_sq = g.div(sum_sq, g.fill(static_cast<float>(D), {B, S, 1}));
    LogicalId std = g.pow(g.add(mean_sq, g.fill(1e-6f, {B, S, 1})), g.fill(0.5f, {B, S, 1}));
    LogicalId inv_std = g.repeat(g.div(g.fill(1.0f, {B, S, 1}), std), D, 2);
    LogicalId x_norm = g.mul(x, inv_std);

    LogicalId w_3d = g.reshape(w, {1, 1, static_cast<int32_t>(D)});
    LogicalId w_exp = g.repeat(g.repeat(w_3d, B, 0), S, 1);
    LogicalId one_full = g.fill(1.0f, {B, S, D});
    LogicalId w_scale = g.add(w_exp, one_full);
    LogicalId x_scaled = g.mul(x_norm, w_scale);

    LogicalId one = g.fill(1.0f, {B, S, D});
    LogicalId one_plus_scale = g.add(one, scale);
    LogicalId scaled_norm = g.mul(one_plus_scale, x_scaled);
    return g.add(scaled_norm, shift);
}

REGISTER_KERNEL("Fused_Krea_Modulated_RMSNorm_V2_F32", 4, 4, matchKreaModulatedRMSNormV2, runKreaModulatedRMSNormV2,
                refFactoryKreaModulatedRMSNormV2, {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
                {{1, 8, 6144}, {6144}, {1, 8, 6144}, {1, 8, 6144}}, {true, true, true, true},
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}});