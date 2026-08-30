#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

inline bool matchKreaRmsNorm(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1)
        return false;
    if (inputs[0].getShape()[2] != inputs[1].getShape()[0])
        return false;
    return isContiguous(output);
}

inline void runKreaRmsNorm(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    const float *w = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &shape = ctx.inViews[0].getShape();
    const uint32_t B = shape[0];
    const uint32_t S = shape[1];
    const uint32_t D = shape[2];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    uint32_t total_rows = B * S;
    num_threads = std::min(num_threads, total_rows);

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
        uint32_t start_row = t * rows_per_thread;
        uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);

        for (uint32_t r = start_row; r < end_row; ++r)
        {
            const float *row_x = x + r * D;
            float *row_out = out + r * D;

            float sum_sq = 0.0f;
#if defined(TG_HAS_NEON)
            float32x4_t v_sum_sq = vdupq_n_f32(0.0f);
            uint32_t d = 0;
            for (; d + 4 <= D; d += 4)
            {
                float32x4_t vx = vld1q_f32(row_x + d);
                v_sum_sq = vfmaq_f32(v_sum_sq, vx, vx);
            }
            sum_sq = vaddvq_f32(v_sum_sq);
            for (; d < D; ++d)
            {
                sum_sq += row_x[d] * row_x[d];
            }
#else
            for (uint32_t d = 0; d < D; ++d)
            {
                sum_sq += row_x[d] * row_x[d];
            }
#endif
            float mean_sq = sum_sq / static_cast<float>(D);
            float inv_std = 1.0f / std::sqrt(mean_sq + 1e-6f);

#if defined(TG_HAS_NEON)
            float32x4_t v_inv = vdupq_n_f32(inv_std);
            d = 0;
            for (; d + 4 <= D; d += 4)
            {
                float32x4_t vx = vld1q_f32(row_x + d);
                float32x4_t vw = vld1q_f32(w + d);
                vst1q_f32(row_out + d, vmulq_f32(vmulq_f32(vx, v_inv), vw));
            }
            for (; d < D; ++d)
            {
                row_out[d] = row_x[d] * inv_std * w[d];
            }
#else
            for (uint32_t d = 0; d < D; ++d)
            {
                row_out[d] = row_x[d] * inv_std * w[d];
            }
#endif
        }
    });
}

inline LogicalId refFactoryKreaRmsNorm(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    LogicalId w = inputs[1];
    auto shape = g.getNode(x).getShape();
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    LogicalId x_sq = g.mul(x, x);
    LogicalId sum_sq = g.sum(x_sq, -1);
    LogicalId mean_sq = g.div(sum_sq, g.fill(static_cast<float>(D), {1, S, 1}));
    LogicalId std = g.pow(g.add(mean_sq, g.fill(1e-6f, {1, S, 1})), g.fill(0.5f, {1, S, 1}));
    LogicalId inv_std = g.repeat(g.div(g.fill(1.0f, {1, S, 1}), std), D, 2);
    LogicalId x_norm = g.mul(x, inv_std);

    LogicalId w_exp = g.repeat(g.reshape(w, {1, 1, static_cast<int32_t>(D)}), S, 1);
    return g.mul(x_norm, w_exp);
}

REGISTER_KERNEL("Krea_RMSNorm_Generic", 2, 2, matchKreaRmsNorm, runKreaRmsNorm, refFactoryKreaRmsNorm, {0},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 4224, 6144}, {6144}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});