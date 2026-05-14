#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchFluxRMSNormF32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1)
        return false;
    if (inputs[0].getShape()[2] != inputs[1].getShape()[0])
        return false;
    return isContiguous(output);
}

inline void runFluxRMSNormF32_3D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                 const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *x = static_cast<const float *>(inputs[0]);
    const float *w = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    const uint32_t B = inViews[0].getShape()[0];
    const uint32_t S = inViews[0].getShape()[1];
    const uint32_t D = inViews[0].getShape()[2];
    const float eps = 1e-6f;

    uint32_t num_threads = std::thread::hardware_concurrency();
    uint32_t total_rows = B * S;
    uint32_t rows_per_thread = (total_rows + num_threads - 1) / (num_threads ? num_threads : 1);

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t start_row = t * rows_per_thread;
            uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);

            for (uint32_t r = start_row; r < end_row; ++r) {
                const float* row_x = x + r * D;
                float* row_out = out + r * D;

                float32x4_t v_sum_sq = vdupq_n_f32(0.0f);
                uint32_t d = 0;
                for (; d + 4 <= D; d += 4) {
                    float32x4_t v_x = vld1q_f32(row_x + d);
                    v_sum_sq = vfmaq_f32(v_sum_sq, v_x, v_x);
                }
                float sum_sq = vaddvq_f32(v_sum_sq);
                for (; d < D; ++d) sum_sq += row_x[d] * row_x[d];

                float inv_std = 1.0f / std::sqrt((sum_sq / (float)D) + eps);
                float32x4_t v_inv_std = vdupq_n_f32(inv_std);

                d = 0;
                for (; d + 4 <= D; d += 4) {
                    float32x4_t v_x = vld1q_f32(row_x + d);
                    float32x4_t v_w = vld1q_f32(w + d);
                    float32x4_t v_norm = vmulq_f32(v_x, v_inv_std);
                    vst1q_f32(row_out + d, vmulq_f32(v_norm, v_w)); // Direct w scale
                }
                for (; d < D; ++d) {
                    row_out[d] = row_x[d] * inv_std * w[d];
                }
            } });
    }
    for (auto &worker : workers)
        worker.join();
}

inline uint32_t refFactoryFluxRMSNorm(const std::vector<uint32_t> &inputs, Graph &g)
{
    uint32_t x_id = inputs[0];
    uint32_t weight_id = inputs[1];
    auto shape = g.getNode(x_id).getShape();
    uint32_t B = shape[0], S = shape[1], D = shape[2];

    uint32_t x_sq = g.mul(x_id, x_id);
    int32_t axis_val = -1;
    uint32_t sum_sq = g.sum(x_sq, g.constant({1}, &axis_val, DType::INT32));

    auto expand = [&](float val, uint32_t last_d)
    {
        int32_t sh[] = {1, 1, 1};
        uint32_t out = g.reshape(g.constant({1}, &val, DType::FLOAT32), g.constant({3}, sh, DType::INT32));
        if (B > 1)
        {
            int32_t r = B, a = 0;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        if (S > 1)
        {
            int32_t r = S, a = 1;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        if (last_d > 1)
        {
            int32_t r = last_d, a = 2;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        return out;
    };

    uint32_t inv_std = g.div(expand(1.0f, 1), g.pow(g.add(g.div(sum_sq, expand((float)D, 1)), expand(1e-6f, 1)), expand(0.5f, 1)));

    int32_t r_d = D, a_d = 2;
    uint32_t x_norm = g.mul(x_id, g.repeat(inv_std, g.constant({1}, &r_d, DType::INT32), g.constant({1}, &a_d, DType::INT32)));

    int32_t sh_w[] = {1, 1, (int32_t)D};
    uint32_t w_exp = g.reshape(weight_id, g.constant({3}, sh_w, DType::INT32));
    if (B > 1)
    {
        int32_t r = B, a = 0;
        w_exp = g.repeat(w_exp, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }
    if (S > 1)
    {
        int32_t r = S, a = 1;
        w_exp = g.repeat(w_exp, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }

    return g.mul(x_norm, w_exp);
}

REGISTER_KERNEL("FluxRMSNorm_F32_3D", 2, matchFluxRMSNormF32_3D, runFluxRMSNormF32_3D, refFactoryFluxRMSNorm, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 8, 3072}, {3072}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});

#endif