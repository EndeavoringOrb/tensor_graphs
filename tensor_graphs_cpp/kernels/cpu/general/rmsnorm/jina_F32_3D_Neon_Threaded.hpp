// File: tensor_graphs_cpp/kernels/cpu/general/rmsnorm/jina_F32_3D_Neon_Threaded.hpp
//
// FUSED KERNEL: RMSNorm (plain, no bias) for jina-embeddings-v5-omni-nano-retrieval
//
// Matches the exact subgraph produced by JinaV5OmniNanoRetrievalModel::rms_norm():
//
//   x_sq     = mul(x, x)                              // {B, S, D}
//   sum_sq   = sum(x_sq, axis=-1)                     // {B, S, 1}
//   n_node   = expand_scalar_to_3d(D, 1, S, 1)        // {1, S, 1}  (D as float = 768.0)
//   mean_sq  = div(sum_sq, n_node)                    // {B, S, 1}
//   eps_exp  = expand_scalar_to_3d(eps, 1, S, 1)      // {1, S, 1}  (eps = 1e-5)
//   ms_eps   = add(mean_sq, eps_exp)                  // {B, S, 1}
//   sqrt_exp = expand_scalar_to_3d(0.5, 1, S, 1)      // {1, S, 1}
//   std      = pow(ms_eps, sqrt_exp)                  // {B, S, 1}
//   one_node = expand_scalar_to_3d(1.0, 1, S, 1)      // {1, S, 1}
//   inv_std  = div(one_node, std)                     // {B, S, 1}
//   inv_exp  = repeat_3d_axis(inv_std, D, 2)          // {B, S, D}
//   x_norm   = mul(x, inv_exp)                        // {B, S, D}
//   w_exp    = expand_1d_to_3d(w, D, 1, S)           // {1, S, D}
//   result   = mul(x_norm, w_exp)                     // {B, S, D}
//
// Hardware: Qualcomm aarch64 (ARMv8.6, 12 cores, NEON FMA).
// Replaces ~10+ separate kernels with a single 2-pass NEON-threaded kernel.
//
// Value constants matched byte-for-byte: D=768.0, eps=1e-5, 0.5, 1.0.
// For jina-v5 text encoder: D=768, eps=1e-5 (text_rms_eps).
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchJinaRMSNorm_F32_3D(const std::vector<TensorNode> &inputs,
                                    const TensorNode &output)
{
    // x: 3-D [B, S, D], w: 1-D [D]
    if (inputs[0].getShape().size() != 3)
        return false;
    if (inputs[1].getShape().size() != 1)
        return false;
    if (inputs[0].getShape()[2] != inputs[1].getShape()[0])
        return false;
    return isContiguous(output);
}

// 2-pass NEON-threaded RMSNorm.
//   Pass 1: sum of squares.
//   Pass 2: x * inv_std * w.
inline void runJinaRMSNorm_F32_3D(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    const float *w = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t B = ctx.inViews[0].getShape()[0];
    const uint32_t S = ctx.inViews[0].getShape()[1];
    const uint32_t D = ctx.inViews[0].getShape()[2];
    const float eps = 1e-5f;
    const float inv_D = 1.0f / (float)D;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    if (num_threads > 12)
        num_threads = 12;
    uint32_t total_rows = B * S;
    uint32_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t start_row = t * rows_per_thread;
            uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);

            for (uint32_t r = start_row; r < end_row; ++r)
            {
                const float *row_x = x + (uint64_t)r * D;
                float *row_out = out + (uint64_t)r * D;

                // --- Pass 1: sum of squares ---
                float32x4_t v_sum_sq = vdupq_n_f32(0.0f);
                uint32_t d = 0;
                for (; d + 8 <= D; d += 8)
                {
                    float32x4_t v_x0 = vld1q_f32(row_x + d);
                    float32x4_t v_x1 = vld1q_f32(row_x + d + 4);
                    v_sum_sq = vfmaq_f32(v_sum_sq, v_x0, v_x0);
                    v_sum_sq = vfmaq_f32(v_sum_sq, v_x1, v_x1);
                }
                for (; d + 4 <= D; d += 4)
                {
                    float32x4_t v_x = vld1q_f32(row_x + d);
                    v_sum_sq = vfmaq_f32(v_sum_sq, v_x, v_x);
                }
                float sum_sq = vaddvq_f32(v_sum_sq);
                for (; d < D; ++d)
                    sum_sq += row_x[d] * row_x[d];

                float mean_sq = sum_sq * inv_D;
                float inv_std = 1.0f / std::sqrt(mean_sq + eps);

                // --- Pass 2: x * inv_std * w ---
                float32x4_t v_inv_std = vdupq_n_f32(inv_std);
                d = 0;
                for (; d + 8 <= D; d += 8)
                {
                    float32x4_t v_x0 = vld1q_f32(row_x + d);
                    float32x4_t v_x1 = vld1q_f32(row_x + d + 4);
                    float32x4_t v_w0 = vld1q_f32(w + d);
                    float32x4_t v_w1 = vld1q_f32(w + d + 4);
                    float32x4_t v_n0 = vmulq_f32(v_x0, v_inv_std);
                    float32x4_t v_n1 = vmulq_f32(v_x1, v_inv_std);
                    vst1q_f32(row_out + d, vmulq_f32(v_n0, v_w0));
                    vst1q_f32(row_out + d + 4, vmulq_f32(v_n1, v_w1));
                }
                for (; d + 4 <= D; d += 4)
                {
                    float32x4_t v_x = vld1q_f32(row_x + d);
                    float32x4_t v_w = vld1q_f32(w + d);
                    float32x4_t v_n = vmulq_f32(v_x, v_inv_std);
                    vst1q_f32(row_out + d, vmulq_f32(v_n, v_w));
                }
                for (; d < D; ++d)
                    row_out[d] = row_x[d] * inv_std * w[d];
            } });
    }
    for (auto &worker : workers)
        worker.join();
}

// Reference Factory — mirrors JinaV5OmniNanoRetrievalModel::rms_norm() exactly.
// Value constants: D=768.0, eps=1e-5, 0.5, 1.0.
inline uint32_t refFactoryJinaRMSNorm_F32_3D(const std::vector<uint32_t> &inputs,
                                             Graph &g)
{
    uint32_t x_id = inputs[0];
    uint32_t w_id = inputs[1];

    const auto &shape = g.getNode(x_id).getShape();
    uint32_t B = shape[0];
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    auto expand_scalar_1S1 = [&](float val) -> uint32_t
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh[] = {1, 1, 1};
        uint32_t out = g.reshape(node, g.constant({3}, sh, DType::INT32));
        if (B > 1)
        {
            int32_t rep = (int32_t)B;
            int32_t ax = 0;
            out = g.repeat(out,
                           g.constant({1}, &rep, DType::INT32),
                           g.constant({1}, &ax, DType::INT32));
        }
        if (S > 1)
        {
            int32_t rep = (int32_t)S;
            int32_t ax = 1;
            out = g.repeat(out,
                           g.constant({1}, &rep, DType::INT32),
                           g.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    auto repeat_d_axis2 = [&](uint32_t node) -> uint32_t
    {
        int32_t rep = (int32_t)D;
        int32_t ax = 2;
        return g.repeat(node,
                        g.constant({1}, &rep, DType::INT32),
                        g.constant({1}, &ax, DType::INT32));
    };

    auto expand_1d_1SD = [&](uint32_t vec) -> uint32_t
    {
        int32_t sh[] = {1, 1, (int32_t)D};
        uint32_t out = g.reshape(vec, g.constant({3}, sh, DType::INT32));
        if (B > 1)
        {
            int32_t rep = (int32_t)B;
            int32_t ax = 0;
            out = g.repeat(out,
                           g.constant({1}, &rep, DType::INT32),
                           g.constant({1}, &ax, DType::INT32));
        }
        if (S > 1)
        {
            int32_t rep = (int32_t)S;
            int32_t ax = 1;
            out = g.repeat(out,
                           g.constant({1}, &rep, DType::INT32),
                           g.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    // x_sq = x * x
    uint32_t x_sq = g.mul(x_id, x_id);

    // sum_sq = sum(x_sq, axis=-1)
    int32_t ax_val = -1;
    uint32_t axis_node = g.constant({1}, &ax_val, DType::INT32);
    uint32_t sum_sq = g.sum(x_sq, axis_node);

    // mean_sq = sum_sq / D
    float d_float = (float)D;
    uint32_t n_node = expand_scalar_1S1(d_float);
    uint32_t mean_sq = g.div(sum_sq, n_node);

    // std = sqrt(mean_sq + eps)
    uint32_t eps_node = expand_scalar_1S1(1e-5f);
    uint32_t mean_sq_plus_eps = g.add(mean_sq, eps_node);
    uint32_t sqrt_node = expand_scalar_1S1(0.5f);
    uint32_t std = g.pow(mean_sq_plus_eps, sqrt_node);

    // inv_std = 1 / std
    uint32_t one_node = expand_scalar_1S1(1.0f);
    uint32_t inv_std = g.div(one_node, std);
    uint32_t inv_std_expanded = repeat_d_axis2(inv_std);

    // x_norm = x * inv_std
    uint32_t x_norm = g.mul(x_id, inv_std_expanded);

    // apply weight
    uint32_t w_exp = expand_1d_1SD(w_id);
    return g.mul(x_norm, w_exp);
}

REGISTER_KERNEL("JinaRMSNorm_F32_3D", 2,
                matchJinaRMSNorm_F32_3D, runJinaRMSNorm_F32_3D,
                refFactoryJinaRMSNorm_F32_3D,
                {Backend::CPU},
                {DType::FLOAT32, DType::FLOAT32},
                {{1, 1024, 768}, {768}},
                {true, true},
                {{Backend::CPU}, {Backend::CPU}});

#endif // TG_HAS_NEON
