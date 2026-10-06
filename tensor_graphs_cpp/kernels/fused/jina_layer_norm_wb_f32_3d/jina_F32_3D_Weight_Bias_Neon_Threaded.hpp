
#include "kernels/fused/jina_layer_norm_wb_f32_3d/ref.hpp"
//
// FUSED KERNEL: LayerNorm (with weight + bias) for
// jina-embeddings-v5-omni-nano-retrieval
//
// Matches the exact subgraph produced by
// JinaV5OmniNanoRetrievalModel::layer_norm():
//
//   ax_node  = constant(-1, INT32)
//   sum_x    = sum(x, ax_node)                        // {B, S, 1}
//   d_node   = expand_scalar_to_3d(D, 1, S, 1)        // {1, S, 1}  (D as
//   float, e.g. 768.0) mean_val = div(sum_x, d_node)                     // {B,
//   S, 1} mean     = repeat_3d_axis(mean_val, D, 2)         // {B, S, D} x_sub
//   = add(x, neg(mean))                      // {B, S, D} sq       = mul(x_sub,
//   x_sub)                      // {B, S, D} sum_sq   = sum(sq, ax_node) // {B,
//   S, 1} var      = div(sum_sq, d_node)                    // {B, S, 1}
//   eps_node = expand_scalar_to_3d(eps, 1, S, 1)      // {1, S, 1}  (eps =
//   1e-6) var_eps  = add(var, eps_node)                     // {B, S, 1}
//   sqrt_exp = expand_scalar_to_3d(0.5, 1, S, 1)      // {1, S, 1}
//   std_dev  = pow(var_eps, sqrt_exp)                 // {B, S, 1}
//   one_node = expand_scalar_to_3d(1.0, 1, S, 1)      // {1, S, 1}
//   inv_std  = div(one_node, std_dev)                 // {B, S, 1}
//   inv_exp  = repeat_3d_axis(inv_std, D, 2)          // {B, S, D}
//   norm     = mul(x_sub, inv_exp)                    // {B, S, D}
//   w_exp    = expand_1d_to_3d(w, D, 1, S)           // {1, S, D}
//   norm     = mul(norm, w_exp)                       // {B, S, D}
//   b_exp    = expand_1d_to_3d(b, D, 1, S)           // {1, S, D}
//   result   = add(norm, b_exp)                       // {B, S, D}
//
// Hardware: Qualcomm aarch64 (ARMv8.6, 12 cores, NEON FMA).
// This kernel replaces ~14+ separate elementwise / reduction / broadcast
// kernels with a single 2-pass NEON-threaded kernel.
//
// The value constants (D as float, eps, 0.5, 1.0) are matched byte-for-byte
// by the e-graph isomorphism check, so this kernel only fires on subgraphs
// that use the exact same constants.  For jina-v5: D=768, eps=1e-6.
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

// ---------------------------------------------------------------------------
// Match function — only structural shape checks (linter-friendly).
// ---------------------------------------------------------------------------
inline bool matchJinaLayerNormWB_F32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // x: 3-D [B, S, D], w: 1-D [D], b: 1-D [D]
    if (inputs[0].getShape().size() != 3)
        return false;
    if (inputs[1].getShape().size() != 1)
        return false;
    if (inputs[2].getShape().size() != 1)
        return false;
    if (inputs[0].getShape()[2] != inputs[1].getShape()[0])
        return false;
    if (inputs[0].getShape()[2] != inputs[2].getShape()[0])
        return false;
    return isContiguous(output);
}

// ---------------------------------------------------------------------------
// Run function — 2-pass NEON-threaded LayerNorm.
//   Pass 1: compute mean and variance (one pass, sum + sum_sq).
//   Pass 2: (x - mean) * inv_std * w + b.
// ---------------------------------------------------------------------------
inline void runJinaLayerNormWB_F32_3D(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    const float *w = static_cast<const float *>(ctx.inputs[1]);
    const float *b = static_cast<const float *>(ctx.inputs[2]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t B = ctx.inViews[0].getShape()[0];
    const uint32_t S = ctx.inViews[0].getShape()[1];
    const uint32_t D = ctx.inViews[0].getShape()[2];
    const float eps = 1e-6f;
    const float inv_D = 1.0f / (float)D;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    if (num_threads > 12)
        num_threads = 12; // cap to physical cores
    uint32_t total_rows = B * S;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
        uint32_t start_row = t * rows_per_thread;
        uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);

        for (uint32_t r = start_row; r < end_row; ++r)
        {
            const float *row_x = x + (uint64_t)r * D;
            float *row_out = out + (uint64_t)r * D;

            // --- Pass 1: compute mean (numerically stable two-pass) ---
            float32x4_t v_sum = vdupq_n_f32(0.0f);
            uint32_t d = 0;
            for (; d + 8 <= D; d += 8)
            {
                float32x4_t v_x0 = vld1q_f32(row_x + d);
                float32x4_t v_x1 = vld1q_f32(row_x + d + 4);
                v_sum = vaddq_f32(v_sum, v_x0);
                v_sum = vaddq_f32(v_sum, v_x1);
            }
            for (; d + 4 <= D; d += 4)
            {
                v_sum = vaddq_f32(v_sum, vld1q_f32(row_x + d));
            }
            float sum = vaddvq_f32(v_sum);
            for (; d < D; ++d)
                sum += row_x[d];
            float mean = sum * inv_D;

            // --- Pass 2: compute variance using (x - mean)^2 ---
            float32x4_t v_mean = vdupq_n_f32(mean);
            float32x4_t v_sum_sq = vdupq_n_f32(0.0f);
            d = 0;
            for (; d + 8 <= D; d += 8)
            {
                float32x4_t v_x0 = vld1q_f32(row_x + d);
                float32x4_t v_x1 = vld1q_f32(row_x + d + 4);
                float32x4_t v_diff0 = vsubq_f32(v_x0, v_mean);
                float32x4_t v_diff1 = vsubq_f32(v_x1, v_mean);
                v_sum_sq = vfmaq_f32(v_sum_sq, v_diff0, v_diff0);
                v_sum_sq = vfmaq_f32(v_sum_sq, v_diff1, v_diff1);
            }
            for (; d + 4 <= D; d += 4)
            {
                float32x4_t v_x = vld1q_f32(row_x + d);
                float32x4_t v_diff = vsubq_f32(v_x, v_mean);
                v_sum_sq = vfmaq_f32(v_sum_sq, v_diff, v_diff);
            }
            float sum_sq = vaddvq_f32(v_sum_sq);
            for (; d < D; ++d)
            {
                float diff = row_x[d] - mean;
                sum_sq += diff * diff;
            }
            float var = sum_sq * inv_D;
            float inv_std = 1.0f / std::sqrt(var + eps);

            // --- Pass 3: (x - mean) * inv_std * w + b ---
            float32x4_t v_inv_std = vdupq_n_f32(inv_std);
            d = 0;
            for (; d + 8 <= D; d += 8)
            {
                float32x4_t v_x0 = vld1q_f32(row_x + d);
                float32x4_t v_x1 = vld1q_f32(row_x + d + 4);
                float32x4_t v_w0 = vld1q_f32(w + d);
                float32x4_t v_w1 = vld1q_f32(w + d + 4);
                float32x4_t v_b0 = vld1q_f32(b + d);
                float32x4_t v_b1 = vld1q_f32(b + d + 4);

                float32x4_t v_n0 = vmulq_f32(vsubq_f32(v_x0, v_mean), v_inv_std);
                float32x4_t v_n1 = vmulq_f32(vsubq_f32(v_x1, v_mean), v_inv_std);
                v_n0 = vfmaq_f32(v_b0, v_n0, v_w0); // n * w + b
                v_n1 = vfmaq_f32(v_b1, v_n1, v_w1);
                vst1q_f32(row_out + d, v_n0);
                vst1q_f32(row_out + d + 4, v_n1);
            }
            for (; d + 4 <= D; d += 4)
            {
                float32x4_t v_x = vld1q_f32(row_x + d);
                float32x4_t v_w = vld1q_f32(w + d);
                float32x4_t v_b = vld1q_f32(b + d);
                float32x4_t v_n = vmulq_f32(vsubq_f32(v_x, v_mean), v_inv_std);
                v_n = vfmaq_f32(v_b, v_n, v_w);
                vst1q_f32(row_out + d, v_n);
            }
            for (; d < D; ++d)
            {
                float n = (row_x[d] - mean) * inv_std;
                row_out[d] = n * w[d] + b[d];
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Reference Factory — mirrors JinaV5OmniNanoRetrievalModel::layer_norm()
// decomposition EXACTLY (same op types, same float constants, same structure).
//
// The float constants (D, eps, 0.5, 1.0) are matched byte-for-byte during
// isomorphism checking, so this factory only matches subgraphs with the
// same D and eps.  For jina-v5 vision blocks + merger: D=768, eps=1e-6.
// ---------------------------------------------------------------------------


REGISTER_KERNEL("JinaLayerNormWB_F32_3D", 3, 3, matchJinaLayerNormWB_F32_3D, runJinaLayerNormWB_F32_3D,
                refFactoryJinaLayerNormWB_F32_3D, {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32}, {{1, 1024, 768}, {768}, {768}}, {true, true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON
