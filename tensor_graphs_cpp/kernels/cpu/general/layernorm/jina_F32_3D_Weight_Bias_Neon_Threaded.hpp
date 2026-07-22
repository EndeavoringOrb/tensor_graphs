// File: tensor_graphs_cpp/kernels/cpu/general/layernorm/jina_F32_3D_Weight_Bias_Neon_Threaded.hpp
//
// FUSED KERNEL: LayerNorm (with weight + bias) for jina-embeddings-v5-omni-nano-retrieval
//
// Matches the exact subgraph produced by JinaV5OmniNanoRetrievalModel::layer_norm():
//
//   ax_node  = constant(-1, INT32)
//   sum_x    = sum(x, ax_node)                        // {B, S, 1}
//   d_node   = expand_scalar_to_3d(D, 1, S, 1)        // {1, S, 1}  (D as float, e.g. 768.0)
//   mean_val = div(sum_x, d_node)                     // {B, S, 1}
//   mean     = repeat_3d_axis(mean_val, D, 2)         // {B, S, D}
//   x_sub    = add(x, neg(mean))                      // {B, S, D}
//   sq       = mul(x_sub, x_sub)                      // {B, S, D}
//   sum_sq   = sum(sq, ax_node)                       // {B, S, 1}
//   var      = div(sum_sq, d_node)                    // {B, S, 1}
//   eps_node = expand_scalar_to_3d(eps, 1, S, 1)      // {1, S, 1}  (eps = 1e-6)
//   var_eps  = add(var, eps_node)                     // {B, S, 1}
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
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

// ---------------------------------------------------------------------------
// Match function — only structural shape checks (linter-friendly).
// ---------------------------------------------------------------------------
inline bool matchJinaLayerNormWB_F32_3D(const std::vector<TensorNode> &inputs,
                                        const TensorNode &output)
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
                    v_n0 = vfmaq_f32(v_b0, v_n0, v_w0);  // n * w + b
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
            } });
    }
    for (auto &worker : workers)
        worker.join();
}

// ---------------------------------------------------------------------------
// Reference Factory — mirrors JinaV5OmniNanoRetrievalModel::layer_norm()
// decomposition EXACTLY (same op types, same float constants, same structure).
//
// The float constants (D, eps, 0.5, 1.0) are matched byte-for-byte during
// isomorphism checking, so this factory only matches subgraphs with the
// same D and eps.  For jina-v5 vision blocks + merger: D=768, eps=1e-6.
// ---------------------------------------------------------------------------
inline LogicalId refFactoryJinaLayerNormWB_F32_3D(const std::vector<LogicalId> &inputs,
                                                 Graph &g)
{
    LogicalId x_id = inputs[0];
    LogicalId w_id = inputs[1];
    LogicalId b_id = inputs[2];

    const auto &shape = g.getNode(x_id).getShape();
    uint32_t B = shape[0];
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    // Helper: expand_scalar_to_3d(val, 1, S, 1) → {1, S, 1}
    // Mirrors JinaV5OmniNanoRetrievalModel::expand_scalar_to_3d exactly.
    auto expand_scalar_1S1 = [&](float val) -> LogicalId
    {
        LogicalId node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh[] = {1, 1, 1};
        LogicalId out = g.reshape(node, g.constant({3}, sh, DType::INT32));
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

    // Helper: repeat_3d_axis(node, D, 2) → broadcast axis 2 by D
    auto repeat_d_axis2 = [&](LogicalId node) -> LogicalId
    {
        int32_t rep = (int32_t)D;
        int32_t ax = 2;
        return g.repeat(node,
                        g.constant({1}, &rep, DType::INT32),
                        g.constant({1}, &ax, DType::INT32));
    };

    // Helper: expand_1d_to_3d(vec, D, 1, S) → {1, S, D}
    // Mirrors JinaV5OmniNanoRetrievalModel::expand_1d_to_3d exactly.
    auto expand_1d_1SD = [&](LogicalId vec) -> LogicalId
    {
        int32_t sh[] = {1, 1, (int32_t)D};
        LogicalId out = g.reshape(vec, g.constant({3}, sh, DType::INT32));
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

    // --- axis = -1 ---
    int32_t ax_val = -1;
    LogicalId axis_node = g.constant({1}, &ax_val, DType::INT32);

    // --- mean ---
    LogicalId sum_x = g.sum(x_id, axis_node);      // {B, S, 1}
    float d_float = (float)D;                     // 768.0f
    LogicalId d_node = expand_scalar_1S1(d_float); // {1, S, 1}
    LogicalId mean_val = g.div(sum_x, d_node);     // {B, S, 1}
    LogicalId mean = repeat_d_axis2(mean_val);     // {B, S, D}

    // --- x - mean ---
    LogicalId x_sub = g.add(x_id, g.neg(mean)); // {B, S, D}

    // --- variance ---
    LogicalId sq = g.mul(x_sub, x_sub);      // {B, S, D}
    LogicalId sum_sq = g.sum(sq, axis_node); // {B, S, 1}
    LogicalId var = g.div(sum_sq, d_node);   // {B, S, 1}

    // --- std = sqrt(var + eps) ---
    LogicalId eps_node = expand_scalar_1S1(1e-6f);     // {1, S, 1}
    LogicalId var_plus_eps = g.add(var, eps_node);     // {B, S, 1}
    LogicalId sqrt_exp = expand_scalar_1S1(0.5f);      // {1, S, 1}
    LogicalId std_dev = g.pow(var_plus_eps, sqrt_exp); // {B, S, 1}

    // --- inv_std = 1 / std ---
    LogicalId one_node = expand_scalar_1S1(1.0f);    // {1, S, 1}
    LogicalId inv_std = g.div(one_node, std_dev);    // {B, S, 1}
    LogicalId inv_std_exp = repeat_d_axis2(inv_std); // {B, S, D}

    // --- normalize ---
    LogicalId normalized = g.mul(x_sub, inv_std_exp); // {B, S, D}

    // --- apply weight ---
    LogicalId w_exp = expand_1d_1SD(w_id);  // {1, S, D}
    normalized = g.mul(normalized, w_exp); // {B, S, D}

    // --- apply bias ---
    LogicalId b_exp = expand_1d_1SD(b_id); // {1, S, D}
    return g.add(normalized, b_exp);      // {B, S, D}
}

REGISTER_KERNEL("JinaLayerNormWB_F32_3D", 3, 3, matchJinaLayerNormWB_F32_3D, runJinaLayerNormWB_F32_3D, refFactoryJinaLayerNormWB_F32_3D, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
                {{1, 1024, 768}, {768}, {768}},
                {true, true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON
