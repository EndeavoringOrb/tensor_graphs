
//
// FUSED KERNEL: L2 Normalize for jina-embeddings-v5-omni-nano-retrieval output
//
// Matches the subgraph produced by
// JinaV5OmniNanoRetrievalModel::l2_normalize():
//
//   x_sq     = mul(x, x)                              // {1, D}
//   sum_sq   = sum(x_sq, axis=-1)                     // {1, 1}
//   eps_exp  = expand_scalar_to_2d(1e-12, 1, 1)       // {1, 1}
//   ms_eps   = add(sum_sq, eps_exp)                   // {1, 1}
//   sqrt_exp = expand_scalar_to_2d(0.5, 1, 1)         // {1, 1}
//   std      = pow(ms_eps, sqrt_exp)                  // {1, 1}
//   one_node = expand_scalar_to_2d(1.0, 1, 1)         // {1, 1}
//   inv_std  = div(one_node, std)                     // {1, 1}
//   inv_exp  = repeat(inv_std, D, axis=1)             // {1, D}
//   result   = mul(x, inv_exp)                        // {1, D}
//
// This is a small kernel (D=768) but eliminates 6 intermediate tensors and
// their associated kernel launch overhead.
//
// Hardware: Qualcomm aarch64 (NEON).  Single-threaded (small tensor).
// Value constants matched byte-for-byte: 1e-12, 0.5, 1.0.
#pragma once
#include <cmath>

#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchJinaL2Norm_F32_2D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // x: 2-D [1, D]
    if (inputs[0].getShape().size() != 2)
        return false;
    return isContiguous(output);
}

inline void runJinaL2Norm_F32_2D(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t D = ctx.inViews[0].getShape()[1];
    const float eps = 1e-12f;

    // Pass 1: sum of squares
    float32x4_t v_sum_sq = vdupq_n_f32(0.0f);
    uint32_t d = 0;
    for (; d + 4 <= D; d += 4)
    {
        float32x4_t v_x = vld1q_f32(x + d);
        v_sum_sq = vfmaq_f32(v_sum_sq, v_x, v_x);
    }
    float sum_sq = vaddvq_f32(v_sum_sq);
    for (; d < D; ++d)
        sum_sq += x[d] * x[d];

    float inv_norm = 1.0f / std::sqrt(sum_sq + eps);

    // Pass 2: x * inv_norm
    float32x4_t v_inv = vdupq_n_f32(inv_norm);
    d = 0;
    for (; d + 4 <= D; d += 4)
    {
        float32x4_t v_x = vld1q_f32(x + d);
        vst1q_f32(out + d, vmulq_f32(v_x, v_inv));
    }
    for (; d < D; ++d)
        out[d] = x[d] * inv_norm;
}

// Reference Factory — mirrors JinaV5OmniNanoRetrievalModel::l2_normalize()
// decomposition EXACTLY.
//
// Note: l2_normalize uses expand_scalar_to_2d (not _3d), creating {1, 1}
// shapes.
inline LogicalId refFactoryJinaL2Norm_F32_2D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x_id = inputs[0];
    const auto &shape = g.getNode(x_id).getShape();
    uint32_t D = shape[1];

    // Helper: expand_scalar_to_2d(val, 1, 1) → {1, 1}
    // Mirrors JinaV5OmniNanoRetrievalModel::expand_scalar_to_2d(val, 1, 1).
    auto expand_scalar_11 = [&](float val) -> LogicalId {
        LogicalId node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh[] = {1, 1};
        return g.reshape(node, g.constant({2}, sh, DType::INT32));
        // dim0=1, dim1=1 → no repeats needed
    };

    // x_sq = x * x
    LogicalId x_sq = g.mul(x_id, x_id);

    // sum_sq = sum(x_sq, axis=-1)
    int32_t ax_val = -1;
    LogicalId axis_node = g.constant({1}, &ax_val, DType::INT32);
    LogicalId sum_sq = g.sum(x_sq, axis_node);

    // std = sqrt(sum_sq + eps)
    LogicalId eps_node = expand_scalar_11(1e-12f);
    LogicalId sum_sq_plus_eps = g.add(sum_sq, eps_node);
    LogicalId sqrt_node = expand_scalar_11(0.5f);
    LogicalId std = g.pow(sum_sq_plus_eps, sqrt_node);

    // inv_std = 1 / std
    LogicalId one_node = expand_scalar_11(1.0f);
    LogicalId inv_std = g.div(one_node, std);

    // inv_std_expanded = repeat(inv_std, D, axis=1)
    int32_t rep = (int32_t)D;
    int32_t ax = 1;
    LogicalId inv_std_expanded =
        g.repeat(inv_std, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));

    // result = x * inv_std_expanded
    return g.mul(x_id, inv_std_expanded);
}

REGISTER_KERNEL("JinaL2Norm_F32_2D", 1, 1, matchJinaL2Norm_F32_2D, runJinaL2Norm_F32_2D, refFactoryJinaL2Norm_F32_2D,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 768}}, {true},
                {{MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON
