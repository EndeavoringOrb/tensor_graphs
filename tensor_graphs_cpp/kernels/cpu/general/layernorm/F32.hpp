#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>

// =============================================================================
// FUSED KERNEL: LayerNorm F32 (no affine parameters)
// Formula: LayerNorm(x) = (x - mean) / sqrt(var + eps)
//
// This kernel replaces the decomposed layer_norm_atomic subgraph which uses
// pow(var + eps, 0.5) as sqrt. The decomposed form is vulnerable because:
//   - If upstream data is corrupted (e.g., NaN from unstable silu_atomic),
//     variance can become negative, making pow(negative, 0.5) = NaN
//   - The decomposed form creates many intermediate nodes, each with
//     potential for numerical drift
//
// This fused kernel computes LayerNorm in a single pass with:
//   - Direct mean/variance computation (no intermediate pow)
//   - std::sqrt which is well-defined for var + eps >= eps > 0
//   - No risk of pow(negative, 0.5) producing NaN
//   - Hardcoded eps = 1e-6 (matching the FLUX model's layer_norm_atomic)
// =============================================================================

// Default epsilon used by the FLUX model's layer_norm_atomic
static constexpr float LAYERNORM_DEFAULT_EPS = 1e-6f;

inline bool matchLayerNormF32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 1)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    // Layer norm operates on 3D tensors [Batch, Seq, Hidden]
    if (inputs[0].getShape().size() != 3)
        return false;
    if (output.getShape() != inputs[0].getShape())
        return false;
    if (!isContiguous(output))
        return false;

    return true;
}

inline void runLayerNormF32_3D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                               const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *x = static_cast<const float *>(inputs[0]);
    float *out = static_cast<float *>(outputs[0]);

    uint32_t B = inViews[0].getShape()[0];
    uint32_t S = inViews[0].getShape()[1];
    uint32_t D = inViews[0].getShape()[2];

    float eps = LAYERNORM_DEFAULT_EPS;

    for (uint32_t b = 0; b < B; ++b)
    {
        for (uint32_t s = 0; s < S; ++s)
        {
            const float *x_row = x + b * S * D + s * D;
            float *out_row = out + b * S * D + s * D;

            // 1. Compute mean
            float sum = 0.0f;
            for (uint32_t d = 0; d < D; ++d)
                sum += x_row[d];
            float mean = sum / (float)D;

            // 2. Compute variance
            float var_sum = 0.0f;
            for (uint32_t d = 0; d < D; ++d)
            {
                float diff = x_row[d] - mean;
                var_sum += diff * diff;
            }
            float var = var_sum / (float)D;

            // 3. Compute inverse standard deviation
            // var + eps is guaranteed positive (var >= 0, eps > 0),
            // so std::sqrt always returns a valid positive number.
            // This eliminates the pow(negative, 0.5) = NaN bug.
            float inv_std = 1.0f / std::sqrt(var + eps);

            // 4. Normalize: (x - mean) * inv_std
            for (uint32_t d = 0; d < D; ++d)
                out_row[d] = (x_row[d] - mean) * inv_std;
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: expand a scalar to shape {1, S, 1} (matching expand_scalar_to_3d
// with d0=1, d1=S, d2=1 as used in layer_norm_atomic)
// ---------------------------------------------------------------------------
inline uint32_t ref_ln_expand_scalar_1S1(Graph &g, float val, uint32_t S)
{
    uint32_t node = g.constant({1}, &val, DType::FLOAT32);
    int32_t sh3[] = {1, 1, 1};
    uint32_t out = g.reshape(node, g.constant({3}, sh3, DType::INT32));
    if (S > 1)
    {
        int32_t rep = (int32_t)S;
        int32_t axis = 1;
        out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &axis, DType::INT32));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Helper: repeat along axis 2 to expand {B, S, 1} -> {B, S, D}
// (mirrors repeat_ax(node, D, 2) used in layer_norm_atomic)
// ---------------------------------------------------------------------------
inline uint32_t ref_ln_repeat_ax2(Graph &g, uint32_t node, uint32_t D)
{
    if (D <= 1)
        return node;
    int32_t rep = (int32_t)D;
    int32_t axis = 2;
    return g.repeat(node, g.constant({1}, &rep, DType::INT32), g.constant({1}, &axis, DType::INT32));
}

// ---------------------------------------------------------------------------
// Reference Factory: decomposes LayerNorm into the same graph structure as
// layer_norm_atomic so the e-graph isomorphism check can match it.
//
// layer_norm_atomic decomposition (from flux-klein-4b-transformer.hpp):
//
//   ax_node  = constant(-1, INT32)
//   mean     = repeat_ax(div(sum(x, ax_node),
//                           expand_scalar_to_3d(D, 1, seq, 1)),
//                        D, 2)
//   x_sub    = add(x, neg(mean))
//   sq       = mul(x_sub, x_sub)
//   var      = div(sum(sq, ax_node),
//                  expand_scalar_to_3d(D, 1, seq, 1))
//   std      = pow(add(var, expand_scalar_to_3d(1e-6, 1, seq, 1)),
//                  expand_scalar_to_3d(0.5, 1, seq, 1))
//   result   = mul(x_sub,
//                  repeat_ax(div(expand_scalar_to_3d(1.0, 1, seq, 1), std),
//                            D, 2))
//
// Note: expand_scalar_to_3d(val, 1, seq, 1) creates shape {1, seq, 1}
//       repeat_ax(node, D, 2) broadcasts axis 2 by D
// ---------------------------------------------------------------------------
inline uint32_t refFactoryLayerNorm(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t x_id = inputs[0];
    const auto &shape = graph.getNode(x_id).getShape();
    uint32_t B = shape[0];
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    // axis = -1 (reduce over last dimension)
    int32_t ax_val = -1;
    uint32_t ax_node = graph.constant({1}, &ax_val, DType::INT32);

    // --- Mean ---
    // sum(x, axis=-1) -> {B, S, 1}
    uint32_t sum_x = graph.sum(x_id, ax_node);

    // D as float, expanded to {1, S, 1}
    float d_float = (float)D;
    uint32_t d_node = ref_ln_expand_scalar_1S1(graph, d_float, S);

    // mean_val = sum(x, -1) / D -> {B, S, 1}
    uint32_t mean_val = graph.div(sum_x, d_node);

    // mean = repeat_ax(mean_val, D, 2) -> {B, S, D}
    uint32_t mean = ref_ln_repeat_ax2(graph, mean_val, D);

    // --- Centered ---
    // x_sub = x + neg(mean) = x - mean
    uint32_t x_sub = graph.add(x_id, graph.neg(mean));

    // --- Variance ---
    // sq = x_sub * x_sub
    uint32_t sq = graph.mul(x_sub, x_sub);

    // sum(sq, axis=-1) -> {B, S, 1}
    uint32_t sum_sq = graph.sum(sq, ax_node);

    // var = sum(sq, -1) / D -> {B, S, 1}
    uint32_t var = graph.div(sum_sq, d_node);

    // --- Standard Deviation ---
    // var + eps -> {B, S, 1}
    float eps = LAYERNORM_DEFAULT_EPS;
    uint32_t eps_node = ref_ln_expand_scalar_1S1(graph, eps, S);
    uint32_t var_plus_eps = graph.add(var, eps_node);

    // std = pow(var + eps, 0.5) -> {B, S, 1}
    float half_val = 0.5f;
    uint32_t sqrt_exp = ref_ln_expand_scalar_1S1(graph, half_val, S);
    uint32_t std_dev = graph.pow(var_plus_eps, sqrt_exp);

    // --- Inverse Std ---
    // 1.0 / std -> {B, S, 1}
    float one_val = 1.0f;
    uint32_t one_node = ref_ln_expand_scalar_1S1(graph, one_val, S);
    uint32_t inv_std = graph.div(one_node, std_dev);

    // repeat_ax(inv_std, D, 2) -> {B, S, D}
    uint32_t inv_std_expanded = ref_ln_repeat_ax2(graph, inv_std, D);

    // --- Result ---
    // x_sub * inv_std_expanded
    return graph.mul(x_sub, inv_std_expanded);
}

REGISTER_KERNEL("LayerNorm", 1, matchLayerNormF32_3D, runLayerNormF32_3D, refFactoryLayerNorm,
                {Backend::CPU}, {DType::FLOAT32}, {{1, 1, 3072}}, {true}, {{Backend::CPU}});