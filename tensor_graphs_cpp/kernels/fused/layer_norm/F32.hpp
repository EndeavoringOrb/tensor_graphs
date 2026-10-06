#pragma once
#include <cmath>

#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/layer_norm/ref.hpp"
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

inline bool matchLayerNormF32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Layer norm operates on 3D tensors [Batch, Seq, Hidden]
    if (inputs[0].getShape().size() != 3)
        return false;
    if (output.getShape() != inputs[0].getShape())
        return false;
    if (!isContiguous(output))
        return false;

    return true;
}

inline void runLayerNormF32_3D(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    uint32_t B = ctx.inViews[0].getShape()[0];
    uint32_t S = ctx.inViews[0].getShape()[1];
    uint32_t D = ctx.inViews[0].getShape()[2];

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

REGISTER_KERNEL("LayerNorm", 1, 1, matchLayerNormF32_3D, runLayerNormF32_3D, refFactoryLayerNorm, {0},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 1, 3072}}, {true},
                {{MemSpace(1, HandleType::CPP)}});
