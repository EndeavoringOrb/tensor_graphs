#pragma once
#include <cmath>

#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/silu/ref.hpp"
// =============================================================================
// FUSED KERNEL: SiLU (Sigmoid Linear Unit) F32
// Formula: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// This kernel replaces the decomposed silu_atomic subgraph which uses
// pow(e, -x) and is numerically unstable for large negative inputs:
//   pow(e, -x) overflows to +inf when -x > 88  (x < -88)
//   Then (-inf) * 0 = NaN via IEEE 754 indeterminate form
//
// This fused kernel uses std::exp with a branch that avoids overflow:
//   For x >= 0:  silu(x) = x / (1 + exp(-x))     — exp(-x) is small, safe
//   For x < 0:   silu(x) = x * exp(x) / (1 + exp(x)) — exp(x) is small, safe
// =============================================================================

inline bool matchSiluF32(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    // SiLU operates on any-rank contiguous F32 tensor
    return true;
}

inline void runSiluF32(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < n; ++i)
    {
        float x = in[i];
        // Numerically stable SiLU:
        //   x >= 0: x / (1 + exp(-x))    — exp(-x) in (0, 1], no overflow
        //   x < 0:  x * exp(x) / (1 + exp(x)) — exp(x) in (0, 1), no overflow
        // Both branches avoid the indeterminate form (-inf)*0 = NaN
        if (x >= 0.0f)
        {
            out[i] = x / (1.0f + std::exp(-x));
        }
        else
        {
            float exp_x = std::exp(x);
            out[i] = x * exp_x / (1.0f + exp_x);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: broadcast a scalar constant to match a target shape
// Mirrors the expand_scalar_to_3d pattern used in silu_atomic
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Reference Factory: decomposes SiLU into the same graph structure as
// silu_atomic so the e-graph isomorphism check can match it.
//
// silu_atomic decomposition:
//   neg_x    = neg(x)
//   exp_neg  = pow(e_expanded, neg_x)
//   den      = add(1_expanded, exp_neg)
//   sig      = div(1_expanded, den)
//   result   = mul(x, sig)
// ---------------------------------------------------------------------------


REGISTER_KERNEL("Silu_3D_1", 1, 1, matchSiluF32, runSiluF32, refFactorySilu, {0}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 1, 2048}}, {true},
                {{MemSpace(1, HandleType::CPP)}});