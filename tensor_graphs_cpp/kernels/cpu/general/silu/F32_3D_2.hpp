#pragma once
#include <cmath>

#include "core/kernels.hpp"
#include "core/types.hpp"

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

inline bool matchSiluF32_2(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    // SiLU operates on any-rank contiguous F32 tensor
    return true;
}

inline void runSiluF32_2(const KernelContext &ctx)
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
inline LogicalId ref_silu_broadcast_scalar_2(Graph &g, LogicalId scalar_id, const std::vector<uint32_t> &target_shape)
{
    std::vector<int32_t> ones(target_shape.size(), 1);
    LogicalId out = g.reshape(scalar_id, g.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));
    for (uint64_t i = 0; i < target_shape.size(); ++i)
    {
        if (target_shape[i] > 1)
        {
            int32_t rep = (int32_t)target_shape[i];
            int32_t axis = (int32_t)i;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &axis, DType::INT32));
        }
    }
    return out;
}

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
inline LogicalId refFactorySilu_2(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x_id = inputs[0];
    const auto &target_shape = graph.getNode(x_id).getShape();

    // 1. neg_x = -x
    LogicalId neg_x = graph.neg(x_id);

    // 2. exp_neg = pow(e, -x)
    float e_val = 2.7182818f;
    LogicalId e_node = ref_silu_broadcast_scalar_2(graph, graph.constant({1}, &e_val, DType::FLOAT32), target_shape);
    LogicalId exp_neg = graph.pow(e_node, neg_x);

    // 3. den = 1 + exp(-x)
    float one_val = 1.0f;
    LogicalId one_node =
        ref_silu_broadcast_scalar_2(graph, graph.constant({1}, &one_val, DType::FLOAT32), target_shape);
    LogicalId den = graph.add(one_node, exp_neg);

    // 4. sig = 1 / den
    LogicalId sig = graph.div(one_node, den);

    // 5. result = x * sig
    return graph.mul(x_id, sig);
}

REGISTER_KERNEL("Silu_3D_2", 1, 1, matchSiluF32_2, runSiluF32_2, refFactorySilu_2, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 4, 2048}}, {true},
                {{MemSpace(1, HandleType::CPP)}});