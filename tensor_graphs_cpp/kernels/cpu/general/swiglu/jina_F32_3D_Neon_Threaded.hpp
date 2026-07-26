// File:
// tensor_graphs_cpp/kernels/cpu/general/swiglu/jina_F32_3D_Neon_Threaded.hpp
//
// FUSED KERNEL: SwiGLU for jina-embeddings-v5-omni-nano-retrieval text encoder
//
// Matches the subgraph produced by JinaV5OmniNanoRetrievalModel::text_mlp():
//
//   gate     = linear(x, "gate_proj", ...)               // {1, S, D}
//   up       = linear(x, "up_proj", ...)                 // {1, S, D}
//   gate_silu = silu_atomic(gate)                        // decomposed ~7 ops
//   gate_up   = mul(gate_silu, up)                       // {1, S, D}
//
// The silu_atomic(gate) decomposition uses mul(gate, -1) for negation:
//   neg_one  = expand_scalar_to_3d(-1.0, 1, S, D)
//   neg_gate = mul(gate, neg_one)
//   e_node   = expand_scalar_to_3d(2.71828..., 1, S, D)
//   exp_neg  = pow(e_node, neg_gate)
//   one_node = expand_scalar_to_3d(1.0, 1, S, D)
//   den      = add(one_node, exp_neg)
//   sig      = div(one_node, den)
//   gate_silu = mul(gate, sig)
//   result   = mul(gate_silu, up)
//
// This fused kernel takes gate and up directly (2 inputs) and computes
// silu(gate) * up in a single pass, eliminating 7+ intermediate tensors.
//
// Hardware: Qualcomm aarch64 (ARMv8.6, 12 cores, NEON FMA).
// Value constants matched byte-for-byte: -1.0, 2.718281828459045, 1.0.
#pragma once
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

static inline float32x4_t jina_swiglu_exp_neon(float32x4_t x)
{
    x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(88.0f)), vdupq_n_f32(-88.0f));

    float32x4_t v_log2e = vdupq_n_f32(1.4426950408889634f);
    float32x4_t v_nf = vmulq_f32(x, v_log2e);

    float32x4_t v_n = vrndnq_f32(v_nf);
    float32x4_t v_f = vsubq_f32(v_nf, v_n);

    float32x4_t v_poly = vdupq_n_f32(0.009618129107628477f);
    v_poly = vfmaq_f32(vdupq_n_f32(0.05550410866482158f), v_f, v_poly);
    v_poly = vfmaq_f32(vdupq_n_f32(0.2402265069591007f), v_f, v_poly);
    v_poly = vfmaq_f32(vdupq_n_f32(0.6931471805599453f), v_f, v_poly);
    v_poly = vfmaq_f32(vdupq_n_f32(1.0f), v_f, v_poly);

    int32x4_t v_n_int = vcvtq_s32_f32(v_n);
    int32x4_t v_exp_bits = vshlq_n_s32(vaddq_s32(v_n_int, vdupq_n_s32(127)), 23);
    float32x4_t v_2n = vreinterpretq_f32_s32(v_exp_bits);

    return vmulq_f32(v_poly, v_2n);
}

inline bool matchJinaSwiGLU_F32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // gate: 3-D [1, S, D], up: 3-D [1, S, D], same shape
    if (inputs[0].getShape().size() != 3)
        return false;
    if (inputs[1].getShape().size() != 3)
        return false;
    if (inputs[0].getShape() != inputs[1].getShape())
        return false;
    return isContiguous(output);
}

// Fused SwiGLU: silu(gate) * up = (gate / (1 + exp(-gate))) * up
inline void runJinaSwiGLU_F32_3D(const KernelContext &ctx)
{
    const float *gate = static_cast<const float *>(ctx.inputs[0]);
    const float *up = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    if (num_threads > 12)
        num_threads = 12;
    uint64_t chunk = (n + num_threads - 1) / num_threads;

    const float32x4_t v_one = vdupq_n_f32(1.0f);
    const float32x4_t v_zero = vdupq_n_f32(0.0f);

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]() {
            uint64_t start = t * chunk;
            uint64_t end = std::min(start + chunk, n);

            uint64_t i = start;
            for (; i + 16 <= end; i += 16)
            {
                float32x4_t v_g0 = vld1q_f32(gate + i);
                float32x4_t v_g1 = vld1q_f32(gate + i + 4);
                float32x4_t v_g2 = vld1q_f32(gate + i + 8);
                float32x4_t v_g3 = vld1q_f32(gate + i + 12);
                float32x4_t v_u0 = vld1q_f32(up + i);
                float32x4_t v_u1 = vld1q_f32(up + i + 4);
                float32x4_t v_u2 = vld1q_f32(up + i + 8);
                float32x4_t v_u3 = vld1q_f32(up + i + 12);

                auto swiglu_vec = [&](float32x4_t v_g, float32x4_t v_u) -> float32x4_t {
                    // silu(g) = g / (1 + exp(-g))  for g >= 0
                    //         = g * exp(g) / (1 + exp(g))  for g < 0
                    float32x4_t v_neg_g = vnegq_f32(v_g);
                    float32x4_t v_exp_neg = jina_swiglu_exp_neon(v_neg_g);
                    float32x4_t v_den = vaddq_f32(v_one, v_exp_neg);
                    float32x4_t v_exp_pos = jina_swiglu_exp_neon(v_g);
                    float32x4_t v_den_neg = vaddq_f32(v_one, v_exp_pos);
                    float32x4_t v_num_neg = vmulq_f32(v_g, v_exp_pos);
                    float32x4_t v_silu_pos = vdivq_f32(v_g, v_den);
                    float32x4_t v_silu_neg = vdivq_f32(v_num_neg, v_den_neg);
                    uint32x4_t v_neg_mask = vcltq_f32(v_g, v_zero);
                    float32x4_t v_silu = vbslq_f32(v_neg_mask, v_silu_neg, v_silu_pos);
                    return vmulq_f32(v_silu, v_u);
                };

                vst1q_f32(out + i, swiglu_vec(v_g0, v_u0));
                vst1q_f32(out + i + 4, swiglu_vec(v_g1, v_u1));
                vst1q_f32(out + i + 8, swiglu_vec(v_g2, v_u2));
                vst1q_f32(out + i + 12, swiglu_vec(v_g3, v_u3));
            }
            for (; i + 4 <= end; i += 4)
            {
                float32x4_t v_g = vld1q_f32(gate + i);
                float32x4_t v_u = vld1q_f32(up + i);
                float32x4_t v_neg_g = vnegq_f32(v_g);
                float32x4_t v_exp_neg = jina_swiglu_exp_neon(v_neg_g);
                float32x4_t v_den = vaddq_f32(v_one, v_exp_neg);
                float32x4_t v_exp_pos = jina_swiglu_exp_neon(v_g);
                float32x4_t v_den_neg = vaddq_f32(v_one, v_exp_pos);
                float32x4_t v_num_neg = vmulq_f32(v_g, v_exp_pos);
                float32x4_t v_silu_pos = vdivq_f32(v_g, v_den);
                float32x4_t v_silu_neg = vdivq_f32(v_num_neg, v_den_neg);
                uint32x4_t v_neg_mask = vcltq_f32(v_g, v_zero);
                float32x4_t v_silu = vbslq_f32(v_neg_mask, v_silu_neg, v_silu_pos);
                vst1q_f32(out + i, vmulq_f32(v_silu, v_u));
            }
            for (; i < end; ++i)
            {
                float g = gate[i];
                float silu;
                if (g >= 0.0f)
                    silu = g / (1.0f + std::exp(-g));
                else
                {
                    float eg = std::exp(g);
                    silu = g * eg / (1.0f + eg);
                }
                out[i] = silu * up[i];
            }
        });
    }
    for (auto &worker : workers)
        worker.join();
}

// Reference Factory — mirrors jina's text_mlp pattern:
//   silu_atomic(gate) * up
// where silu_atomic uses mul(gate, -1) for negation.
inline LogicalId refFactoryJinaSwiGLU_F32_3D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId gate_id = inputs[0];
    LogicalId up_id = inputs[1];
    const auto &shape = g.getNode(gate_id).getShape();
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    auto expand_scalar_SD = [&](float val) -> LogicalId {
        LogicalId node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh[] = {1, 1, 1};
        LogicalId out = g.reshape(node, g.constant({3}, sh, DType::INT32));
        if (S > 1)
        {
            int32_t rep = (int32_t)S;
            int32_t ax = 1;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));
        }
        if (D > 1)
        {
            int32_t rep = (int32_t)D;
            int32_t ax = 2;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    // --- silu_atomic(gate) ---
    LogicalId neg_one = expand_scalar_SD(-1.0f);
    LogicalId neg_gate = g.mul(gate_id, neg_one);
    LogicalId e_node = expand_scalar_SD(2.718281828459045f);
    LogicalId exp_neg = g.pow(e_node, neg_gate);
    LogicalId one_node = expand_scalar_SD(1.0f);
    LogicalId den = g.add(one_node, exp_neg);
    LogicalId sig = g.div(one_node, den);
    LogicalId gate_silu = g.mul(gate_id, sig);

    // --- mul(gate_silu, up) ---
    return g.mul(gate_silu, up_id);
}

REGISTER_KERNEL("JinaSwiGLU_F32_3D", 2, 2, matchJinaSwiGLU_F32_3D, runJinaSwiGLU_F32_3D, refFactoryJinaSwiGLU_F32_3D,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 1024, 3072}, {1, 1024, 3072}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON
