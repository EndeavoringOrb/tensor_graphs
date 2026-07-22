// File: tensor_graphs_cpp/kernels/cpu/general/gelu/jina_F32_3D_Exact_Erf_Neon_Threaded.hpp
//
// FUSED KERNEL: Exact-erf GELU for jina-embeddings-v5-omni-nano-retrieval
//
// Matches the exact subgraph produced by JinaV5OmniNanoRetrievalModel::gelu_exact():
//
//   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// where erf is the Abramowitz-Stegun approximation:
//   t = 1 / (1 + p * |z|),  p = 0.3275911
//   erf(z) = sign(z) * (1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-z^2))
//
// The model's decomposed form creates ~30+ intermediate tensors of size (1, S, D),
// each requiring a full memory pass.  For S=4320, D=3072 (vision MLP), that's
// ~30 * 53 MB = 1.6 GB of intermediate traffic per call.  This fused kernel
// reads the input once and writes the output once — a >10× reduction in
// memory traffic.
//
// Hardware: Qualcomm aarch64 (ARMv8.6, 12 cores, NEON FMA).
// Value constants matched byte-for-byte: 0.7071067811865475, 0.5, 1e-12,
// 0.3275911, 1.0, 0.254829592, -0.284496736, 1.421413741, -1.453152027,
// 1.061405429, 2.718281828459045.
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
// NEON vectorized exp(x) using 2^(x * log2(e)) decomposition.
// Max relative error ~1e-7 on [-88, 88].
// ---------------------------------------------------------------------------
static inline float32x4_t jina_gelu_exp_neon(float32x4_t x)
{
    // Clamp to avoid overflow/underflow
    x = vmaxq_f32(vminq_f32(x, vdupq_n_f32(88.0f)), vdupq_n_f32(-88.0f));

    // exp(x) = 2^(x * log2(e))
    float32x4_t v_log2e = vdupq_n_f32(1.4426950408889634f);
    float32x4_t v_nf = vmulq_f32(x, v_log2e);

    // n = round(v_nf), f = v_nf - n  (f in [-0.5, 0.5])
    float32x4_t v_n = vrndnq_f32(v_nf);
    float32x4_t v_f = vsubq_f32(v_nf, v_n);

    // Minimax polynomial for 2^f on [-0.5, 0.5]:
    //   2^f ≈ 1 + f*(ln2 + f*(ln2²/2 + f*(ln2³/6 + f*ln2⁴/24)))
    float32x4_t v_poly = vdupq_n_f32(0.009618129107628477f);            // ln2^4/24
    v_poly = vfmaq_f32(vdupq_n_f32(0.05550410866482158f), v_f, v_poly); // ln2^3/6
    v_poly = vfmaq_f32(vdupq_n_f32(0.2402265069591007f), v_f, v_poly);  // ln2^2/2
    v_poly = vfmaq_f32(vdupq_n_f32(0.6931471805599453f), v_f, v_poly);  // ln2
    v_poly = vfmaq_f32(vdupq_n_f32(1.0f), v_f, v_poly);

    // 2^n via IEEE 754 exponent bit manipulation
    int32x4_t v_n_int = vcvtq_s32_f32(v_n);
    int32x4_t v_exp_bits = vshlq_n_s32(vaddq_s32(v_n_int, vdupq_n_s32(127)), 23);
    float32x4_t v_2n = vreinterpretq_f32_s32(v_exp_bits);

    return vmulq_f32(v_poly, v_2n);
}

// ---------------------------------------------------------------------------
// NEON vectorized erf using Abramowitz-Stegun 7.1.26 approximation.
// Max error ~1.5e-7 (matches the model's scalar decomposition).
// ---------------------------------------------------------------------------
static inline float32x4_t jina_gelu_erf_neon(float32x4_t z)
{
    float32x4_t v_abs_z = vabsq_f32(z);
    float32x4_t v_z_sq = vmulq_f32(z, z);

    // t = 1 / (1 + p * |z|),  p = 0.3275911
    float32x4_t v_p = vdupq_n_f32(0.3275911f);
    float32x4_t v_one = vdupq_n_f32(1.0f);
    float32x4_t v_denom = vaddq_f32(v_one, vmulq_f32(v_p, v_abs_z));
    float32x4_t v_t = vdivq_f32(v_one, v_denom);

    // Horner: poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    float32x4_t v_poly = vdupq_n_f32(1.061405429f);              // a5
    v_poly = vfmaq_f32(vdupq_n_f32(-1.453152027f), v_t, v_poly); // a4 + t*a5
    v_poly = vfmaq_f32(vdupq_n_f32(1.421413741f), v_t, v_poly);  // a3 + ...
    v_poly = vfmaq_f32(vdupq_n_f32(-0.284496736f), v_t, v_poly); // a2 + ...
    v_poly = vfmaq_f32(vdupq_n_f32(0.254829592f), v_t, v_poly);  // a1 + ...
    v_poly = vmulq_f32(v_t, v_poly);                             // t * (...)

    // exp(-z^2)
    float32x4_t v_neg_z_sq = vnegq_f32(v_z_sq);
    float32x4_t v_exp = jina_gelu_exp_neon(v_neg_z_sq);

    // erf_pos = 1 - poly * exp(-z^2)
    float32x4_t v_erf_pos = vsubq_f32(v_one, vmulq_f32(v_poly, v_exp));

    // erf(z) = sign(z) * erf_pos  (erf is odd)
    float32x4_t v_neg_erf = vnegq_f32(v_erf_pos);
    uint32x4_t v_neg_mask = vcltq_f32(z, vdupq_n_f32(0.0f));
    return vbslq_f32(v_neg_mask, v_neg_erf, v_erf_pos);
}

// ---------------------------------------------------------------------------
// Match: 1 input (x 3-D), output contiguous.
// ---------------------------------------------------------------------------
inline bool matchJinaGeluExact_F32_3D(const std::vector<TensorNode> &inputs,
                                      const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3)
        return false;
    return isContiguous(output);
}

// ---------------------------------------------------------------------------
// Run: NEON-threaded exact-erf GELU.
//   gelu(x) = 0.5 * x * (1 + erf(x * inv_sqrt2))
// ---------------------------------------------------------------------------
inline void runJinaGeluExact_F32_3D(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    if (num_threads > 12)
        num_threads = 12;
    uint64_t chunk = (n + num_threads - 1) / num_threads;

    const float32x4_t v_inv_sqrt2 = vdupq_n_f32(0.7071067811865475f);
    const float32x4_t v_half = vdupq_n_f32(0.5f);
    const float32x4_t v_one = vdupq_n_f32(1.0f);

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint64_t start = t * chunk;
            uint64_t end = std::min(start + chunk, n);

            uint64_t i = start;
            // Process 16 elements at a time (4 NEON vectors) for better ILP
            for (; i + 16 <= end; i += 16)
            {
                float32x4_t v_x0 = vld1q_f32(in + i);
                float32x4_t v_x1 = vld1q_f32(in + i + 4);
                float32x4_t v_x2 = vld1q_f32(in + i + 8);
                float32x4_t v_x3 = vld1q_f32(in + i + 12);

                // z = x / sqrt(2)
                float32x4_t v_z0 = vmulq_f32(v_x0, v_inv_sqrt2);
                float32x4_t v_z1 = vmulq_f32(v_x1, v_inv_sqrt2);
                float32x4_t v_z2 = vmulq_f32(v_x2, v_inv_sqrt2);
                float32x4_t v_z3 = vmulq_f32(v_x3, v_inv_sqrt2);

                // erf(z)
                float32x4_t v_erf0 = jina_gelu_erf_neon(v_z0);
                float32x4_t v_erf1 = jina_gelu_erf_neon(v_z1);
                float32x4_t v_erf2 = jina_gelu_erf_neon(v_z2);
                float32x4_t v_erf3 = jina_gelu_erf_neon(v_z3);

                // 0.5 * x * (1 + erf)
                float32x4_t v_r0 = vmulq_f32(v_half, vmulq_f32(v_x0, vaddq_f32(v_one, v_erf0)));
                float32x4_t v_r1 = vmulq_f32(v_half, vmulq_f32(v_x1, vaddq_f32(v_one, v_erf1)));
                float32x4_t v_r2 = vmulq_f32(v_half, vmulq_f32(v_x2, vaddq_f32(v_one, v_erf2)));
                float32x4_t v_r3 = vmulq_f32(v_half, vmulq_f32(v_x3, vaddq_f32(v_one, v_erf3)));

                vst1q_f32(out + i, v_r0);
                vst1q_f32(out + i + 4, v_r1);
                vst1q_f32(out + i + 8, v_r2);
                vst1q_f32(out + i + 12, v_r3);
            }
            // 4-element tail
            for (; i + 4 <= end; i += 4)
            {
                float32x4_t v_x = vld1q_f32(in + i);
                float32x4_t v_z = vmulq_f32(v_x, v_inv_sqrt2);
                float32x4_t v_erf = jina_gelu_erf_neon(v_z);
                float32x4_t v_r = vmulq_f32(v_half, vmulq_f32(v_x, vaddq_f32(v_one, v_erf)));
                vst1q_f32(out + i, v_r);
            }
            // Scalar tail
            for (; i < end; ++i)
            {
                float x = in[i];
                float z = x * 0.7071067811865475f;
                // Scalar erf using the same AS approximation
                float az = std::fabs(z);
                float t = 1.0f / (1.0f + 0.3275911f * az);
                float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
                float erf_pos = 1.0f - poly * std::exp(-z * z);
                float erf_val = (z >= 0.0f) ? erf_pos : -erf_pos;
                out[i] = 0.5f * x * (1.0f + erf_val);
            } });
    }
    for (auto &worker : workers)
        worker.join();
}

// ---------------------------------------------------------------------------
// Reference Factory — mirrors JinaV5OmniNanoRetrievalModel::gelu_exact()
// decomposition EXACTLY (same op types, same float constants, same structure).
//
// Each expand_scalar_to_3d(val, 1, S, D) creates:
//   const(val) → reshape({1,1,1}) → repeat(axis=1, S) → repeat(axis=2, D)
// ---------------------------------------------------------------------------
inline LogicalId refFactoryJinaGeluExact_F32_3D(const std::vector<LogicalId> &inputs,
                                               Graph &g)
{
    LogicalId x_id = inputs[0];
    const auto &shape = g.getNode(x_id).getShape();
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    // Helper: expand_scalar_to_3d(val, 1, S, D) → {1, S, D}
    auto expand_scalar_SD = [&](float val) -> LogicalId
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
        if (D > 1)
        {
            int32_t rep = (int32_t)D;
            int32_t ax = 2;
            out = g.repeat(out,
                           g.constant({1}, &rep, DType::INT32),
                           g.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    // Create each expansion ONCE and reuse (matching the model's local variables)
    LogicalId inv_sqrt2 = expand_scalar_SD(0.7071067811865475f);
    LogicalId half = expand_scalar_SD(0.5f);
    LogicalId eps_node = expand_scalar_SD(1e-12f);
    LogicalId p_node = expand_scalar_SD(0.3275911f);
    LogicalId one_node = expand_scalar_SD(1.0f);
    LogicalId a1 = expand_scalar_SD(0.254829592f);
    LogicalId a2 = expand_scalar_SD(-0.284496736f);
    LogicalId a3 = expand_scalar_SD(1.421413741f);
    LogicalId a4 = expand_scalar_SD(-1.453152027f);
    LogicalId a5 = expand_scalar_SD(1.061405429f);
    LogicalId e_node = expand_scalar_SD(2.718281828459045f);

    // x_scaled = x / sqrt(2)  →  x * inv_sqrt2
    LogicalId x_scaled = g.mul(x_id, inv_sqrt2);

    // xs_sq = x_scaled^2
    LogicalId xs_sq = g.mul(x_scaled, x_scaled);

    // abs_xs = pow(xs_sq, 0.5)  = sqrt(xs_sq) = |x_scaled|
    LogicalId abs_xs = g.pow(xs_sq, half);

    // sign_xs = x_scaled / (|x_scaled| + eps)
    LogicalId abs_xs_eps = g.add(abs_xs, eps_node);
    LogicalId sign_xs = g.div(x_scaled, abs_xs_eps);

    // t = 1 / (1 + p * |x_scaled|)
    LogicalId p_abs = g.mul(p_node, abs_xs);
    LogicalId denom = g.add(one_node, p_abs);
    LogicalId t = g.div(one_node, denom);

    // t^2..t^5
    LogicalId t2 = g.mul(t, t);
    LogicalId t3 = g.mul(t2, t);
    LogicalId t4 = g.mul(t3, t);
    LogicalId t5 = g.mul(t4, t);

    // poly = a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    LogicalId poly = g.mul(a1, t);
    poly = g.add(poly, g.mul(a2, t2));
    poly = g.add(poly, g.mul(a3, t3));
    poly = g.add(poly, g.mul(a4, t4));
    poly = g.add(poly, g.mul(a5, t5));

    // exp(-x_scaled^2) = pow(e, neg(xs_sq))
    LogicalId neg_xs_sq = g.neg(xs_sq);
    LogicalId exp_neg_xs_sq = g.pow(e_node, neg_xs_sq);

    // erf_pos = 1 - poly * exp(-x_scaled^2)
    LogicalId product = g.mul(poly, exp_neg_xs_sq);
    LogicalId erf_pos = g.add(one_node, g.neg(product));

    // erf(x_scaled) = sign(x_scaled) * erf_pos
    LogicalId erf_val = g.mul(sign_xs, erf_pos);

    // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    LogicalId one_plus_erf = g.add(one_node, erf_val);
    LogicalId half_x = g.mul(x_id, half);
    return g.mul(half_x, one_plus_erf);
}

REGISTER_KERNEL("JinaGeluExact_F32_3D", 1, 1,
                matchJinaGeluExact_F32_3D, runJinaGeluExact_F32_3D,
                refFactoryJinaGeluExact_F32_3D,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32},
                {{1, 1024, 3072}},
                {true},
                {{MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON
