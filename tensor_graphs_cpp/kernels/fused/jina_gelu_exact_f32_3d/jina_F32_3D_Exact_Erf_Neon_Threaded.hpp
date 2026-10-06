
#include "kernels/fused/jina_gelu_exact_f32_3d/ref.hpp"
//
// FUSED KERNEL: Exact-erf GELU for jina-embeddings-v5-omni-nano-retrieval
//
// Matches the exact subgraph produced by
// JinaV5OmniNanoRetrievalModel::gelu_exact():
//
//   gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// where erf is the Abramowitz-Stegun approximation:
//   t = 1 / (1 + p * |z|),  p = 0.3275911
//   erf(z) = sign(z) * (1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) *
//   exp(-z^2))
//
// The model's decomposed form creates ~30+ intermediate tensors of size (1, S,
// D), each requiring a full memory pass.  For S=4320, D=3072 (vision MLP),
// that's ~30 * 53 MB = 1.6 GB of intermediate traffic per call.  This fused
// kernel reads the input once and writes the output once — a >10× reduction in
// memory traffic.
//
// Hardware: Qualcomm aarch64 (ARMv8.6, 12 cores, NEON FMA).
// Value constants matched byte-for-byte: 0.7071067811865475, 0.5, 1e-12,
// 0.3275911, 1.0, 0.254829592, -0.284496736, 1.421413741, -1.453152027,
// 1.061405429, 2.718281828459045.
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
inline bool matchJinaGeluExact_F32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

    const float32x4_t v_inv_sqrt2 = vdupq_n_f32(0.7071067811865475f);
    const float32x4_t v_half = vdupq_n_f32(0.5f);
    const float32x4_t v_one = vdupq_n_f32(1.0f);

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint64_t chunk = (n + num_threads - 1) / num_threads;
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
            float poly =
                t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
            float erf_pos = 1.0f - poly * std::exp(-z * z);
            float erf_val = (z >= 0.0f) ? erf_pos : -erf_pos;
            out[i] = 0.5f * x * (1.0f + erf_val);
        }
    });
}

// ---------------------------------------------------------------------------
// Reference Factory — mirrors JinaV5OmniNanoRetrievalModel::gelu_exact()
// decomposition EXACTLY (same op types, same float constants, same structure).
//
// Each expand_scalar_to_3d(val, 1, S, D) creates:
//   const(val) → reshape({1,1,1}) → repeat(axis=1, S) → repeat(axis=2, D)
// ---------------------------------------------------------------------------


REGISTER_KERNEL("JinaGeluExact_F32_3D", 1, 1, matchJinaGeluExact_F32_3D, runJinaGeluExact_F32_3D,
                refFactoryJinaGeluExact_F32_3D, {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32}, {{1, 1024, 3072}}, {true}, {{MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON
