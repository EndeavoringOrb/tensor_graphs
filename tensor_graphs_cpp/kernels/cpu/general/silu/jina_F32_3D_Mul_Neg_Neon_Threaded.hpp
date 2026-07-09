// File: tensor_graphs_cpp/kernels/cpu/general/silu/jina_F32_3D_Mul_Neg_Neon_Threaded.hpp
//
// FUSED KERNEL: SiLU for jina-embeddings-v5-omni-nano-retrieval
//
// Matches the exact subgraph produced by JinaV5OmniNanoRetrievalModel::silu_atomic():
//
//   neg_one  = expand_scalar_to_3d(-1.0, N, L, D)   // {1, S, D}  (note: N=1)
//   neg_x    = mul(x, neg_one)                      = -x          // {1, S, D}
//   e_node   = expand_scalar_to_3d(2.718281828459045, 1, L, D)   // {1, S, D}
//   exp_neg  = pow(e_node, neg_x)                    = exp(-x)    // {1, S, D}
//   one_node = expand_scalar_to_3d(1.0, 1, L, D)                  // {1, S, D}
//   den      = add(one_node, exp_neg)                             // {1, S, D}
//   sig      = div(one_node, den)                                 // {1, S, D}
//   result   = mul(x, sig)                                        = silu(x)
//
// NOTE: jina's silu_atomic uses mul(x, -1) for negation (NOT neg(x)).
// The existing Silu_3D_* kernels use neg(x), so they do NOT match this
// subgraph.  This kernel specifically matches the mul-by-(-1) pattern.
//
// Hardware: Qualcomm aarch64 (ARMv8.6, 12 cores, NEON FMA).
// Value constants matched byte-for-byte: -1.0, 2.718281828459045, 1.0.
// TODO: remove this once we have mul(x, -1) -> neg(x) rewrite rule which will match other silu kernels
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

// NEON vectorized exp(x) — same implementation as the GELU kernel.
static inline float32x4_t jina_silu_exp_neon(float32x4_t x)
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

inline bool matchJinaSiluMulNeg_F32_3D(const std::vector<TensorNode> &inputs,
                                       const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3)
        return false;
    return isContiguous(output);
}

// Numerically stable SiLU: x / (1 + exp(-|x|)) * sign(x) ... actually
// for the fused kernel we use: x * sigmoid(x) where sigmoid(x) = 1/(1+exp(-x)).
// For x >= 0: exp(-x) in (0, 1], no overflow.
// For x < 0:  use exp(x) / (1 + exp(x)) trick to avoid overflow of exp(-x).
// Both paths give the same result as the reference pow(e, -x) decomposition
// within ~1 ULP, well within the 1e-4 test tolerance.
inline void runJinaSiluMulNeg_F32_3D(const KernelContext &ctx)
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

    const float32x4_t v_one = vdupq_n_f32(1.0f);
    const float32x4_t v_zero = vdupq_n_f32(0.0f);

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint64_t start = t * chunk;
            uint64_t end = std::min(start + chunk, n);

            uint64_t i = start;
            for (; i + 16 <= end; i += 16)
            {
                float32x4_t v_x0 = vld1q_f32(in + i);
                float32x4_t v_x1 = vld1q_f32(in + i + 4);
                float32x4_t v_x2 = vld1q_f32(in + i + 8);
                float32x4_t v_x3 = vld1q_f32(in + i + 12);

                // For each lane: silu(x) = x / (1 + exp(-x))
                // Use stable formulation:
                //   x >= 0: x / (1 + exp(-x))
                //   x <  0: x * exp(x) / (1 + exp(x))
                auto silu_vec = [&](float32x4_t v_x) -> float32x4_t {
                    float32x4_t v_neg_x = vnegq_f32(v_x);
                    float32x4_t v_exp_neg = jina_silu_exp_neon(v_neg_x);
                    float32x4_t v_den = vaddq_f32(v_one, v_exp_neg);
                    // x / (1 + exp(-x)) — stable for x >= 0.
                    // For x < 0, exp(-x) can overflow for x < -88.
                    // Use: x * exp(x) / (1 + exp(x)) for x < 0.
                    float32x4_t v_exp_pos = jina_silu_exp_neon(v_x);
                    float32x4_t v_den_neg = vaddq_f32(v_one, v_exp_pos);
                    float32x4_t v_num_neg = vmulq_f32(v_x, v_exp_pos);
                    float32x4_t v_res_pos = vdivq_f32(v_x, v_den);
                    float32x4_t v_res_neg = vdivq_f32(v_num_neg, v_den_neg);
                    uint32x4_t v_neg_mask = vcltq_f32(v_x, v_zero);
                    return vbslq_f32(v_neg_mask, v_res_neg, v_res_pos);
                };

                vst1q_f32(out + i, silu_vec(v_x0));
                vst1q_f32(out + i + 4, silu_vec(v_x1));
                vst1q_f32(out + i + 8, silu_vec(v_x2));
                vst1q_f32(out + i + 12, silu_vec(v_x3));
            }
            for (; i + 4 <= end; i += 4)
            {
                float32x4_t v_x = vld1q_f32(in + i);
                float32x4_t v_neg_x = vnegq_f32(v_x);
                float32x4_t v_exp_neg = jina_silu_exp_neon(v_neg_x);
                float32x4_t v_den = vaddq_f32(v_one, v_exp_neg);
                float32x4_t v_exp_pos = jina_silu_exp_neon(v_x);
                float32x4_t v_den_neg = vaddq_f32(v_one, v_exp_pos);
                float32x4_t v_num_neg = vmulq_f32(v_x, v_exp_pos);
                float32x4_t v_res_pos = vdivq_f32(v_x, v_den);
                float32x4_t v_res_neg = vdivq_f32(v_num_neg, v_den_neg);
                uint32x4_t v_neg_mask = vcltq_f32(v_x, v_zero);
                vst1q_f32(out + i, vbslq_f32(v_neg_mask, v_res_neg, v_res_pos));
            }
            for (; i < end; ++i)
            {
                float x = in[i];
                float r;
                if (x >= 0.0f)
                    r = x / (1.0f + std::exp(-x));
                else
                {
                    float ex = std::exp(x);
                    r = x * ex / (1.0f + ex);
                }
                out[i] = r;
            } });
    }
    for (auto &worker : workers)
        worker.join();
}

// Reference Factory — mirrors JinaV5OmniNanoRetrievalModel::silu_atomic()
// decomposition EXACTLY.
//
// silu_atomic decomposition:
//   neg_one  = expand_scalar_to_3d(-1.0, 1, L, D)    // {1, S, D}
//   neg_x    = mul(x, neg_one)                       // {1, S, D}   <-- mul, NOT neg
//   e_node   = expand_scalar_to_3d(2.718281828459045, 1, L, D)
//   exp_neg  = pow(e_node, neg_x)
//   one_node = expand_scalar_to_3d(1.0, 1, L, D)
//   den      = add(one_node, exp_neg)
//   sig      = div(one_node, den)
//   result   = mul(x, sig)
inline uint32_t refFactoryJinaSiluMulNeg_F32_3D(const std::vector<uint32_t> &inputs,
                                                Graph &g)
{
    uint32_t x_id = inputs[0];
    const auto &shape = g.getNode(x_id).getShape();
    uint32_t B = shape[0];
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    // Helper: expand_scalar_to_3d(val, 1, S, D) → {1, S, D}
    auto expand_scalar_SD = [&](float val) -> uint32_t
    {
        uint32_t node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh[] = {1, 1, 1};
        uint32_t out = g.reshape(node, g.constant({3}, sh, DType::INT32));
        if (B > 1)
        {
            int32_t rep = (int32_t)B;
            int32_t ax = 0;
            out = g.repeat(out,
                           g.constant({1}, &rep, DType::INT32),
                           g.constant({1}, &ax, DType::INT32));
        }
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

    // neg_one = expand_scalar_to_3d(-1.0, 1, S, D)
    uint32_t neg_one = expand_scalar_SD(-1.0f);

    // neg_x = mul(x, neg_one)   <-- mul by -1, NOT neg(x)
    uint32_t neg_x = g.mul(x_id, neg_one);

    // e_node = expand_scalar_to_3d(2.718281828459045, 1, S, D)
    uint32_t e_node = expand_scalar_SD(2.718281828459045f);

    // exp_neg = pow(e, -x)
    uint32_t exp_neg = g.pow(e_node, neg_x);

    // one_node = expand_scalar_to_3d(1.0, 1, S, D)
    uint32_t one_node = expand_scalar_SD(1.0f);

    // den = 1 + exp(-x)
    uint32_t den = g.add(one_node, exp_neg);

    // sig = 1 / den
    uint32_t sig = g.div(one_node, den);

    // result = x * sig
    return g.mul(x_id, sig);
}

REGISTER_KERNEL("JinaSiluMulNeg_F32_3D", 1,
                matchJinaSiluMulNeg_F32_3D, runJinaSiluMulNeg_F32_3D,
                refFactoryJinaSiluMulNeg_F32_3D,
                {Backend::CPU},
                {DType::FLOAT32},
                {{1, 1024, 3072}},
                {true},
                {{Backend::CPU}});

#endif // TG_HAS_NEON
