#pragma once

#include <cmath>

#include "core/kernels.hpp"

// Gemma RMS norm exactly as models/gemma-3-270m.hpp builds it: the scalars n
// (the normalized size), eps, 1/2 and 1 enter through fills, and the weight
// through reshape + repeat. The scalars are kernel inputs, so one pattern
// covers every normalized size and every row slice (fill shapes and repeat
// counts are constant arguments and match any value).
//
// inputs: x [B, S, D], w [D], n [1], eps [1], half [1], one [1]
// out = (x * (one / (sum(x*x, -1) / n + eps)^half)) * (w + one)
//
// `heads` adds the repeat of the weight over B (per-head q/k norms, B > 1).
inline LogicalId gemmaRmsNormFillPattern(const std::vector<LogicalId> &inputs, Graph &g, bool heads)
{
    const LogicalId x = inputs[0], w = inputs[1], n = inputs[2], eps = inputs[3], half = inputs[4], one = inputs[5];
    const auto shape = g.getNode(x).getShape();
    const uint32_t B = shape[0], S = shape[1], D = shape[2];

    LogicalId x_sq = g.mul(x, x);
    int32_t axis_val = -1;
    LogicalId sum_sq = g.sum(x_sq, g.constant({1}, &axis_val, DType::INT32));
    LogicalId mean_sq = g.div(sum_sq, g.fill(n, {B, S, 1}));
    LogicalId mean_sq_plus_eps = g.add(mean_sq, g.fill(eps, {B, S, 1}));
    LogicalId std = g.pow(mean_sq_plus_eps, g.fill(half, {B, S, 1}));
    LogicalId inv_std = g.div(g.fill(one, {B, S, 1}), std);
    LogicalId x_norm = g.mul(x, g.repeat(inv_std, D, 2));

    int32_t shape_3d[] = {1, 1, (int32_t)D};
    LogicalId w3 = g.reshape(w, g.constant({3}, shape_3d, DType::INT32));
    if (heads)
        w3 = g.repeat(w3, B, 0);
    w3 = g.repeat(w3, S, 1);
    LogicalId scale = g.add(w3, g.fill(one, {B, S, D}));
    return g.mul(x_norm, scale);
}

inline LogicalId refFactoryGemmaRmsNormFillRows(const std::vector<LogicalId> &inputs, Graph &g)
{
    return gemmaRmsNormFillPattern(inputs, g, false);
}

inline LogicalId refFactoryGemmaRmsNormFillHeads(const std::vector<LogicalId> &inputs, Graph &g)
{
    return gemmaRmsNormFillPattern(inputs, g, true);
}

inline bool matchGemmaRmsNormFill(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 6)
        return false;
    const auto &sx = inputs[0].getShape();
    if (sx.size() != 3 || inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != sx[2])
        return false;
    for (size_t i = 2; i < 6; ++i)
        if (countElements(inputs[i].getShape()) != 1)
            return false;
    return output.getShape() == sx && isContiguous(output) && isContiguous(inputs[0]) && isContiguous(inputs[1]);
}

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

// Rows [row_begin, row_end) of the [rows, D] view.
inline void gemmaRmsNormFillRowsNeon(const float *x, const float *w, float n, float eps, float half, float one,
                                     float *out, uint32_t D, uint64_t row_begin, uint64_t row_end)
{
    const uint32_t d4 = D & ~3u;
    const float32x4_t onev = vdupq_n_f32(one);
    for (uint64_t r = row_begin; r < row_end; ++r)
    {
        const float *xr = x + r * D;
        float *orow = out + r * D;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (uint32_t d = 0; d < d4; d += 4)
        {
            const float32x4_t v = vld1q_f32(xr + d);
            acc = vfmaq_f32(acc, v, v);
        }
        float ss = vaddvq_f32(acc);
        for (uint32_t d = d4; d < D; ++d)
            ss += xr[d] * xr[d];
        const float inv = one / std::pow(ss / n + eps, half);
        const float32x4_t invv = vdupq_n_f32(inv);
        for (uint32_t d = 0; d < d4; d += 4)
        {
            const float32x4_t xn = vmulq_f32(vld1q_f32(xr + d), invv);
            vst1q_f32(orow + d, vmulq_f32(xn, vaddq_f32(vld1q_f32(w + d), onev)));
        }
        for (uint32_t d = d4; d < D; ++d)
            orow[d] = (xr[d] * inv) * (w[d] + one);
    }
}
#endif
