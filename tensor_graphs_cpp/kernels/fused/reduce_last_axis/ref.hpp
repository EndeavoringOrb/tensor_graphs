#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/kernels.hpp"
#include "kernels/fused/max_f32_nd/ref.hpp"
#include "kernels/fused/sum_f32_nd/ref.hpp"

// Reductions over the last axis: input [..., N] with a unit last stride and
// any strides on the leading axes (row views need no copy), output [..., 1]
// contiguous. Matches only when the last axis is the one reduced: the output
// shape equals the input shape except for a last dimension of 1 (and the
// input's last dimension is above 1).
inline bool matchReduceLastAxis(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;
    const auto &in = inputs[0].getShape();
    const auto &out = output.getShape();
    if (in.empty() || in.size() > 4 || out.size() != in.size() || in.back() < 2 || out.back() != 1)
        return false;
    for (size_t d = 0; d + 1 < in.size(); ++d)
        if (in[d] != out[d])
            return false;
    return inputs[0].strides.back() == 1 && isContiguous(output);
}

// Row r of the leading axes -> element offset of the row start.
struct ReduceRows
{
    uint64_t rows = 1;
    uint32_t n = 0;
    std::vector<uint32_t> lead;   // leading dims
    std::vector<uint64_t> stride; // their strides (elements)

    explicit ReduceRows(const TensorView &view)
    {
        const auto &shape = view.getShape();
        if (view.strides.size() != shape.size() || shape.empty())
            Error::throw_err("[ReduceLastAxis] view strides do not match its shape");
        if (view.strides.back() != 1)
            Error::throw_err("[ReduceLastAxis] last axis must have unit stride");
        n = shape.back();
        for (size_t d = 0; d + 1 < shape.size(); ++d)
        {
            lead.push_back(shape[d]);
            stride.push_back(view.strides[d]);
            rows *= shape[d];
        }
    }

    uint64_t offset(uint64_t r) const
    {
        uint64_t off = 0;
        for (size_t d = lead.size(); d-- > 0;)
        {
            off += (r % lead[d]) * stride[d];
            r /= lead[d];
        }
        return off;
    }
};

inline void checkReduceAxis(const KernelContext &ctx)
{
    const int32_t rank = static_cast<int32_t>(ctx.inViews[0].getShape().size());
    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[1]);
    if (axis < 0)
        axis += rank;
    if (axis != rank - 1)
        Error::throw_err("[ReduceLastAxis] kernel reduces the last axis only");
}

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline void sumLastAxisRows(const float *in, float *out, const ReduceRows &rows, uint64_t begin, uint64_t end)
{
    const uint32_t n4 = rows.n & ~3u;
    for (uint64_t r = begin; r < end; ++r)
    {
        const float *row = in + rows.offset(r);
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (uint32_t i = 0; i < n4; i += 4)
            acc = vaddq_f32(acc, vld1q_f32(row + i));
        float s = vaddvq_f32(acc);
        for (uint32_t i = n4; i < rows.n; ++i)
            s += row[i];
        out[r] = s;
    }
}

inline void maxLastAxisRows(const float *in, float *out, const ReduceRows &rows, uint64_t begin, uint64_t end)
{
    const uint32_t n4 = rows.n & ~3u;
    for (uint64_t r = begin; r < end; ++r)
    {
        const float *row = in + rows.offset(r);
        float m = -std::numeric_limits<float>::infinity();
        if (n4 > 0)
        {
            float32x4_t acc = vld1q_f32(row);
            for (uint32_t i = 4; i < n4; i += 4)
                acc = vmaxq_f32(acc, vld1q_f32(row + i));
            m = vmaxvq_f32(acc);
        }
        for (uint32_t i = n4; i < rows.n; ++i)
            m = std::max(m, row[i]);
        out[r] = m;
    }
}
#endif
