#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>

inline bool matchCosF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return isContiguous(output);
}

inline void runCosF32_ND(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.outViews[0].getShape());
    for (uint64_t i = 0; i < n; ++i)
        out[i] = std::cos(in[i]);
}

REGISTER_REF_KERNEL(OpType::COS, 1, matchCosF32_ND, runCosF32_ND, {Backend::CPU}, {DType::FLOAT32}, {{8, 32}}, {true}, {{Backend::CPU}});
