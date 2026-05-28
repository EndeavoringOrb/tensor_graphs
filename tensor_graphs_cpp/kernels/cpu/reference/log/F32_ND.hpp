// File: tensor_graphs_cpp/kernels/cpu/reference/log/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>

inline bool matchLogF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return isContiguous(output);
}

inline void runLogF32_ND(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.outViews[0].getShape());
    for (uint64_t i = 0; i < n; ++i)
        out[i] = std::log(in[i]);
}

REGISTER_REF_KERNEL(OpType::LOG, 1, matchLogF32_ND, runLogF32_ND, {Backend::CPU}, {DType::FLOAT32}, {{8, 32}}, {true}, {{Backend::CPU}});