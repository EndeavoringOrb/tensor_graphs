#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchAddF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape()) return false;
    return true;
}

inline void runAddF32_ND(const KernelContext &ctx)
{
    const float *a = static_cast<const float *>(ctx.inputs[0]);
    const float *b = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[getStridedIndex(i, ctx.outViews[0].getShape(), ctx.outViews[0].strides)] = 
            a[getStridedIndex(i, ctx.inViews[0].getShape(), ctx.inViews[0].strides)] + 
            b[getStridedIndex(i, ctx.inViews[1].getShape(), ctx.inViews[1].strides)];
    }
}

REGISTER_REF_KERNEL(OpType::ADD, 2, matchAddF32_ND, runAddF32_ND, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{8, 32}, {8, 32}}, {false, false}, {{Backend::CPU}, {Backend::CPU}});

