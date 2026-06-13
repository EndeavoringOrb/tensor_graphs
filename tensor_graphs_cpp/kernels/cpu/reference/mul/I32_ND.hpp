#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchMulI32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    return true;
}

inline void runMulI32_ND(const KernelContext &ctx)
{
    const int32_t *a = static_cast<const int32_t *>(ctx.inputs[0]);
    const int32_t *b = static_cast<const int32_t *>(ctx.inputs[1]);
    int32_t *out = static_cast<int32_t *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[getStridedIndex(i, ctx.outViews[0].getShape(), ctx.outViews[0].strides)] =
            a[getStridedIndex(i, ctx.inViews[0].getShape(), ctx.inViews[0].strides)] *
            b[getStridedIndex(i, ctx.inViews[1].getShape(), ctx.inViews[1].strides)];
    }
}

REGISTER_REF_KERNEL(
    OpType::MUL,
    2,
    matchMulI32_ND,
    runMulI32_ND,
    {Backend::CPU},
    {DType::INT32, DType::INT32},
    {{8, 32}, {8, 32}},
    {false, false},
    {{Backend::CPU}, {Backend::CPU}});