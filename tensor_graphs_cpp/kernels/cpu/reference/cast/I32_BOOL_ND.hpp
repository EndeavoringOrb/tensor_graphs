#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchCastI32_BOOL_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(inputs[0]) || !isContiguous(output))
        return false;
    return output.dtype == DType::BOOL;
}

inline void runCastI32_BOOL_ND(const KernelContext &ctx)
{
    const int32_t *src = static_cast<const int32_t *>(ctx.inputs[0]);
    bool *dst = static_cast<bool *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        dst[i] = (src[i] != 0);
    }
}

REGISTER_REF_KERNEL(OpType::CAST, 1, matchCastI32_BOOL_ND, runCastI32_BOOL_ND, {Backend::CPU}, {DType::INT32}, {{8, 32}}, {true}, {{Backend::CPU}});

