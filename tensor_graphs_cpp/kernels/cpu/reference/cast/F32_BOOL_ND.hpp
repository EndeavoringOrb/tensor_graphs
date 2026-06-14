#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchCastF32_BOOL_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return output.dtype == DType::BOOL;
}

inline void runCastF32_BOOL_ND(const KernelContext &ctx)
{
    const float *src = static_cast<const float *>(ctx.inputs[0]);
    bool *dst = static_cast<bool *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        dst[i] = (src[i] != 0.0f);
    }
}

REGISTER_REF_KERNEL(OpType::CAST, 1, matchCastF32_BOOL_ND, runCastF32_BOOL_ND, {Backend::CPU}, {DType::FLOAT32}, {{8, 32}}, {true}, {{Backend::CPU}});

