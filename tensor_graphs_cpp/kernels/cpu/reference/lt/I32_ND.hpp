#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchLtI32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return output.dtype == DType::BOOL;
}

inline void runLtI32_ND(const KernelContext &ctx)
{
    const int32_t *a = static_cast<const int32_t *>(ctx.inputs[0]);
    const int32_t *b = static_cast<const int32_t *>(ctx.inputs[1]);
    bool *out = static_cast<bool *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[i] = (a[i] < b[i]);
    }
}

REGISTER_REF_KERNEL(OpType::LT, 2, matchLtI32_ND, runLtI32_ND, {Backend::CPU}, {DType::INT32, DType::INT32}, {{8, 32}, {8, 32}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});