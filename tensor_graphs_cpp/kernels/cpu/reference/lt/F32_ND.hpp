#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchLtF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(inputs[0]) || !isContiguous(inputs[1]) || !isContiguous(output))
        return false;
    return output.dtype == DType::BOOL;
}

inline void runLtF32_ND(const KernelContext &ctx)
{
    const float *a = static_cast<const float *>(ctx.inputs[0]);
    const float *b = static_cast<const float *>(ctx.inputs[1]);
    bool *out = static_cast<bool *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[i] = (a[i] < b[i]);
    }
}

REGISTER_REF_KERNEL(OpType::LT, 2, matchLtF32_ND, runLtF32_ND, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{8, 32}, {8, 32}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});

