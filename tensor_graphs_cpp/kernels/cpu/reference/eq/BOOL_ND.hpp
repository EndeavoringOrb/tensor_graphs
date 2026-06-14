#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchEqBool_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(inputs[0]) || !isContiguous(inputs[1]) || !isContiguous(output))
        return false;
    return output.dtype == DType::BOOL;
}

inline void runEqBool_ND(const KernelContext &ctx)
{
    const bool *a = static_cast<const bool *>(ctx.inputs[0]);
    const bool *b = static_cast<const bool *>(ctx.inputs[1]);
    bool *out = static_cast<bool *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[i] = (a[i] == b[i]);
    }
}

REGISTER_REF_KERNEL(OpType::EQ, 2, matchEqBool_ND, runEqBool_ND, {Backend::CPU}, {DType::BOOL, DType::BOOL}, {{8, 32}, {8, 32}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});

