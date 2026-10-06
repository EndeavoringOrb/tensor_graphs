#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchNot_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return output.dtype == DType::BOOL;
}

inline void runNot_ND(const KernelContext &ctx)
{
    const bool *a = static_cast<const bool *>(ctx.inputs[0]);
    bool *out = static_cast<bool *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[i] = !a[i];
    }
}

REGISTER_REF_KERNEL(OpType::NOT, 1, 1, matchNot_ND, runNot_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::BOOL}, {{8, 32}}, {true}, {{MemSpace(1, HandleType::CPP)}});
