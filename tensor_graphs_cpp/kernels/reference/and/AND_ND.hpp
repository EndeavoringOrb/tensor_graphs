#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchAnd_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return output.dtype == DType::BOOL;
}

inline void runAnd_ND(const KernelContext &ctx)
{
    const bool *a = static_cast<const bool *>(ctx.inputs[0]);
    const bool *b = static_cast<const bool *>(ctx.inputs[1]);
    bool *out = static_cast<bool *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[i] = a[i] && b[i];
    }
}

REGISTER_REF_KERNEL(OpType::AND, 2, 2, matchAnd_ND, runAnd_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::BOOL, DType::BOOL}, {{8, 32}, {8, 32}}, {true, true},
                    {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
