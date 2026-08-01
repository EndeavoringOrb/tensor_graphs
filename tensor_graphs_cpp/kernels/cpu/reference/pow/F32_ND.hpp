#pragma once
#include <cmath>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchPowF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    return true;
}

inline void runPowF32_ND(const KernelContext &ctx)
{
    const float *base = static_cast<const float *>(ctx.inputs[0]);
    const float *exponent = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[getStridedIndex(i, ctx.outViews[0].getShape(), ctx.outViews[0].strides)] =
            std::pow(base[getStridedIndex(i, ctx.inViews[0].getShape(), ctx.inViews[0].strides)],
                     exponent[getStridedIndex(i, ctx.inViews[1].getShape(), ctx.inViews[1].strides)]);
    }
}

REGISTER_REF_KERNEL(OpType::POWER, 2, 2, matchPowF32_ND, runPowF32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32}, {{8, 32}, {8, 32}}, {false, false},
                    {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
