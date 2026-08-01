#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchCastBOOL_F32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return output.dtype == DType::FLOAT32;
}

inline void runCastBOOL_F32_ND(const KernelContext &ctx)
{
    const bool *src = static_cast<const bool *>(ctx.inputs[0]);
    float *dst = static_cast<float *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        dst[i] = src[i] ? 1.0f : 0.0f;
    }
}

REGISTER_REF_KERNEL(OpType::CAST, 1, 1, matchCastBOOL_F32_ND, runCastBOOL_F32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::BOOL}, {{8, 32}}, {true}, {{MemSpace(1, HandleType::CPP)}});
