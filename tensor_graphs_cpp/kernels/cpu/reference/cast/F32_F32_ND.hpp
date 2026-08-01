#pragma once
#include <cstring>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchCastF32_F32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;

    if (!isContiguous(output))
        return false;

    return true;
}

inline void runCastF32_F32_ND(const KernelContext &ctx)
{
    const float *src = static_cast<const float *>(ctx.inputs[0]);
    float *dst = static_cast<float *>(ctx.outputs[0]);

    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    // F32 to F32 is an identity copy, so a direct memcpy is sufficient.
    std::memcpy(dst, src, numElements * sizeof(float));
}

REGISTER_REF_KERNEL(OpType::CAST, 1, 1, matchCastF32_F32_ND, runCastF32_F32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{8, 32}}, {true},
                    {{MemSpace(1, HandleType::CPP)}});