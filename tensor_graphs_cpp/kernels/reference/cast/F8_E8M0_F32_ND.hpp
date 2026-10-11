#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
#include <cmath>

inline bool matchCastF8_E8M0_F32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (output.dtype != DType::FLOAT32)
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline float fp8e8m0fnu_to_fp32(uint8_t input)
{
    if (input == 0xFF)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return std::ldexp(1.0f, static_cast<int>(input) - 127);
}

inline void runCastF8_E8M0_F32_ND(const KernelContext &ctx)
{
    const uint8_t *src = static_cast<const uint8_t *>(ctx.inputs[0]);
    float *dst = static_cast<float *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        dst[i] = fp8e8m0fnu_to_fp32(src[i]);
    }
}

REGISTER_REF_KERNEL(OpType::CAST, 1, 1, matchCastF8_E8M0_F32_ND, runCastF8_E8M0_F32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::F8_E8M0}, {{8, 32}}, {true},
                    {{MemSpace(1, HandleType::CPP)}});