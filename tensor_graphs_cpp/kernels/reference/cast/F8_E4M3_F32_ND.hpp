#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
#include <cmath>
#include <limits>

inline bool matchCastF8_E4M3_F32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (output.dtype != DType::FLOAT32)
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline float fp8e4m3fn_to_fp32(uint8_t input)
{
    if (input == 0x7F || input == 0xFF)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float sign = (input & 0x80) ? -1.0f : 1.0f;
    uint32_t exp = (input & 0x78) >> 3;
    uint32_t mant = input & 0x07;
    if (exp == 0)
    {
        if (mant == 0)
        {
            return sign * 0.0f;
        }
        else
        {
            return sign * std::ldexp(static_cast<float>(mant), -9);
        }
    }
    else
    {
        return sign * std::ldexp(1.0f + static_cast<float>(mant) * 0.125f, static_cast<int>(exp) - 7);
    }
}

inline void runCastF8_E4M3_F32_ND(const KernelContext &ctx)
{
    const uint8_t *src = static_cast<const uint8_t *>(ctx.inputs[0]);
    float *dst = static_cast<float *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        dst[i] = fp8e4m3fn_to_fp32(src[i]);
    }
}

REGISTER_REF_KERNEL(OpType::CAST, 1, 1, matchCastF8_E4M3_F32_ND, runCastF8_E4M3_F32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::F8_E4M3}, {{8, 32}}, {true},
                    {{MemSpace(1, HandleType::CPP)}});