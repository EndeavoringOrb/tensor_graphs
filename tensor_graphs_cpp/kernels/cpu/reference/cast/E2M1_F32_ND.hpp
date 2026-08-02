#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchCastE2M1_F32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (inputs[0].dtype != DType::E2M1 || output.dtype != DType::FLOAT32)
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

static const float FP4_TABLE[16] = {0.0f, 0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

inline void runCastE2M1_F32_ND(const KernelContext &ctx)
{
    const uint8_t *src = static_cast<const uint8_t *>(ctx.inputs[0]);
    float *dst = static_cast<float *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        uint8_t val = src[i] & 0x0F;
        dst[i] = FP4_TABLE[val];
    }
}

REGISTER_REF_KERNEL(OpType::CAST, 1, 1, matchCastE2M1_F32_ND, runCastE2M1_F32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::E2M1}, {{8, 32}}, {true}, {{MemSpace(1, HandleType::CPP)}});