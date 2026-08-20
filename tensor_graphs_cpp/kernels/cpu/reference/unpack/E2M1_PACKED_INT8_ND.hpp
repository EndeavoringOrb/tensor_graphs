#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchUnpackE2M1_PACKED_INT8_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.dtype != DType::E2M1)
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runUnpackE2M1_PACKED_INT8_ND(const KernelContext &ctx)
{
    const uint8_t *src = static_cast<const uint8_t *>(ctx.inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(ctx.outputs[0]);
    uint64_t numElementsSrc = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElementsSrc; ++i)
    {
        uint8_t val = src[i];
        dst[2 * i] = val & 0x0F;
        dst[2 * i + 1] = (val >> 4) & 0x0F;
    }
}

REGISTER_REF_KERNEL(OpType::UNPACK, 1, 1, matchUnpackE2M1_PACKED_INT8_ND, runUnpackE2M1_PACKED_INT8_ND,
                    MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::E2M1_PACKED_INT8}, {{8, 32}},
                    {true}, {{MemSpace(1, HandleType::CPP)}});