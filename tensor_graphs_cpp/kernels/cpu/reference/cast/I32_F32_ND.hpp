#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

/**
 * KERNEL: CAST INT32 -> FLOAT32 (ND, Contiguous)
 * ---------------------------------------------------------
 * This kernel performs a standard numerical cast from 32-bit
 * integers to 32-bit floating point numbers.
 */

inline bool matchCastI32_F32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Check Shape Identity
    if (inputs[0].getShape() != output.getShape())
        return false;

    // Reference implementation requires contiguity
    if (!isContiguous(output))
        return false;

    return true;
}

inline void runCastI32_F32_ND(const KernelContext &ctx)
{
    const int32_t *src = static_cast<const int32_t *>(ctx.inputs[0]);
    float *dst = static_cast<float *>(ctx.outputs[0]);

    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        dst[i] = static_cast<float>(src[i]);
    }
}

// Register as a CPU kernel for the CAST operation
REGISTER_REF_KERNEL(OpType::CAST, 1, 1, matchCastI32_F32_ND, runCastI32_F32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::INT32}, {{8, 32}}, {true}, {{MemSpace(1, HandleType::CPP)}});
