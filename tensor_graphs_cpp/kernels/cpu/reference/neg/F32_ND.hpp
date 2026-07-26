#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

/**
 * KERNEL: NEGATE F32 ND (Generic ND, Contiguous)
 * Performs element-wise negation: out = -x
 */

inline bool matchNegF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Check Shapes (Must match)
    if (inputs[0].getShape() != output.getShape())
        return false;

    return true;
}

inline void runNegF32_ND(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[getStridedIndex(i, ctx.outViews[0].getShape(), ctx.outViews[0].strides)] =
            -x[getStridedIndex(i, ctx.inViews[0].getShape(), ctx.inViews[0].strides)];
    }
}

// Register as a CPU kernel for the NEGATE operation
REGISTER_REF_KERNEL(OpType::NEGATE, 1, 1, matchNegF32_ND, runNegF32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{8, 32}}, {false},
                    {{MemSpace(1, HandleType::CPP)}});
