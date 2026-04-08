#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

/**
 * KERNEL: NEGATE F32 ND (Generic ND, Contiguous)
 * Performs element-wise negation: out = -x
 */

inline bool matchNegF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 1)
        return false;

    // Check Dtypes
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Check Shapes (Must match)
    if (inputs[0].getShape() != output.getShape())
        return false;

    return true;
}

inline void runNegF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *x = static_cast<const float *>(inputs[0]);
    float *out = static_cast<float *>(outputs[0]);

    uint64_t numElements = countElements(inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[getStridedIndex(i, outViews[0].getShape(), outViews[0].strides)] = -x[getStridedIndex(i, inViews[0].getShape(), inViews[0].strides)];
    }
}

// Register as a CPU kernel for the NEGATE operation
REGISTER_REF_KERNEL(OpType::NEGATE, 1, matchNegF32_ND, runNegF32_ND, {Backend::CPU}, {DType::FLOAT32}, {{8, 32}}, {false}, {{Backend::CPU}});