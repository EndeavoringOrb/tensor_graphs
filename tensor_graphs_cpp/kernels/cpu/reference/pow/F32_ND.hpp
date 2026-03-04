#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>

/**
 * KERNEL: POW F32 ND (Generic ND, Contiguous)
 * Performs element-wise power: out = base ^ exponent
 */

inline bool matchPowF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;

    // Check Dtypes
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Check Shapes (Must match for this element-wise operation)
    if (inputs[0].shape != inputs[1].shape || inputs[0].shape != output.shape)
        return false;

    // Check Contiguity
    if (!inputs[0].view.isContiguous() || !inputs[1].view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

inline void runPowF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *base = static_cast<const float *>(inputs[0]);
    const float *exponent = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint64_t numElements = countElements(inViews[0].shape);

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[i] = std::pow(base[i], exponent[i]);
    }
}

// Register as a CPU kernel for the POWER operation
REGISTER_KERNEL(OpType::POWER, Backend::CPU, matchPowF32_ND, runPowF32_ND);