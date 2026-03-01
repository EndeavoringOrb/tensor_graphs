#pragma once
#include "core/types.hpp"

// ---------------------------------------------------------
// KERNEL: ADD F32 1D (Contiguous)
// ---------------------------------------------------------

// Match Function: Verifies DType, Rank (1D), Shape Match, and Contiguity
bool matchAddF32_1D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;

    // Check Dtypes
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Check Ranks (Must be 1D)
    if (inputs[0].shape.size() != 1 || inputs[1].shape.size() != 1 || output.shape.size() != 1)
        return false;

    // Check Dimension Matching
    if (inputs[0].shape[0] != inputs[1].shape[0] || inputs[0].shape[0] != output.shape[0])
        return false;

    // Check Contiguity (Required for this specific kernel optimization)
    if (!inputs[0].view.isContiguous() || !inputs[1].view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

// Run Function: Naive 1D Loop
void runAddF32_1D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                  const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    // Cast raw memory to F32 pointers
    const float *a = static_cast<const float *>(inputs[0]);
    const float *b = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint32_t size = inViews[0].shape[0];

    for (uint32_t i = 0; i < size; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

// Register into the global singleton registry using the Table Pattern
REGISTER_KERNEL(OpType::ADD, Backend::CPU, matchAddF32_1D, runAddF32_1D);
