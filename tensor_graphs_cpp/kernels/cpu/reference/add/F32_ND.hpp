#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

/**
 * KERNEL: ADD F32 ND (Generic ND, Contiguous)
 */

inline bool matchAddF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2)
        return false;

    // Check Dtypes
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Check Shapes (Must match for this element-wise addition)
    if (inputs[0].shape != inputs[1].shape || inputs[0].shape != output.shape)
        return false;

    // Check Contiguity
    if (!inputs[0].view.isContiguous() || !inputs[1].view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

inline void runAddF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *a = static_cast<const float *>(inputs[0]);
    const float *b = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    // Use countElements to get total size regardless of rank
    uint64_t numElements = countElements(inViews[0].shape);

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[i] = a[i] + b[i];
    }
}

// Register as a CPU kernel for the ADD operation
REGISTER_REF_KERNEL(OpType::ADD, Backend::CPU, matchAddF32_ND, runAddF32_ND);