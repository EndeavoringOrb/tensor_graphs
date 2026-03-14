#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

/**
 * KERNEL: DIVIDE F32 ND (Generic ND, Contiguous)
 */

inline bool matchDivF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2)
        return false;

    // Check Dtypes
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Check Shapes (Must match for this naive element-wise division)
    if (inputs[0].shape != inputs[1].shape || inputs[0].shape != output.shape)
        return false;

    // Check Contiguity
    if (!inputs[0].view.isContiguous() || !inputs[1].view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

inline void runDivF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *a = static_cast<const float *>(inputs[0]);
    const float *b = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint64_t numElements = countElements(inViews[0].shape);

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[i] = a[i] / b[i];
    }
}

// Register as a CPU kernel for the DIVIDE operation
REGISTER_REF_KERNEL(OpType::DIVIDE, Backend::CPU, matchDivF32_ND, runDivF32_ND);