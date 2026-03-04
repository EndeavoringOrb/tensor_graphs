#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

/**
 * KERNEL: RESHAPE (Copy ND)
 * This kernel handles reshapes where the input and output
 * reside in different memory locations, requiring a physical copy.
 */
inline bool matchReshapeND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;
    return countElements(inputs[0].shape) == countElements(output.shape);
}

inline void runReshapeND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const void *src = inputs[0];
    void *dst = outputs[0];

    uint64_t sizeBytes = countElements(inViews[0].shape) * getDTypeSize(inViews[0].dtype);
    std::memcpy(dst, src, sizeBytes);
}

// Registered as a standard kernel (not inplace)
REGISTER_KERNEL(OpType::RESHAPE, Backend::CPU, matchReshapeND, runReshapeND);