#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

/**
 * KERNEL: RESHAPE (Generic ND)
 * Handles both copy-reshapes and in-place reshapes.
 */
inline bool matchReshapeND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (inputs.size() != 2) return false;
    return countElements(inputs[0].shape) == countElements(output.shape);
}

inline void runReshapeND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                        const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews) {
    const void* src = inputs[0];
    void* dst = outputs[0];

    // If the framework optimized this to be in-place, pointers will alias.
    if (src == dst) {
        return; 
    }

    // Otherwise, we must copy the data to the new allocation.
    uint64_t sizeBytes = countElements(inViews[0].shape) * getDTypeSize(inViews[0].dtype);
    std::memcpy(dst, src, sizeBytes);
}

REGISTER_KERNEL_INPLACE(OpType::RESHAPE, Backend::CPU, matchReshapeND, runReshapeND);