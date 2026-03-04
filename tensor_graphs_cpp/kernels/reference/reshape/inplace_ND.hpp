#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

/**
 * KERNEL: RESHAPE (In-place ND)
 * This kernel handles reshapes where the planner has aliased
 * the input and output memory. No data movement is required.
 */
inline bool matchReshapeInplaceND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;
    if (inputs[0].view.baseOffset != output.view.baseOffset)
        return false;
    return countElements(inputs[0].shape) == countElements(output.shape);
}

inline void runReshapeInplaceND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    // Logic: In-place reshapes are NO-OPs at runtime as the metadata
    // change is handled by the framework.
    return;
}

// Registered as INPLACE to tell the planner this kernel can reuse the input buffer
REGISTER_KERNEL_INPLACE(OpType::RESHAPE, Backend::CPU, matchReshapeInplaceND, runReshapeInplaceND);