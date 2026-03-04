#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

/**
 * KERNEL: REPEAT F32 ND (In-place/Identity)
 * Handles the case where repeats=1, allowing for a zero-copy operation.
 */

inline bool matchRepeatF32_Inplace_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 3)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Identity check: repeat only supports in-place if the shape doesn't change
    if (inputs[0].shape != output.shape)
        return false;
    if (inputs[0].view.baseOffset != output.view.baseOffset)
        return false;

    return true;
}

inline void runRepeatF32_Inplace_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                    const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    return;
}

REGISTER_KERNEL_INPLACE(OpType::REPEAT, Backend::CPU, matchRepeatF32_Inplace_ND, runRepeatF32_Inplace_ND);