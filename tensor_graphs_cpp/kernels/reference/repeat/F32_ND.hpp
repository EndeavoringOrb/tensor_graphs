#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

/**
 * KERNEL: REPEAT F32 ND (Standard)
 * Replicates a tensor along a specific axis into a new allocation.
 */

inline bool matchRepeatF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 3)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[1].dtype != DType::INT32 || inputs[2].dtype != DType::INT32)
        return false;
    if (!inputs[0].view.isContiguous() || !output.view.isContiguous())
        return false;
    return true;
}

inline void runRepeatF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                            const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *src = static_cast<const float *>(inputs[0]);
    int32_t repeats = *static_cast<const int32_t *>(inputs[1]);
    int32_t axis = *static_cast<const int32_t *>(inputs[2]);
    float *dst = static_cast<float *>(outputs[0]);

    // Safety check: The standard kernel expects distinct buffers.
    if (src == dst)
        return;

    int32_t ndim = static_cast<int32_t>(inViews[0].shape.size());
    if (axis < 0)
        axis += ndim;

    uint64_t outer_size = 1;
    for (int32_t i = 0; i < axis; ++i)
        outer_size *= inViews[0].shape[i];

    uint64_t inner_size = 1;
    for (int32_t i = axis + 1; i < ndim; ++i)
        inner_size *= inViews[0].shape[i];

    uint64_t dim_size = inViews[0].shape[axis];
    uint64_t dst_idx = 0;

    for (uint64_t o = 0; o < outer_size; ++o)
    {
        for (uint64_t d = 0; d < dim_size; ++d)
        {
            const float *chunk = src + (o * dim_size + d) * inner_size;
            for (int32_t r = 0; r < repeats; ++r)
            {
                std::memcpy(dst + dst_idx, chunk, inner_size * sizeof(float));
                dst_idx += inner_size;
            }
        }
    }
}

REGISTER_KERNEL(OpType::REPEAT, Backend::CPU, matchRepeatF32_ND, runRepeatF32_ND);