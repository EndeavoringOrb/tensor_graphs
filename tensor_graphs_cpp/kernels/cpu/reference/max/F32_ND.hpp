#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cfloat>

inline bool matchMaxF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs.size() == 2 && inputs[0].dtype == DType::FLOAT32 && output.dtype == DType::FLOAT32;
}

inline void runMaxF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    int32_t axis = *static_cast<const int32_t *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    const auto &inShape = inViews[0].getShape();
    const auto &outShape = outViews[0].getShape();
    int32_t ndim = static_cast<int32_t>(inShape.size());

    // Normalize axis
    if (axis < 0)
        axis += ndim;

    // 1. Initialize output buffer with the smallest possible float
    // We use getStridedIndex to ensure we initialize the correct physical memory
    // if the output is a strided view.
    uint64_t out_count = countElements(outShape);
    for (uint64_t i = 0; i < out_count; ++i)
    {
        out[getStridedIndex(i, outShape, outViews[0].strides)] = -FLT_MAX;
    }

    // 2. Iterate through all elements of the input
    uint64_t in_count = countElements(inShape);
    for (uint64_t i = 0; i < in_count; ++i)
    {
        uint64_t temp = i;
        uint64_t out_phys_idx = 0;

        // Unravel the flat input index 'i' into multi-dimensional coordinates,
        // then map those coordinates to the output's physical memory index.
        for (int32_t d = ndim - 1; d >= 0; --d)
        {
            uint32_t coord = temp % inShape[d];
            temp /= inShape[d];

            // In a reduction, the coordinate of the reduced axis in the output is 0
            // (since the dimension size is 1). All other coordinates match.
            uint32_t out_coord = (d == axis) ? 0 : coord;
            out_phys_idx += (uint64_t)out_coord * outViews[0].strides[d];
        }

        // Access input using system strides and perform the Max reduction
        float val = in[getStridedIndex(i, inShape, inViews[0].strides)];
        if (val > out[out_phys_idx])
        {
            out[out_phys_idx] = val;
        }
    }
}

REGISTER_REF_KERNEL(OpType::MAX, 2, matchMaxF32_ND, runMaxF32_ND, {Backend::CPU}, {DType::FLOAT32, DType::INT32}, {{8, 32}, {1}}, {false, false}, {{Backend::CPU}, {Backend::CPU}});
