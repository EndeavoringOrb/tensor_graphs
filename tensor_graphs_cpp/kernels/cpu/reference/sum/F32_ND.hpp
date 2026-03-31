#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchSumF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    return inputs.size() == 2 && inputs[0].dtype == DType::FLOAT32 && output.dtype == DType::FLOAT32 && inputs[1].dtype == DType::INT32;
}

inline void runSumF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    int32_t axis = *static_cast<const int32_t *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    const auto &inShape = inViews[0].getShape();
    const auto &outShape = outViews[0].getShape();
    int ndim = static_cast<int>(inShape.size());
    if (axis < 0)
        axis += ndim;

    // Initialize output with 0 safely
    uint64_t out_count = countElements(outShape);
    for (uint64_t i = 0; i < out_count; ++i)
    {
        out[getStridedIndex(i, outShape, outViews[0].strides)] = 0.0f;
    }

    uint64_t in_count = countElements(inShape);
    for (uint64_t i = 0; i < in_count; ++i)
    {
        uint64_t temp = i;
        uint64_t out_phys_idx = 0;

        // Map input flat index 'i' to output physical offset
        for (int d = ndim - 1; d >= 0; --d)
        {
            uint32_t coord = temp % inShape[d];
            temp /= inShape[d];
            // If d is the reduction axis, it contributes to output coord 0 (since dim is 1)
            uint32_t out_coord = (d == axis) ? 0 : coord;
            out_phys_idx += (uint64_t)out_coord * outViews[0].strides[d];
        }

        out[out_phys_idx] += in[getStridedIndex(i, inShape, inViews[0].strides)];
    }
}

REGISTER_REF_KERNEL(OpType::SUM, matchSumF32_ND, runSumF32_ND, {Backend::CPU});