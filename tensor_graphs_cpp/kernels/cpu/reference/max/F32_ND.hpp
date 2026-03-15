#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cfloat>

inline bool matchMaxF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    return inputs.size() == 2 && inputs[0].dtype == DType::FLOAT32 && output.dtype == DType::FLOAT32;
}

inline void runMaxF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    int32_t axis = *static_cast<const int32_t *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);
    const auto &inShape = inViews[0].shape;
    int32_t ndim = static_cast<int32_t>(inShape.size());
    if (axis < 0) axis += ndim;

    uint64_t out_count = countElements(outViews[0].shape);
    for (uint64_t i = 0; i < out_count; ++i) {
        out[getStridedIndex(i, outViews[0].shape, outViews[0].strides)] = -FLT_MAX;
    }

    uint64_t in_count = countElements(inViews[0].shape);
    for(uint64_t i = 0; i < in_count; ++i) {
        uint64_t temp = i;
        uint64_t out_flat = 0;
        uint64_t out_stride = 1;
        for(int32_t d = ndim - 1; d >= 0; --d) {
            uint32_t coord = temp % inShape[d];
            temp /= inShape[d];
            if (d != axis) {
                out_flat += coord * out_stride;
                out_stride *= outViews[0].shape[d];
            }
        }
        float val = in[getStridedIndex(i, inViews[0].shape, inViews[0].strides)];
        uint64_t out_idx = getStridedIndex(out_flat, outViews[0].shape, outViews[0].strides);
        if (val > out[out_idx]) {
            out[out_idx] = val;
        }
    }
}

REGISTER_REF_KERNEL(OpType::MAX, Backend::CPU, matchMaxF32_ND, runMaxF32_ND);