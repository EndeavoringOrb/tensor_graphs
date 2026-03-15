#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchReshapeND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2) return false;
    return countElements(inputs[0].shape) == countElements(output.shape);
}

inline void runReshapeND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const uint8_t *src = static_cast<const uint8_t *>(inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(outputs[0]);

    uint64_t numElements = countElements(inViews[0].shape);
    uint64_t elementSize = getDTypeSize(inViews[0].dtype);

    for (uint64_t i = 0; i < numElements; ++i)
    {
        uint64_t src_idx = getStridedIndex(i, inViews[0].shape, inViews[0].strides);
        uint64_t dst_idx = getStridedIndex(i, outViews[0].shape, outViews[0].strides);
        std::memcpy(dst + dst_idx * elementSize, src + src_idx * elementSize, elementSize);
    }
}

REGISTER_REF_KERNEL(OpType::RESHAPE, Backend::CPU, matchReshapeND, runReshapeND);