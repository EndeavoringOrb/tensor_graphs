#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchRepeatF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 3) return false;
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32) return false;
    if (inputs[1].dtype != DType::INT32 || inputs[2].dtype != DType::INT32) return false;
    return true; // Contiguity check removed
}

inline void runRepeatF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                            const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *src = static_cast<const float *>(inputs[0]);
    int32_t repeats = *static_cast<const int32_t *>(inputs[1]);
    int32_t axis = *static_cast<const int32_t *>(inputs[2]);
    float *dst = static_cast<float *>(outputs[0]);

    int32_t ndim = static_cast<int32_t>(inViews[0].shape.size());
    if (axis < 0) axis += ndim;

    const auto &outShape = outViews[0].shape;
    uint64_t numElements = countElements(outShape);

    for (uint64_t i = 0; i < numElements; ++i)
    {
        uint64_t temp = i;
        uint64_t src_flat = 0;
        uint64_t stride = 1;

        for (int32_t d = ndim - 1; d >= 0; --d)
        {
            uint32_t coord = temp % outShape[d];
            temp /= outShape[d];

            uint32_t src_coord = (d == axis) ? (coord / repeats) : coord;
            src_flat += src_coord * stride;
            stride *= inViews[0].shape[d];
        }

        dst[getStridedIndex(i, outShape, outViews[0].strides)] = 
            src[getStridedIndex(src_flat, inViews[0].shape, inViews[0].strides)];
    }
}

REGISTER_REF_KERNEL(OpType::REPEAT, Backend::CPU, matchRepeatF32_ND, runRepeatF32_ND);