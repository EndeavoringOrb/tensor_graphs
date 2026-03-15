#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchMulF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2) return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32) return false;
    if (inputs[0].shape != inputs[1].shape || inputs[0].shape != output.shape) return false;
    return true;
}

inline void runMulF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *a = static_cast<const float *>(inputs[0]);
    const float *b = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);
    uint64_t numElements = countElements(inViews[0].shape);

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[getStridedIndex(i, outViews[0].shape, outViews[0].strides)] = 
            a[getStridedIndex(i, inViews[0].shape, inViews[0].strides)] * 
            b[getStridedIndex(i, inViews[1].shape, inViews[1].strides)];
    }
}

REGISTER_REF_KERNEL(OpType::MUL, Backend::CPU, matchMulF32_ND, runMulF32_ND);