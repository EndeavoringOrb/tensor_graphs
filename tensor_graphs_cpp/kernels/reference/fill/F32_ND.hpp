#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchFillF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs.size() == 2 && inputs[0].dtype == DType::FLOAT32 && output.dtype == DType::FLOAT32;
}

inline void runFillF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                          const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    float val = *static_cast<const float *>(inputs[0]);
    float *out = static_cast<float *>(outputs[0]);
    uint64_t n = countElements(outViews[0].shape);
    for (uint64_t i = 0; i < n; ++i)
        out[i] = val;
}

REGISTER_KERNEL(OpType::FILL, Backend::CPU, matchFillF32_ND, runFillF32_ND);