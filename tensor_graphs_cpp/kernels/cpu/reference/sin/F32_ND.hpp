#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>

inline bool matchSinF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 1)
        return false;
    return inputs[0].dtype == DType::FLOAT32 && output.dtype == DType::FLOAT32 && isContiguous(inputs[0]) && isContiguous(output);
}

inline void runSinF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    float *out = static_cast<float *>(outputs[0]);
    uint64_t n = countElements(outViews[0].getShape());
    for (uint64_t i = 0; i < n; ++i)
        out[i] = std::sin(in[i]);
}

REGISTER_REF_KERNEL(OpType::SIN, 1, matchSinF32_ND, runSinF32_ND, {Backend::CPU}, {DType::FLOAT32}, {{8, 32}}, {false});