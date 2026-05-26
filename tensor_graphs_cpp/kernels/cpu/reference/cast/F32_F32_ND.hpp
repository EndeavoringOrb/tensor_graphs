// File: tensor_graphs_cpp/kernels/cpu/reference/cast/F32_F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchCastF32_F32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;

    if (!isContiguous(output))
        return false;

    return true;
}

inline void runCastF32_F32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *src = static_cast<const float *>(inputs[0]);
    float *dst = static_cast<float *>(outputs[0]);

    uint64_t numElements = countElements(inViews[0].getShape());

    // F32 to F32 is an identity copy, so a direct memcpy is sufficient.
    std::memcpy(dst, src, numElements * sizeof(float));
}

REGISTER_REF_KERNEL(OpType::CAST, 1, matchCastF32_F32_ND, runCastF32_F32_ND, {Backend::CPU}, {DType::FLOAT32}, {{8, 32}}, {true}, {{Backend::CPU}});