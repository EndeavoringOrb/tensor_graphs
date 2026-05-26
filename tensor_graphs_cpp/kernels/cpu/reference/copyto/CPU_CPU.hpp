#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchCopyTo_CPU_CPU(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &in = inputs[0];

    if (in.dtype != output.dtype)
        return false;

    if (in.getShape() != output.getShape())
        return false;

    if (in.strides != output.strides)
        return false;

    if (!isContiguous(output))
        return false;

    return true;
}

inline void runCopyTo_CPU_CPU(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const uint8_t *src = static_cast<const uint8_t *>(inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(outputs[0]);

    uint64_t numElements = countElements(inViews[0].getShape());
    uint64_t elemSize = getDTypeSize(inViews[0].dtype);

    std::memcpy(dst, src, numElements * elemSize);
}

// Register the kernel for the COPY_TO operation on the CPU backend
REGISTER_REF_KERNEL(
    OpType::COPY_TO,
    1,
    matchCopyTo_CPU_CPU,
    runCopyTo_CPU_CPU,
    {Backend::CPU},
    {DType::ANY},
    {{8, 32}},
    {true},
    {{Backend::CPU}});