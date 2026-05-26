#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchAddI32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    return true;
}

inline void runAddI32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const int32_t *a = static_cast<const int32_t *>(inputs[0]);
    const int32_t *b = static_cast<const int32_t *>(inputs[1]);
    int32_t *out = static_cast<int32_t *>(outputs[0]);
    uint64_t numElements = countElements(inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[getStridedIndex(i, outViews[0].getShape(), outViews[0].strides)] =
            a[getStridedIndex(i, inViews[0].getShape(), inViews[0].strides)] +
            b[getStridedIndex(i, inViews[1].getShape(), inViews[1].strides)];
    }
}

REGISTER_REF_KERNEL(
    OpType::ADD,
    2,
    matchAddI32_ND,
    runAddI32_ND,
    {Backend::CPU},
    {DType::INT32, DType::INT32},
    {{8, 32}, {8, 32}},
    {false, false},
    {{Backend::CPU}, {Backend::CPU}});