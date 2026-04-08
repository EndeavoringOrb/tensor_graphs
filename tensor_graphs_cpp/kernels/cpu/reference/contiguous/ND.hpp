#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

/**
 * KERNEL: CONTIGUOUS (Generic ND)
 * Purpose: Ensures the tensor is contiguous in memory.
 * Operation: Copies data from a potentially strided source to a contiguous destination.
 */

inline bool matchContiguous_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 1)
        return false;

    const auto &in = inputs[0];

    // Check Dtypes and Shapes
    if (in.dtype != output.dtype)
        return false;
    if (in.getShape() != output.getShape())
        return false;

    // Input must not be contiguous (otherwise this kernel is redundant, though technically valid)
    // Output must be contiguous
    if (!isContiguous(output))
        return false;

    return true;
}

inline void runContiguous_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                             const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const uint8_t *src_base = static_cast<const uint8_t *>(inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(outputs[0]);

    const auto &view = inViews[0];
    const auto &shape = view.getShape();
    const auto &strides = view.strides;
    uint64_t elementSize = getDTypeSize(view.dtype);

    if (shape.empty())
    {
        // Scalar or empty
        if (countElements(shape) == 1)
        {
            std::memcpy(dst, src_base, elementSize);
        }
        return;
    }

    // Generic ND iteration
    std::vector<uint32_t> coords(shape.size(), 0);
    uint64_t totalElements = countElements(shape);

    for (uint64_t i = 0; i < totalElements; ++i)
    {
        uint64_t offset = 0;
        for (size_t d = 0; d < coords.size(); ++d)
        {
            offset += static_cast<uint64_t>(coords[d]) * strides[d];
        }

        // Copy single element
        std::memcpy(dst, src_base + offset * elementSize, elementSize);

        // Advance pointer in contiguous destination
        dst += elementSize;

        // Increment coordinates (odometer style)
        for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d)
        {
            coords[d]++;
            if (coords[d] < shape[d])
                break;
            coords[d] = 0;
        }
    }
}

REGISTER_REF_KERNEL(OpType::CONTIGUOUS, 1, matchContiguous_ND, runContiguous_ND, {Backend::CPU}, {DType::FLOAT32}, {{8, 32}}, {false}, {{Backend::CPU}});