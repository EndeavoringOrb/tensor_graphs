#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

/**
 * KERNEL: COPY_TO (CPU -> CPU)
 * ---------------------------------------------------------
 * This kernel handles moving data between two CPU buffers.
 * It supports cases where either the source or destination
 * (or both) may have non-contiguous strides.
 */

/**
 * Match Function:
 * Validates that both input and output are on the CPU,
 * DTypes match, and Shapes match.
 */
inline bool matchCopyTo_CPU_CPU(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 1)
        return false;

    if (inputs[0].backend != Backend::CPU || output.backend != Backend::CPU)
        return false;

    if (inputs[0].dtype != output.dtype)
        return false;

    if (inputs[0].getShape() != output.getShape())
        return false;

    return true;
}

/**
 * Run Function:
 * Performs a standard memory copy. Uses std::memcpy for the fast path
 * (both contiguous) and element-wise strided copying for the slow path.
 */
inline void runCopyTo_CPU_CPU(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const uint8_t *src = static_cast<const uint8_t *>(inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(outputs[0]);

    uint64_t numElements = countElements(inViews[0].getShape());
    uint64_t elemSize = getDTypeSize(inViews[0].dtype);

    bool srcContig = isContiguous(inViews[0]);
    bool dstContig = isContiguous(outViews[0]);

    // Fast Path: Both are contiguous
    if (srcContig && dstContig)
    {
        std::memcpy(dst, src, numElements * elemSize);
    }
    // Slow Path: Handle arbitrary striding
    else
    {
        for (uint64_t i = 0; i < numElements; ++i)
        {
            uint64_t srcIdx = getStridedIndex(i, inViews[0].getShape(), inViews[0].strides);
            uint64_t dstIdx = getStridedIndex(i, outViews[0].getShape(), outViews[0].strides);

            std::memcpy(dst + (dstIdx * elemSize),
                        src + (srcIdx * elemSize),
                        elemSize);
        }
    }
}

// Register the kernel for the COPY_TO operation on the CPU backend
REGISTER_REF_KERNEL(
    OpType::COPY_TO,
    1,
    matchCopyTo_CPU_CPU,
    runCopyTo_CPU_CPU,
    {Backend::CPU},   // Output backend
    {DType::FLOAT32}, // Supported DTypes (generic)
    {{8, 32}},        // Dummy shape for registration
    {false},          // Input does not strictly require contiguity
    {{Backend::CPU}}  // Input backends
);