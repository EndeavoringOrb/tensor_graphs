#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>
#include <numeric>

/**
 * KERNEL: SUM F32 ND
 * Performs reduction along specified axes.
 */

inline bool matchSumF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    // Sum requires: data (0), axis (1)
    if (inputs.size() != 2) return false;
    
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32) return false;
    if (inputs[1].dtype != DType::INT32) return false;

    // Ensure contiguity for the simple reference implementation
    if (!inputs[0].view.isContiguous() || !output.view.isContiguous()) return false;

    return true;
}

inline void runSumF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *data = static_cast<const float *>(inputs[0]);
    const int32_t axis = *static_cast<const int32_t *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    const auto &inShape = inViews[0].shape;
    uint32_t ndim = static_cast<uint32_t>(inShape.size());
    int32_t actualAxis = (axis < 0) ? (axis + ndim) : axis;

    uint64_t totalElements = countElements(inShape);
    uint32_t dimSize = inShape[actualAxis];

    // Calculate strides for the axis
    uint64_t outerStride = 1;
    for (uint32_t i = actualAxis + 1; i < ndim; ++i) outerStride *= inShape[i];
    uint64_t innerStride = 1;
    for (uint32_t i = 0; i < actualAxis; ++i) innerStride *= inShape[i];

    // Naive reduction
    // Initialize output to 0
    uint64_t outElements = countElements(outViews[0].shape);
    for (uint64_t i = 0; i < outElements; ++i) out[i] = 0.0f;

    for (uint64_t o = 0; o < innerStride; ++o) {
        for (uint32_t d = 0; d < dimSize; ++d) {
            for (uint64_t i = 0; i < outerStride; ++i) {
                float val = data[(o * dimSize + d) * outerStride + i];
                out[(o * outerStride) + i] += val;
            }
        }
    }
}

REGISTER_REF_KERNEL(OpType::SUM, Backend::CPU, matchSumF32_ND, runSumF32_ND);