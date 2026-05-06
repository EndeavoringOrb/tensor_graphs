// File: tensor_graphs_cpp/kernels/cpu/reference/concat/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "core/shapes.hpp"
#include <cstring>

inline bool matchConcatF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Axis tensor is the last input. We need at least one data tensor + axis.
    if (inputs.size() < 2 || output.dtype != DType::FLOAT32)
        return false;
    return true;
}

inline void runConcatF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                            const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    float *out = static_cast<float *>(outputs[0]);
    const std::vector<uint32_t> &outShape = outViews[0].getShape();
    const std::vector<uint64_t> &outStrides = outViews[0].strides;
    uint32_t rank = static_cast<uint32_t>(outShape.size());

    // The axis is stored in the last input tensor
    int32_t axis = *static_cast<const int32_t *>(inputs.back());
    if (axis < 0)
        axis += static_cast<int32_t>(rank);

    // Calculate the starting offset along the 'axis' for each input tensor
    std::vector<uint32_t> axis_offsets(inputs.size(), 0);
    for (size_t n = 0; n < inputs.size() - 1; ++n)
    {
        axis_offsets[n + 1] = axis_offsets[n] + inViews[n].getShape()[axis];
    }

    uint64_t totalElements = countElements(outViews[0]);

    for (uint64_t i = 0; i < totalElements; ++i)
    {
        // 1. Convert flat output index to coordinates
        std::vector<uint32_t> coords = coordsFromFlatIndex(i, outShape);
        uint32_t axis_coord = coords[axis];

        // 2. Find which source tensor 'n' this element belongs to
        size_t n = 0;
        // Search for n such that axis_offsets[n] <= axis_coord < axis_offsets[n+1]
        // data tensors are in indices 0 to inputs.size() - 2
        while (n < inputs.size() - 2 && axis_coord >= axis_offsets[n + 1])
        {
            n++;
        }

        // 3. Map global coordinates to local coordinates of input 'n'
        std::vector<uint32_t> local_coords = coords;
        local_coords[axis] = axis_coord - axis_offsets[n];

        // 4. Calculate flat index within the local input tensor
        uint64_t local_flat_idx = flatIndexFromCoords(local_coords, inViews[n].getShape());

        // 5. Use strides to find actual physical memory locations
        uint64_t out_phys_idx = getStridedIndex(i, outShape, outStrides);
        uint64_t in_phys_idx = getStridedIndex(local_flat_idx, inViews[n].getShape(), inViews[n].strides);

        const float *in_ptr = static_cast<const float *>(inputs[n]);
        out[out_phys_idx] = in_ptr[in_phys_idx];
    }
}

REGISTER_REF_KERNEL(OpType::CONCAT, 2, matchConcatF32_ND, runConcatF32_ND, {Backend::CPU}, {DType::FLOAT32, DType::INT32}, {{8, 32}, {8, 32}}, {false, false}, {{Backend::CPU}, {Backend::CPU}});