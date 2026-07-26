// File: tensor_graphs_cpp/kernels/cpu/reference/concat/F32_ND.hpp
#pragma once
#include <cstring>

#include "core/kernels.hpp"
#include "core/shapes.hpp"
#include "core/types.hpp"

inline bool matchConcatF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Axis tensor is the last input. We need at least one data tensor + axis.
    return true;
}

inline void runConcatF32_ND(const KernelContext &ctx)
{
    float *out = static_cast<float *>(ctx.outputs[0]);
    const std::vector<uint32_t> &outShape = ctx.outViews[0].getShape();
    const std::vector<uint64_t> &outStrides = ctx.outViews[0].strides;
    uint32_t rank = static_cast<uint32_t>(outShape.size());

    // The axis is stored in the last input tensor
    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[0]);
    if (axis < 0)
        axis += static_cast<int32_t>(rank);

    // Calculate the starting offset along the 'axis' for each input tensor
    std::vector<uint32_t> axis_offsets(ctx.inputs.size(), 0);
    for (uint64_t n = 1; n < ctx.inputs.size() - 1; ++n)
    {
        axis_offsets[n + 1] = axis_offsets[n] + ctx.inViews[n].getShape()[axis];
    }

    uint64_t totalElements = countElements(ctx.outViews[0]);

    for (uint64_t i = 0; i < totalElements; ++i)
    {
        // 1. Convert flat output index to coordinates
        std::vector<uint32_t> coords = coordsFromFlatIndex(i, outShape);
        uint32_t axis_coord = coords[axis];

        // 2. Find which source tensor 'n' this element belongs to
        uint64_t n = 0;
        // Search for n such that axis_offsets[n] <= axis_coord < axis_offsets[n+1]
        // data tensors are in indices 0 to inputs.size() - 2
        while (n < ctx.inputs.size() - 2 && axis_coord >= axis_offsets[n + 2])
        {
            n++;
        }

        // 3. Map global coordinates to local coordinates of input 'n'
        std::vector<uint32_t> local_coords = coords;
        local_coords[axis] = axis_coord - axis_offsets[n + 1];

        // 4. Calculate flat index within the local input tensor
        uint64_t local_flat_idx = flatIndexFromCoords(local_coords, ctx.inViews[n + 1].getShape());

        // 5. Use strides to find actual physical memory locations
        uint64_t out_phys_idx = getStridedIndex(i, outShape, outStrides);
        uint64_t in_phys_idx =
            getStridedIndex(local_flat_idx, ctx.inViews[n + 1].getShape(), ctx.inViews[n + 1].strides);

        const float *in_ptr = static_cast<const float *>(ctx.inputs[n + 1]);
        out[out_phys_idx] = in_ptr[in_phys_idx];
    }
}

REGISTER_REF_KERNEL(OpType::CONCAT, 2, UINT32_MAX, matchConcatF32_ND, runConcatF32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::INT32, DType::FLOAT32}, {{1}, {8, 32}}, {false, false},
                    {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});