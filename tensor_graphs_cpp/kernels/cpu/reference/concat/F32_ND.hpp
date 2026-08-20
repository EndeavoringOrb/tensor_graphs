#pragma once
#include <cstring>
#include <vector>

#include "core/kernels.hpp"
#include "core/shape_propagator.hpp"
#include "core/types.hpp"

inline bool matchConcatF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Axis tensor is the last input. We need at least one data tensor + axis.
    return isContiguous(output);
}

inline void runConcatF32_ND(const KernelContext &ctx)
{
    float *out = static_cast<float *>(ctx.outputs[0]);
    const std::vector<uint32_t> &outShape = ctx.outViews[0].getShape();
    uint32_t rank = static_cast<uint32_t>(outShape.size());

    // The axis is stored in the first input tensor
    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[0]);
    if (axis < 0)
        axis += static_cast<int32_t>(rank);

    // Calculate outer_dim (product of dimensions before axis)
    uint64_t outer_dim = 1;
    for (int32_t i = 0; i < axis; ++i)
    {
        outer_dim *= outShape[i];
    }

    // Calculate inner_dim (product of dimensions after axis)
    uint64_t inner_dim = 1;
    for (uint32_t i = static_cast<uint32_t>(axis) + 1; i < rank; ++i)
    {
        inner_dim *= outShape[i];
    }

    size_t num_data_tensors = ctx.inputs.size() - 1; // input 0 is the axis tensor

    // Precompute per-input slice metadata outside the execution loop
    struct InputSlice
    {
        const float *ptr;
        size_t copy_bytes;
        size_t stride_elements; // elements per outer loop step
    };

    std::vector<InputSlice> slices(num_data_tensors);
    size_t out_stride_elements = 0;

    for (size_t k = 0; k < num_data_tensors; ++k)
    {
        uint32_t in_axis_dim = ctx.inViews[k + 1].getShape()[axis];
        size_t slice_elements = static_cast<size_t>(in_axis_dim) * inner_dim;

        slices[k].ptr = static_cast<const float *>(ctx.inputs[k + 1]);
        slices[k].copy_bytes = slice_elements * sizeof(float);
        slices[k].stride_elements = slice_elements;

        out_stride_elements += slice_elements;
    }

    // Block copy slices for each outer dimension step
    for (uint64_t o = 0; o < outer_dim; ++o)
    {
        float *out_ptr = out + o * out_stride_elements;
        for (size_t k = 0; k < num_data_tensors; ++k)
        {
            const float *in_ptr = slices[k].ptr + o * slices[k].stride_elements;
            std::memcpy(out_ptr, in_ptr, slices[k].copy_bytes);
            out_ptr += slices[k].stride_elements;
        }
    }
}

REGISTER_REF_KERNEL(OpType::CONCAT, 2, UINT32_MAX, matchConcatF32_ND, runConcatF32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::INT32, DType::FLOAT32}, {{1}, {8, 32}}, {false, true},
                    {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});