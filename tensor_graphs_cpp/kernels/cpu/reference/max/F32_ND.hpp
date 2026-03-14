// File: tensor_graphs_cpp/kernels/cpu/reference/max/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cfloat>

inline bool matchMaxF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    return inputs.size() == 2 && inputs[0].dtype == DType::FLOAT32 && output.dtype == DType::FLOAT32 && inputs[0].view.isContiguous() && output.view.isContiguous();
}

inline void runMaxF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    int32_t axis = *static_cast<const int32_t *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);
    const auto &inShape = inViews[0].shape;
    int32_t ndim = static_cast<int32_t>(inShape.size());
    if (axis < 0)
        axis += ndim;

    uint64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= inShape[d];
    for (int d = axis + 1; d < ndim; ++d)
        inner *= inShape[d];
    uint32_t dim_size = inShape[axis];

    uint64_t out_count = countElements(outViews[0].shape);
    for (uint64_t i = 0; i < out_count; ++i)
        out[i] = -FLT_MAX;

    for (uint64_t o = 0; o < outer; ++o)
    {
        for (uint32_t d = 0; d < dim_size; ++d)
        {
            for (uint64_t i = 0; i < inner; ++i)
            {
                float val = in[(o * dim_size + d) * inner + i];
                uint64_t dst_idx = o * inner + i;
                if (val > out[dst_idx])
                    out[dst_idx] = val;
            }
        }
    }
}

REGISTER_REF_KERNEL(OpType::MAX, Backend::CPU, matchMaxF32_ND, runMaxF32_ND);