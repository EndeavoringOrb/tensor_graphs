// File: tensor_graphs_cpp/kernels/cpu/reference/concat/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchConcatF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() < 2 || output.dtype != DType::FLOAT32) return false;
    for (size_t i = 0; i < inputs.size() - 1; ++i) {
        if (!inputs[i].view.isContiguous()) return false;
    }
    return output.view.isContiguous();
}

inline void runConcatF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                            const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    float *out = static_cast<float *>(outputs[0]);
    int32_t axis = *static_cast<const int32_t *>(inputs.back());
    if (axis < 0)
        axis += static_cast<int32_t>(outViews[0].shape.size());

    uint64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= outViews[0].shape[d];
    for (size_t d = axis + 1; d < outViews[0].shape.size(); ++d)
        inner *= outViews[0].shape[d];

    uint64_t out_offset = 0;
    for (uint64_t outer_idx = 0; outer_idx < outer; ++outer_idx)
    {
        for (size_t n = 0; n < inputs.size() - 1; ++n)
        {
            const float *in = static_cast<const float *>(inputs[n]);
            uint64_t chunk = static_cast<uint64_t>(inViews[n].shape[axis]) * inner;
            std::memcpy(out + out_offset, in + outer_idx * chunk, chunk * sizeof(float));
            out_offset += chunk;
        }
    }
}

REGISTER_REF_KERNEL(OpType::CONCAT, Backend::CPU, matchConcatF32_ND, runConcatF32_ND);