#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchSliceF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs.size() == 4 && inputs[0].dtype == DType::FLOAT32 && output.dtype == DType::FLOAT32;
}

inline void runSliceF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                           const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    const int32_t *starts = static_cast<const int32_t *>(inputs[1]);
    const int32_t *steps = static_cast<const int32_t *>(inputs[3]);
    float *out = static_cast<float *>(outputs[0]);

    uint64_t n = countElements(outViews[0].shape);
    const auto &out_shape = outViews[0].shape;
    const auto &in_shape = inViews[0].shape;

    std::vector<uint64_t> out_strides(out_shape.size(), 1);
    std::vector<uint64_t> in_strides(in_shape.size(), 1);
    for (int d = static_cast<int>(out_shape.size()) - 2; d >= 0; --d)
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    for (int d = static_cast<int>(in_shape.size()) - 2; d >= 0; --d)
        in_strides[d] = in_strides[d + 1] * in_shape[d + 1];

    for (uint64_t idx = 0; idx < n; ++idx)
    {
        uint64_t temp = idx;
        uint64_t in_idx = 0;
        for (size_t d = 0; d < out_shape.size(); ++d)
        {
            uint64_t coord = temp / out_strides[d];
            temp %= out_strides[d];

            int32_t s = (d < inViews[1].shape[0]) ? starts[d] : 0;
            if (s < 0)
                s += in_shape[d];
            int32_t st = (d < inViews[3].shape[0]) ? steps[d] : 1;

            in_idx += (s + coord * st) * in_strides[d];
        }
        out[idx] = in[in_idx];
    }
}

REGISTER_REF_KERNEL(OpType::SLICE, Backend::CPU, matchSliceF32_ND, runSliceF32_ND);