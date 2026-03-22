// tensor_graphs_cpp/kernels/cpu/reference/scatter/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchScatterF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    return inputs.size() == 5 &&
           inputs[0].dtype == DType::FLOAT32 &&
           inputs[1].dtype == DType::FLOAT32 &&
           output.dtype == DType::FLOAT32;
}

inline void runScatterF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                             const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *target = static_cast<const float *>(inputs[0]);
    const float *updates = static_cast<const float *>(inputs[1]);
    const int32_t *starts = static_cast<const int32_t *>(inputs[2]);
    const int32_t *steps = static_cast<const int32_t *>(inputs[4]);
    float *out = static_cast<float *>(outputs[0]);

    const auto &out_shape = outViews[0].shape;
    const auto &upd_shape = inViews[1].shape;
    uint64_t n_target = countElements(out_shape);

    if (target != out)
    {
        std::memcpy(out, target, n_target * sizeof(float));
    }

    uint64_t n_updates = countElements(upd_shape);

    std::vector<uint64_t> out_strides(out_shape.size(), 1);
    std::vector<uint64_t> upd_strides(upd_shape.size(), 1);
    for (int d = static_cast<int>(out_shape.size()) - 2; d >= 0; --d)
        out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
    for (int d = static_cast<int>(upd_shape.size()) - 2; d >= 0; --d)
        upd_strides[d] = upd_strides[d + 1] * upd_shape[d + 1];

    for (uint64_t idx = 0; idx < n_updates; ++idx)
    {
        uint64_t temp = idx;
        uint64_t out_idx = 0;
        for (size_t d = 0; d < out_shape.size(); ++d)
        {
            uint64_t coord = temp / upd_strides[d];
            temp %= upd_strides[d];

            int32_t s = (d < inViews[2].shape[0]) ? starts[d] : 0;
            if (s < 0)
                s += out_shape[d];
            int32_t st = (d < inViews[4].shape[0]) ? steps[d] : 1;

            out_idx += (s + coord * st) * out_strides[d];
        }
        out[out_idx] = updates[idx];
    }
}

REGISTER_REF_KERNEL(OpType::SCATTER, {Backend::CPU}, matchScatterF32_ND, runScatterF32_ND);