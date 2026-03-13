// File: tensor_graphs_cpp/kernels/cpu/reference/triu/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchTriuF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs.size() == 2 && inputs[0].dtype == DType::FLOAT32 && output.dtype == DType::FLOAT32 && inputs[0].view.isContiguous() && output.view.isContiguous();
}

inline void runTriuF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                          const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    int32_t k = *static_cast<const int32_t *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);
    const auto &shape = outViews[0].shape;
    uint32_t cols = shape.back();
    uint32_t rows = shape[shape.size() - 2];
    uint64_t batch = countElements(shape) / (rows * cols);

    for (uint64_t b = 0; b < batch; ++b)
    {
        for (uint32_t r = 0; r < rows; ++r)
        {
            for (uint32_t c = 0; c < cols; ++c)
            {
                uint64_t idx = b * rows * cols + r * cols + c;
                out[idx] = (static_cast<int32_t>(c) >= static_cast<int32_t>(r) + k) ? in[idx] : 0.0f;
            }
        }
    }
}

REGISTER_REF_KERNEL(OpType::TRIU, Backend::CPU, matchTriuF32_ND, runTriuF32_ND);