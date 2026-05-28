// File: tensor_graphs_cpp/kernels/cpu/reference/triu/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchTriuF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return output.dtype == DType::FLOAT32 && isContiguous(output);
}

inline void runTriuF32_ND(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    int32_t k = *static_cast<const int32_t *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    const auto &shape = ctx.outViews[0].getShape();
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

REGISTER_REF_KERNEL(OpType::TRIU, 2, matchTriuF32_ND, runTriuF32_ND, {Backend::CPU}, {DType::FLOAT32, DType::INT32}, {{8, 32}, {1}}, {true, false}, {{Backend::CPU}, {Backend::CPU}});
