#pragma once

#include <cstring>
#include <vector>

#include "core/kernels.hpp"

inline LogicalId refFactoryScatterF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.scatter(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
}

inline bool matchScatterF32Rows(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const uint64_t rank = output.getShape().size();
    if (rank == 0 || inputs[0].getShape().size() != rank || output.dtype != DType::FLOAT32)
        return false;
    for (uint32_t i = 1; i < 5; ++i)
        if (inputs[i].getShape().size() != 1 || inputs[i].getShape()[0] != rank)
            return false;
    return isContiguous(output);
}

// Copies update rows [row_begin, row_end) into the output.
inline void scatterF32RowsRange(const KernelContext &ctx, uint64_t row_begin, uint64_t row_end)
{
    const float *updates = static_cast<const float *>(ctx.inputs[0]);
    const int32_t *starts = static_cast<const int32_t *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    const auto &upd_shape = ctx.inViews[0].getShape();
    const auto &out_strides = ctx.outViews[0].strides;
    const size_t rank = upd_shape.size();
    const uint64_t cols = upd_shape[rank - 1];
    for (uint64_t row = row_begin; row < row_end; ++row)
    {
        uint64_t rest = row;
        uint64_t offset = static_cast<uint64_t>(starts[rank - 1]);
        for (size_t d = rank - 1; d-- > 0;)
        {
            const uint64_t coord = rest % upd_shape[d];
            rest /= upd_shape[d];
            offset += (static_cast<uint64_t>(starts[d]) + coord) * out_strides[d];
        }
        std::memcpy(out + offset, updates + row * cols, cols * sizeof(float));
    }
}

inline uint64_t validateScatterF32Rows(const KernelContext &ctx)
{
    const int32_t *starts = static_cast<const int32_t *>(ctx.inputs[1]);
    const int32_t *steps = static_cast<const int32_t *>(ctx.inputs[3]);
    const int32_t *scatter_shape = static_cast<const int32_t *>(ctx.inputs[4]);
    const auto &upd_shape = ctx.inViews[0].getShape();
    const auto &out_shape = ctx.outViews[0].getShape();
    if (upd_shape.size() != out_shape.size() || upd_shape.empty())
        Error::throw_err("[Scatter_F32_Rows] rank mismatch");
    uint64_t rows = 1;
    for (size_t d = 0; d < out_shape.size(); ++d)
    {
        if (steps[d] != 1)
            Error::throw_err("[Scatter_F32_Rows] only unit steps are supported");
        if (starts[d] < 0 || static_cast<uint64_t>(starts[d]) + upd_shape[d] > out_shape[d] ||
            scatter_shape[d] != static_cast<int32_t>(out_shape[d]))
            Error::throw_err("[Scatter_F32_Rows] update box outside the output");
        if (d + 1 < out_shape.size())
            rows *= upd_shape[d];
    }
    return rows;
}
