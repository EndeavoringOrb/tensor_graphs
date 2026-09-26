// tensor_graphs_cpp/kernels/cpu/reference/scatter/F32_ND.hpp
#pragma once
#include <cstring>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchScatterF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 5)
    {
        return false;
    }

    // Ensure updates and output have the same rank.  Scatter writes into the
    // already allocated output buffer; it has no target tensor input.
    if (inputs[0].getShape().size() != output.getShape().size())
    {
        return false;
    }

    // Ensure the index tensors (starts, ends, steps) have size matching the rank
    uint32_t rank = static_cast<uint32_t>(output.getShape().size());
    if (inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != rank)
        return false;
    if (inputs[2].getShape().size() != 1 || inputs[2].getShape()[0] != rank)
        return false;
    if (inputs[3].getShape().size() != 1 || inputs[3].getShape()[0] != rank)
        return false;
    if (inputs[4].getShape().size() != 1 || inputs[4].getShape()[0] != rank)
        return false;

    return true;
}

inline void runScatterF32_ND(const KernelContext &ctx)
{
    if (ctx.inputs.size() < 5 || ctx.outputs.empty() || ctx.inViews.size() < 5 || ctx.outViews.empty() ||
        ctx.inputs[0] == nullptr || ctx.inputs[1] == nullptr || ctx.inputs[2] == nullptr || ctx.inputs[3] == nullptr ||
        ctx.inputs[4] == nullptr || ctx.outputs[0] == nullptr)
    {
        return;
    }

    const float *updates = static_cast<const float *>(ctx.inputs[0]);
    const int32_t *starts = static_cast<const int32_t *>(ctx.inputs[1]);
    const int32_t *steps = static_cast<const int32_t *>(ctx.inputs[3]);
    const int32_t *scatter_shape = static_cast<const int32_t *>(ctx.inputs[4]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &out_shape = ctx.outViews[0].getShape();
    const auto &upd_shape = ctx.inViews[0].getShape();
    uint64_t n_updates = countElements(upd_shape);
    size_t ndim = upd_shape.size();

    if (ndim == 0 || out_shape.size() != ndim || ctx.outViews[0].strides.size() != ndim ||
        ctx.inViews[0].strides.size() != ndim || ctx.inViews[1].getShape().size() != 1 ||
        ctx.inViews[2].getShape().size() != 1 || ctx.inViews[3].getShape().size() != 1 ||
        ctx.inViews[4].getShape().size() != 1 || ctx.inViews[1].getShape()[0] != ndim ||
        ctx.inViews[2].getShape()[0] != ndim || ctx.inViews[3].getShape()[0] != ndim ||
        ctx.inViews[4].getShape()[0] != ndim)
    {
        return;
    }

    // Reject malformed control tensors before touching the output.  In
    // particular, a zero step or a shape tensor that disagrees with the
    // output view must never be allowed to reach the address calculation.
    for (size_t d = 0; d < ndim; ++d)
    {
        if (out_shape[d] == 0 || upd_shape[d] == 0 || scatter_shape[d] != static_cast<int32_t>(out_shape[d]) ||
            steps[d] == 0)
            return;
    }

    uint64_t out_max_index = 0;
    for (size_t d = 0; d < ndim; ++d)
        out_max_index += (static_cast<uint64_t>(out_shape[d]) - 1) * ctx.outViews[0].strides[d];

    for (uint64_t i = 0; i < n_updates; ++i)
    {
        // 1. Get update value safely
        float val = updates[getStridedIndex(i, upd_shape, ctx.inViews[0].strides)];

        // 2. Unravel flat index 'i' into update coordinates, map to target, and
        // calculate output offset
        uint64_t temp = i;
        uint64_t out_phys_idx = 0;

        // We iterate backwards to unravel the coordinates correctly
        bool valid = true;
        for (size_t d = ndim; d-- > 0;)
        {
            uint32_t coord = temp % upd_shape[d];
            temp /= upd_shape[d];

            int64_t s = starts[d];
            if (s < 0)
                s += static_cast<int64_t>(out_shape[d]);
            int64_t target_coord = s + static_cast<int64_t>(coord) * static_cast<int64_t>(steps[d]);

            if (target_coord < 0 || target_coord >= static_cast<int64_t>(out_shape[d]))
            {
                valid = false;
                break;
            }
            out_phys_idx += static_cast<uint64_t>(target_coord) * ctx.outViews[0].strides[d];
        }

        if (valid && out_phys_idx <= out_max_index)
            out[out_phys_idx] = val;
    }
}

REGISTER_REF_KERNEL(OpType::SCATTER, 5, 5, matchScatterF32_ND, runScatterF32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)},
                    {DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32, DType::INT32},
                    {{8, 32}, {2}, {2}, {2}, {1, 8}}, {false, false, false, false, false},
                    {{MemSpace(1, HandleType::CPP)},
                     {MemSpace(1, HandleType::CPP)},
                     {MemSpace(1, HandleType::CPP)},
                     {MemSpace(1, HandleType::CPP)},
                     {MemSpace(1, HandleType::CPP)}});
