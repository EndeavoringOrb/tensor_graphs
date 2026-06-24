// tensor_graphs_cpp/kernels/cpu/reference/scatter/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchScatterF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Ensure target (inputs[0]), updates (inputs[1]), and output have the same rank
    if (inputs[0].getShape().size() != inputs[1].getShape().size() ||
        inputs[0].getShape().size() != output.getShape().size())
    {
        return false;
    }

    // Ensure the index tensors (starts, ends, steps) have size matching the rank
    uint32_t rank = static_cast<uint32_t>(inputs[0].getShape().size());
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
    const float *target = static_cast<const float *>(ctx.inputs[0]);
    const float *updates = static_cast<const float *>(ctx.inputs[1]);
    const int32_t *starts = static_cast<const int32_t *>(ctx.inputs[2]);
    const int32_t *steps = static_cast<const int32_t *>(ctx.inputs[4]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &out_shape = ctx.outViews[0].getShape();
    const auto &upd_shape = ctx.inViews[1].getShape();
    uint64_t n_target = countElements(out_shape);

    // If target and out are different buffers, copy target to out first.
    // Use getStridedIndex to handle potentially strided target/out.
    if (target != out)
    {
        for (uint64_t i = 0; i < n_target; ++i)
        {
            out[getStridedIndex(i, out_shape, ctx.outViews[0].strides)] =
                target[getStridedIndex(i, out_shape, ctx.inViews[0].strides)];
        }
    }

    uint64_t n_updates = countElements(upd_shape);
    int ndim = static_cast<int>(upd_shape.size());

    for (uint64_t i = 0; i < n_updates; ++i)
    {
        // 1. Get update value safely
        float val = updates[getStridedIndex(i, upd_shape, ctx.inViews[1].strides)];

        // 2. Unravel flat index 'i' into update coordinates, map to target, and calculate output offset
        uint64_t temp = i;
        uint64_t out_phys_idx = 0;

        // We iterate backwards to unravel the coordinates correctly
        for (int d = ndim - 1; d >= 0; --d)
        {
            uint32_t coord = temp % upd_shape[d];
            temp /= upd_shape[d];

            int32_t s = (d < (int)ctx.inViews[2].getShape()[0]) ? starts[d] : 0;
            if (s < 0)
                s += out_shape[d];
            int32_t st = (d < (int)ctx.inViews[4].getShape()[0]) ? steps[d] : 1;

            uint32_t target_coord = s + coord * st;
            out_phys_idx += (uint64_t)target_coord * ctx.outViews[0].strides[d];
        }
        out[out_phys_idx] = val;
    }
}

REGISTER_REF_KERNEL(OpType::SCATTER, 5, matchScatterF32_ND, runScatterF32_ND, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32}, {{8, 32}, {8, 32}, {8}, {8}, {8}}, {false, false, false, false, false}, {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});
