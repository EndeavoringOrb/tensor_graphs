// tensor_graphs_cpp/kernels/cpu/reference/scatter/F32_ND.hpp
#pragma once
#include <cstring>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchScatterF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
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

    return true;
}

inline void runScatterF32_ND(const KernelContext &ctx)
{
    const float *updates = static_cast<const float *>(ctx.inputs[0]);
    const int32_t *starts = static_cast<const int32_t *>(ctx.inputs[1]);
    const int32_t *steps = static_cast<const int32_t *>(ctx.inputs[3]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &out_shape = ctx.outViews[0].getShape();
    const auto &upd_shape = ctx.inViews[0].getShape();
    uint64_t n_updates = countElements(upd_shape);
    int ndim = static_cast<int>(upd_shape.size());

        for (uint64_t i = 0; i < n_updates; ++i)
    {
        // 1. Get update value safely
        float val = updates[getStridedIndex(i, upd_shape, ctx.inViews[0].strides)];

        // 2. Unravel flat index 'i' into update coordinates, map to target, and
        // calculate output offset
        uint64_t temp = i;
        uint64_t out_phys_idx = 0;

        // We iterate backwards to unravel the coordinates correctly
        for (int d = ndim - 1; d >= 0; --d)
        {
            uint32_t coord = temp % upd_shape[d];
            temp /= upd_shape[d];

            int32_t s = (d < (int)ctx.inViews[1].getShape()[0]) ? starts[d] : 0;
            if (s < 0)
                s += out_shape[d];
            int32_t st = (d < (int)ctx.inViews[3].getShape()[0]) ? steps[d] : 1;

            uint32_t target_coord = s + coord * st;
            out_phys_idx += (uint64_t)target_coord * ctx.outViews[0].strides[d];
        }
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
