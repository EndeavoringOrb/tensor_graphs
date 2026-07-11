// File: tensor_graphs_cpp/kernels/cpu/general/scatter/inplace_F32_ND_v3.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

inline bool matchScatterF32_ND_Inplace_v3(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

inline void runInplaceScatterF32_ND_v3(const KernelContext &ctx)
{
    const float *target = static_cast<const float *>(ctx.inputs[0]);
    const float *updates = static_cast<const float *>(ctx.inputs[1]);
    const int32_t *starts = static_cast<const int32_t *>(ctx.inputs[2]);
    const int32_t *steps = static_cast<const int32_t *>(ctx.inputs[4]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &out_shape = ctx.outViews[0].getShape();
    const auto &upd_shape = ctx.inViews[1].getShape();
    const auto &out_strides = ctx.outViews[0].strides;

    if (target != out)
    {
        uint64_t n_target = countElements(out_shape);
        for (uint64_t i = 0; i < n_target; ++i)
        {
            out[getStridedIndex(i, out_shape, out_strides)] = target[getStridedIndex(i, out_shape, ctx.inViews[0].strides)];
        }
    }

    uint64_t n_updates = countElements(upd_shape);
    if (n_updates == 0)
        return;

    int ndim = static_cast<int>(upd_shape.size());
    if (ndim == 0)
    {
        int32_t s = ctx.inViews[2].getShape().empty() ? 0 : starts[0];
        if (s < 0)
            s += out_shape.empty() ? 1 : out_shape[0];
        out[s * out_strides[0]] = updates[0];
        return;
    }

    bool inner_contig = true;
    uint32_t inner_dim = upd_shape[ndim - 1];

    int32_t inner_step = (ndim - 1 < (int)ctx.inViews[4].getShape()[0]) ? steps[ndim - 1] : 1;
    if (inner_step != 1)
        inner_contig = false;
    if (out_strides[ndim - 1] != 1)
        inner_contig = false;
    if (ctx.inViews[1].strides[ndim - 1] != 1)
        inner_contig = false;

    if (inner_contig && inner_dim > 1)
    {
        uint64_t outer_iters = n_updates / inner_dim;
        for (uint64_t i = 0; i < outer_iters; ++i)
        {
            uint64_t temp = i;
            uint64_t out_phys_idx = 0;
            uint64_t upd_phys_idx = 0;

            for (int d = ndim - 2; d >= 0; --d)
            {
                uint32_t coord = temp % upd_shape[d];
                temp /= upd_shape[d];

                int32_t s = (d < (int)ctx.inViews[2].getShape()[0]) ? starts[d] : 0;
                if (s < 0)
                    s += out_shape[d];
                int32_t st = (d < (int)ctx.inViews[4].getShape()[0]) ? steps[d] : 1;

                out_phys_idx += (uint64_t)(s + coord * st) * out_strides[d];
                upd_phys_idx += (uint64_t)coord * ctx.inViews[1].strides[d];
            }

            int32_t inner_s = (ndim - 1 < (int)ctx.inViews[2].getShape()[0]) ? starts[ndim - 1] : 0;
            if (inner_s < 0)
                inner_s += out_shape[ndim - 1];
            out_phys_idx += inner_s;

            std::memcpy(out + out_phys_idx, updates + upd_phys_idx, inner_dim * sizeof(float));
        }
    }
    else
    {
        for (uint64_t i = 0; i < n_updates; ++i)
        {
            float val = updates[getStridedIndex(i, upd_shape, ctx.inViews[1].strides)];
            uint64_t temp = i;
            uint64_t out_phys_idx = 0;

            for (int d = ndim - 1; d >= 0; --d)
            {
                uint32_t coord = temp % upd_shape[d];
                temp /= upd_shape[d];

                int32_t s = (d < (int)ctx.inViews[2].getShape()[0]) ? starts[d] : 0;
                if (s < 0)
                    s += out_shape[d];
                int32_t st = (d < (int)ctx.inViews[4].getShape()[0]) ? steps[d] : 1;

                out_phys_idx += (uint64_t)(s + coord * st) * out_strides[d];
            }
            out[out_phys_idx] = val;
        }
    }
}

inline uint32_t refFactoryScatterF32_ND_Inplace_v3(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.scatter(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
}

REGISTER_KERNEL_INPLACE("SCATTER_inplace_v3", 5, matchScatterF32_ND_Inplace_v3, runInplaceScatterF32_ND_v3, refFactoryScatterF32_ND_Inplace_v3, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32}, {{8, 32}, {8, 32}, {2}, {2}, {2}}, {false, false, false, false, false}, {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});