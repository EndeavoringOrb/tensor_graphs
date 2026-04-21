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
    if (inputs.size() != 5)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;
    return true;
}

inline void runInplaceScatterF32_ND_v3(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                       const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *target = static_cast<const float *>(inputs[0]);
    const float *updates = static_cast<const float *>(inputs[1]);
    const int32_t *starts = static_cast<const int32_t *>(inputs[2]);
    const int32_t *steps = static_cast<const int32_t *>(inputs[4]);
    float *out = static_cast<float *>(outputs[0]);

    const auto &out_shape = outViews[0].getShape();
    const auto &upd_shape = inViews[1].getShape();
    const auto &out_strides = outViews[0].strides;

    if (target != out)
    {
        uint64_t n_target = countElements(out_shape);
        for (uint64_t i = 0; i < n_target; ++i)
        {
            out[getStridedIndex(i, out_shape, out_strides)] = target[getStridedIndex(i, out_shape, inViews[0].strides)];
        }
    }

    uint64_t n_updates = countElements(upd_shape);
    if (n_updates == 0)
        return;

    int ndim = static_cast<int>(upd_shape.size());
    if (ndim == 0)
    {
        int32_t s = inViews[2].getShape().empty() ? 0 : starts[0];
        if (s < 0)
            s += out_shape.empty() ? 1 : out_shape[0];
        out[s * out_strides[0]] = updates[0];
        return;
    }

    bool inner_contig = true;
    uint32_t inner_dim = upd_shape[ndim - 1];

    int32_t inner_step = (ndim - 1 < (int)inViews[4].getShape()[0]) ? steps[ndim - 1] : 1;
    if (inner_step != 1)
        inner_contig = false;
    if (out_strides[ndim - 1] != 1)
        inner_contig = false;
    if (inViews[1].strides[ndim - 1] != 1)
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

                int32_t s = (d < (int)inViews[2].getShape()[0]) ? starts[d] : 0;
                if (s < 0)
                    s += out_shape[d];
                int32_t st = (d < (int)inViews[4].getShape()[0]) ? steps[d] : 1;

                out_phys_idx += (uint64_t)(s + coord * st) * out_strides[d];
                upd_phys_idx += (uint64_t)coord * inViews[1].strides[d];
            }

            int32_t inner_s = (ndim - 1 < (int)inViews[2].getShape()[0]) ? starts[ndim - 1] : 0;
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
            float val = updates[getStridedIndex(i, upd_shape, inViews[1].strides)];
            uint64_t temp = i;
            uint64_t out_phys_idx = 0;

            for (int d = ndim - 1; d >= 0; --d)
            {
                uint32_t coord = temp % upd_shape[d];
                temp /= upd_shape[d];

                int32_t s = (d < (int)inViews[2].getShape()[0]) ? starts[d] : 0;
                if (s < 0)
                    s += out_shape[d];
                int32_t st = (d < (int)inViews[4].getShape()[0]) ? steps[d] : 1;

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

REGISTER_KERNEL_INPLACE("SCATTER_inplace_v3", 5, matchScatterF32_ND_Inplace_v3, runInplaceScatterF32_ND_v3, refFactoryScatterF32_ND_Inplace_v3, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32}, {{8, 32}, {8, 32}, {8}, {8}, {8}}, {false, false, false, false, false}, {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});