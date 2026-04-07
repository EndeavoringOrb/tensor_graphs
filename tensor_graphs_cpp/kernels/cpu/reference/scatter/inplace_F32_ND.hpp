// tensor_graphs_cpp/kernels/cpu/reference/scatter/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchScatterF32_ND_Inplace(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 5) return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32) return false;
    if (inputs[0].storageType == StorageType::PERSISTENT) return false;
    auto it = refCounts.find(inputs[0].id);
    if (it == refCounts.end() || it->second != 1) return false;
    return true;
}

inline void runInplaceScatterF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                             const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *target = static_cast<const float *>(inputs[0]);
    const float *updates = static_cast<const float *>(inputs[1]);
    const int32_t *starts = static_cast<const int32_t *>(inputs[2]);
    const int32_t *steps = static_cast<const int32_t *>(inputs[4]);
    float *out = static_cast<float *>(outputs[0]);

    const auto &out_shape = outViews[0].getShape();
    const auto &upd_shape = inViews[1].getShape();
    uint64_t n_target = countElements(out_shape);

    // If target and out are different buffers, copy target to out first.
    // Use getStridedIndex to handle potentially strided target/out.
    if (target != out)
    {
        for (uint64_t i = 0; i < n_target; ++i)
        {
            out[getStridedIndex(i, out_shape, outViews[0].strides)] =
                target[getStridedIndex(i, out_shape, inViews[0].strides)];
        }
    }

    uint64_t n_updates = countElements(upd_shape);
    int ndim = static_cast<int>(upd_shape.size());

    for (uint64_t i = 0; i < n_updates; ++i)
    {
        // 1. Get update value safely
        float val = updates[getStridedIndex(i, upd_shape, inViews[1].strides)];

        // 2. Unravel flat index 'i' into update coordinates, map to target, and calculate output offset
        uint64_t temp = i;
        uint64_t out_phys_idx = 0;

        // We iterate backwards to unravel the coordinates correctly
        for (int d = ndim - 1; d >= 0; --d)
        {
            uint32_t coord = temp % upd_shape[d];
            temp /= upd_shape[d];

            int32_t s = (d < (int)inViews[2].getShape()[0]) ? starts[d] : 0;
            if (s < 0)
                s += out_shape[d];
            int32_t st = (d < (int)inViews[4].getShape()[0]) ? steps[d] : 1;

            uint32_t target_coord = s + coord * st;
            out_phys_idx += (uint64_t)target_coord * outViews[0].strides[d];
        }
        out[out_phys_idx] = val;
    }
}

REGISTER_REF_KERNEL_INPLACE(OpType::SCATTER, 5, matchScatterF32_ND_Inplace, runInplaceScatterF32_ND, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32}, {{8, 32}, {8, 32}, {8}, {8}, {8}}, {false, false, false, false, false});