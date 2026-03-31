// File: tensor_graphs_cpp/kernels/cpu/reference/slice/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchSliceF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    (void)refCounts;
    return inputs.size() == 4 &&
           inputs[0].dtype == output.dtype &&
           inputs[1].dtype == DType::INT32 &&
           inputs[2].dtype == DType::INT32 &&
           inputs[3].dtype == DType::INT32 &&
           isContiguous(inputs[0]) &&
           isContiguous(output);
}

inline void runSliceF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                           const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const uint8_t *in = static_cast<const uint8_t *>(inputs[0]);
    const int32_t *starts = static_cast<const int32_t *>(inputs[1]);
    const int32_t *steps = static_cast<const int32_t *>(inputs[3]);
    uint8_t *out = static_cast<uint8_t *>(outputs[0]);
    const uint64_t elementSize = getDTypeSize(inViews[0].dtype);

    uint64_t n = countElements(outViews[0].getShape());
    const auto &out_shape = outViews[0].getShape();
    const auto &in_shape = inViews[0].getShape();
    int ndim = static_cast<int>(out_shape.size());

    for (uint64_t i = 0; i < n; ++i)
    {
        uint64_t temp = i;
        uint64_t in_phys_idx = 0;
        for (int d = ndim - 1; d >= 0; --d)
        {
            uint32_t coord = temp % out_shape[d];
            temp /= out_shape[d];

            int32_t s = (d < (int)inViews[1].getShape()[0]) ? starts[d] : 0;
            if (s < 0)
                s += in_shape[d];
            int32_t st = (d < (int)inViews[3].getShape()[0]) ? steps[d] : 1;

            uint32_t in_coord = s + coord * st;
            in_phys_idx += (uint64_t)in_coord * inViews[0].strides[d];
        }

        uint64_t out_phys_idx = getStridedIndex(i, out_shape, outViews[0].strides);
        std::memcpy(out + out_phys_idx * elementSize, in + in_phys_idx * elementSize, elementSize);
    }
}

REGISTER_REF_KERNEL(OpType::SLICE, matchSliceF32_ND, runSliceF32_ND, {Backend::CPU});
