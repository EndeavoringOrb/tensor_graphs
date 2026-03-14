// File: tensor_graphs_cpp/kernels/cpu/reference/arange/I32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchArangeI32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    return inputs.size() == 3 && output.dtype == DType::INT32 && output.view.isContiguous();
}

inline void runArangeI32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                            const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    int32_t start = *static_cast<const int32_t *>(inputs[0]);
    int32_t step = *static_cast<const int32_t *>(inputs[2]);
    int32_t *out = static_cast<int32_t *>(outputs[0]);
    uint64_t n = countElements(outViews[0].shape);
    for (uint64_t i = 0; i < n; ++i)
        out[i] = start + static_cast<int32_t>(i) * step;
}

REGISTER_REF_KERNEL(OpType::ARANGE, Backend::CPU, matchArangeI32_ND, runArangeI32_ND);