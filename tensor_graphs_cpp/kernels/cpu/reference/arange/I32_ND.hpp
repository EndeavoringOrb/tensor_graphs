// File: tensor_graphs_cpp/kernels/cpu/reference/arange/I32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchArangeI32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return isContiguous(output);
}

inline void runArangeI32_ND(const KernelContext &ctx)
{
    int32_t start = *static_cast<const int32_t *>(ctx.inputs[0]);
    int32_t step = *static_cast<const int32_t *>(ctx.inputs[2]);
    int32_t *out = static_cast<int32_t *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.outViews[0].getShape());
    for (uint64_t i = 0; i < n; ++i)
        out[i] = start + static_cast<int32_t>(i) * step;
}

REGISTER_REF_KERNEL(OpType::ARANGE, 3, 3, matchArangeI32_ND, runArangeI32_ND, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::INT32, DType::INT32, DType::INT32}, {{1}, {1}, {1}}, {false, false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
