#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchFillF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return isContiguous(output);
}

inline void runFillF32_ND(const KernelContext &ctx)
{
    float val = *static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.outViews[0].getShape());
    for (uint64_t i = 0; i < n; ++i)
        out[i] = val;
}

REGISTER_REF_KERNEL(OpType::FILL, 2, 2, matchFillF32_ND, runFillF32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::INT32}, {{1}, {1}}, {false, false},
                    {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
