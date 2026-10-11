#pragma once
#include <cmath>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchLogF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return isContiguous(output);
}

inline void runLogF32_ND(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.outViews[0].getShape());
    for (uint64_t i = 0; i < n; ++i)
        out[i] = std::log(in[i]);
}

REGISTER_REF_KERNEL(OpType::LOG, 1, 1, matchLogF32_ND, runLogF32_ND, MemSpace(1, HandleType::CPP),
                    {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{8, 32}}, {true},
                    {{MemSpace(1, HandleType::CPP)}});