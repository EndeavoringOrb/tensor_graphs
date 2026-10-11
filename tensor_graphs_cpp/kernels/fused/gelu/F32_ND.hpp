#pragma once
#include <cmath>

#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/gelu/ref.hpp"
inline bool matchGeluF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runGeluF32_ND(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());
    for (uint64_t i = 0; i < n; ++i)
    {
        float x = in[i];
        float x_sq = x * x;
        float x_cube = x_sq * x;
        float term3 = (x + 0.044715f * x_cube) * 0.79788456f;
        float exp_neg_2x = std::exp(-2.0f * term3);
        float tanh_res = (2.0f / (1.0f + exp_neg_2x)) - 1.0f;
        out[i] = 0.5f * x * (1.0f + tanh_res);
    }
}

REGISTER_KERNEL("Gelu", 1, 1, matchGeluF32_ND, runGeluF32_ND, refFactoryGelu, {0}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 1, 2048}}, {true},
                {{MemSpace(1, HandleType::CPP)}});
