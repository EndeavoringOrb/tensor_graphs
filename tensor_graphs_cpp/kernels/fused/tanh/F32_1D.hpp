#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/tanh/ref.hpp"
// ---------------------------------------------------------
// FUSED KERNEL: TANH F32 1D (Contiguous)
// Formula: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
// ---------------------------------------------------------

bool matchTanhF32_1D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 1 || output.getShape().size() != 1)
        return false;
    if (inputs[0].getShape()[0] != output.getShape()[0])
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

void runTanhF32_1D(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    uint32_t size = ctx.inViews[0].getShape()[0];

    for (uint32_t i = 0; i < size; ++i)
    {
        float exp_x = std::exp(x[i]);
        float exp_neg_x = std::exp(-x[i]);
        out[i] = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
    }
}



REGISTER_KERNEL("Tanh", 1, 1, matchTanhF32_1D, runTanhF32_1D, refFactoryTanh, {0}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1}}, {true}, {{MemSpace(1, HandleType::CPP)}});
