#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

inline bool matchPowF32_1D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // inputs[0]: base, inputs[1]: exponent
    if (inputs[0].getShape().size() != 1 || inputs[1].getShape().size() != 1)
        return false;
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runPowF32_1D(const KernelContext &ctx)
{
    const float *base = static_cast<const float *>(ctx.inputs[0]);
    const float *exponent = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < n; ++i)
    {
        out[i] = std::pow(base[i], exponent[i]);
    }
}

inline LogicalId refFactoryPowF32_1D(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.pow(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Pow_1D", 2, 2, matchPowF32_1D, runPowF32_1D, refFactoryPowF32_1D, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
    {DType::FLOAT32, DType::FLOAT32},
    {{2048}, {2048}},
    {true, true},
    {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}}
);