#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

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

LogicalId refFactoryTanh(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 1)
        Error::throw_err("Tanh requires 1 input");
    LogicalId x = inputs[0];
    uint32_t n_elements = graph.getNode(x).getShape()[0];

    // Create Constant 'e' as a scalar
    float e_val = 2.718281828459f;
    LogicalId e_scalar = graph.constant({1}, &e_val, DType::FLOAT32);

    // Explicitly broadcast 'e' to match the shape of 'x' [N]
    int32_t repeats_val = (int32_t)n_elements;
    int32_t axis_val = 0;
    LogicalId repeats_node = graph.constant({1}, &repeats_val, DType::INT32);
    LogicalId axis_node = graph.constant({1}, &axis_val, DType::INT32);
    LogicalId e_node = graph.repeat(e_scalar, repeats_node, axis_node);

    // Decompose using explicitly matched shapes
    LogicalId exp_x = graph.pow(e_node, x);

    LogicalId neg_x = graph.neg(x);
    LogicalId exp_neg_x = graph.pow(e_node, neg_x);

    LogicalId neg_exp_neg = graph.neg(exp_neg_x);
    LogicalId num = graph.add(exp_x, neg_exp_neg);

    LogicalId den = graph.add(exp_x, exp_neg_x);

    return graph.div(num, den);
}

REGISTER_KERNEL("Tanh", 1, 1, matchTanhF32_1D, runTanhF32_1D, refFactoryTanh, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1}}, {true}, {{MemSpace(1, HandleType::CPP)}});
