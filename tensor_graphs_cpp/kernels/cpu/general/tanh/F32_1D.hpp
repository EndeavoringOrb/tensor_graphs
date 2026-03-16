#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

// ---------------------------------------------------------
// FUSED KERNEL: TANH F32 1D (Contiguous)
// Formula: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
// ---------------------------------------------------------

bool matchTanhF32_1D(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 1)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].shape.size() != 1 || output.shape.size() != 1)
        return false;
    if (inputs[0].shape[0] != output.shape[0])
        return false;
    if (!inputs[0].view.isContiguous() || !output.view.isContiguous())
        return false;
    return true;
}

void runTanhF32_1D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                   const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *x = static_cast<const float *>(inputs[0]);
    float *out = static_cast<float *>(outputs[0]);

    uint32_t size = inViews[0].shape[0];

    for (uint32_t i = 0; i < size; ++i)
    {
        float exp_x = std::exp(x[i]);
        float exp_neg_x = std::exp(-x[i]);
        out[i] = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
    }
}

uint32_t refFactoryTanh(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 1)
        Error::throw_error("Tanh requires 1 input");
    uint32_t x = inputs[0];
    uint32_t n_elements = graph.nodes[x].shape[0];

    // Create Constant 'e' as a scalar
    float e_val = 2.718281828459f;
    uint32_t e_scalar = graph.constant({1}, &e_val, DType::FLOAT32);

    // Explicitly broadcast 'e' to match the shape of 'x' [N]
    int32_t repeats_val = (int32_t)n_elements;
    int32_t axis_val = 0;
    uint32_t repeats_node = graph.constant({1}, &repeats_val, DType::INT32);
    uint32_t axis_node = graph.constant({1}, &axis_val, DType::INT32);
    uint32_t e_node = graph.repeat(e_scalar, repeats_node, axis_node);

    // Decompose using explicitly matched shapes
    uint32_t exp_x = graph.pow(e_node, x);

    uint32_t neg_x = graph.neg(x);
    uint32_t exp_neg_x = graph.pow(e_node, neg_x);

    uint32_t neg_exp_neg = graph.neg(exp_neg_x);
    uint32_t num = graph.add(exp_x, neg_exp_neg);

    uint32_t den = graph.add(exp_x, exp_neg_x);

    return graph.div(num, den);
}

REGISTER_KERNEL("Tanh", 1, Backend::CPU, matchTanhF32_1D, runTanhF32_1D, refFactoryTanh, {DType::FLOAT32}, {{1}}, {true});