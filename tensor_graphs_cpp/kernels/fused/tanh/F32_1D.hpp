#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

// ---------------------------------------------------------
// FUSED KERNEL: TANH F32 1D (Contiguous)
// Formula: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
// ---------------------------------------------------------

bool matchTanhF32_1D(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

uint32_t refFactoryTanh(const std::vector<uint32_t> &inputs, Graph &graph, MemoryManager &memManager)
{
    if (inputs.size() != 1)
        throw std::runtime_error("Tanh requires 1 input");
    uint32_t x = inputs[0];

    // Create Constant 'e'
    uint32_t e_id = graph.allocateId();
    uint64_t offset = memManager.allocate(Backend::CPU, e_id, sizeof(float), StorageType::PERSISTENT);

    TensorView e_view = memManager.getView(Backend::CPU, e_id, {1});

    uint32_t e_node = graph.inputWithId(e_id, {1}, DType::FLOAT32, e_view);

    float e_val = 2.718281828459f;
    memManager.write(Backend::CPU, e_id, &e_val, sizeof(float));

    // Decompose
    uint32_t exp_x = graph.pow(e_node, x);

    uint32_t neg_x = graph.neg(x);
    uint32_t exp_neg_x = graph.pow(e_node, neg_x);

    uint32_t neg_exp_neg = graph.neg(exp_neg_x);
    uint32_t num = graph.add(exp_x, neg_exp_neg);

    uint32_t den = graph.add(exp_x, exp_neg_x);

    return graph.div(num, den);
}

REGISTER_FUSED_KERNEL("Tanh", 1, Backend::CPU, matchTanhF32_1D, runTanhF32_1D, refFactoryTanh);