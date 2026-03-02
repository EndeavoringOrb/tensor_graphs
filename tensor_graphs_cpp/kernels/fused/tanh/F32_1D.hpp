#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

// ---------------------------------------------------------
// FUSED KERNEL: TANH F32 1D (Contiguous)
// Formula: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
// ---------------------------------------------------------

// Match Function: Verifies DType, Rank (1D), Shape Match, and Contiguity
bool matchTanhF32_1D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 1)
        return false;

    // Check Dtypes
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Check Ranks (Must be 1D)
    if (inputs[0].shape.size() != 1 || output.shape.size() != 1)
        return false;

    // Check Dimension Matching
    if (inputs[0].shape[0] != output.shape[0])
        return false;

    // Check Contiguity (Required for this specific kernel optimization)
    if (!inputs[0].view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

// Run Function: Element-wise Tanh Computation
void runTanhF32_1D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                   const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    // Cast raw memory to F32 pointers
    const float *x = static_cast<const float *>(inputs[0]);
    float *out = static_cast<float *>(outputs[0]);

    uint32_t size = inViews[0].shape[0];

    for (uint32_t i = 0; i < size; ++i)
    {
        // Compute tanh using the exponential formula: (e^x - e^-x) / (e^x + e^-x)
        float exp_x = std::exp(x[i]);
        float exp_neg_x = std::exp(-x[i]);

        float num = exp_x - exp_neg_x;
        float den = exp_x + exp_neg_x;

        out[i] = num / den;
    }
}

// Reference Factory: Builds the equivalent subgraph using atomic operations
uint32_t refFactoryTanh(const std::vector<uint32_t> &inputs, Graph &graph, MemoryManager &memManager)
{
    if (inputs.size() != 1)
        throw std::runtime_error("Tanh requires 1 input");
    uint32_t x = inputs[0];

    // --- Create Constant 'e' ---
    uint32_t e_id = graph.allocateId();
    uint64_t offset = memManager.allocate(Backend::CPU, e_id, sizeof(float), StorageType::PERSISTENT);

    TensorView e_view;
    e_view.baseOffset = offset;
    e_view.shape = {1};
    e_view.strides = {1};

    // Actually register the input node for 'e'
    uint32_t e_node = graph.input({1}, DType::FLOAT32, e_view);

    // Initialize the memory for 'e' (Ensure arena is init'd or handle this in a loader)
    auto &cpuBuf = memManager.buffers.at(Backend::CPU);
    if (!cpuBuf.arena.empty())
    {
        float *e_ptr = reinterpret_cast<float *>(cpuBuf.arena.data() + offset);
        *e_ptr = 2.718281828459f; // TODO: should I be writing this here?
    }

    // --- Decompose: tanh(x) = (e^x - e^-x) / (e^x + e^-x) ---

    // exp_x = e^x
    uint32_t exp_x = graph.pow(e_node, x);

    // exp_neg_x = e^(-x)
    uint32_t neg_x = graph.neg(x);
    uint32_t exp_neg_x = graph.pow(e_node, neg_x);

    // num = exp_x - exp_neg_x
    uint32_t neg_exp_neg = graph.neg(exp_neg_x);
    uint32_t num = graph.add(exp_x, neg_exp_neg);

    // den = exp_x + exp_neg_x
    uint32_t den = graph.add(exp_x, exp_neg_x);

    // result = num / den
    return graph.div(num, den);
}

// Register the FUSED kernel
REGISTER_FUSED_KERNEL("Tanh", 1, Backend::CPU, matchTanhF32_1D, runTanhF32_1D, refFactoryTanh);
