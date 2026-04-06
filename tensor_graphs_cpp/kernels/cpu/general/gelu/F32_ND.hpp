// File: tensor_graphs_cpp/kernels/cpu/general/gelu/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>

inline bool matchGeluF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 1)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(inputs[0]) || !isContiguous(output))
        return false;
    return true;
}

inline void runGeluF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                          const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    float *out = static_cast<float *>(outputs[0]);
    uint64_t n = countElements(inViews[0].getShape());
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

inline uint32_t ref_gelu_expand_scalar(Graph &g, uint32_t scalar_id)
{
    int32_t shape_3d[] = {1, 1, 1};
    uint32_t out = g.reshape(scalar_id, g.constant({3}, shape_3d, DType::INT32));
    int32_t rep = 1;
    int32_t a0 = 0, a1 = 1, a2 = 2;
    uint32_t rN = g.constant({1}, &rep, DType::INT32);
    out = g.repeat(out, rN, g.constant({1}, &a0, DType::INT32));
    out = g.repeat(out, rN, g.constant({1}, &a1, DType::INT32));
    out = g.repeat(out, rN, g.constant({1}, &a2, DType::INT32));
    return out;
}

inline uint32_t refFactoryGelu(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t x_id = inputs[0];

    float c1_val = 0.044715f;
    uint32_t c1_node = ref_gelu_expand_scalar(graph, graph.constant({1}, &c1_val, DType::FLOAT32));

    float c2_val = 0.79788456f;
    uint32_t c2_node = ref_gelu_expand_scalar(graph, graph.constant({1}, &c2_val, DType::FLOAT32));

    uint32_t x_sq = graph.mul(x_id, x_id);
    uint32_t x_cube = graph.mul(x_sq, x_id);

    uint32_t term1 = graph.mul(x_cube, c1_node);
    uint32_t term2 = graph.add(x_id, term1);
    uint32_t term3 = graph.mul(term2, c2_node);

    float neg_two_val = -2.0f;
    uint32_t neg_two = ref_gelu_expand_scalar(graph, graph.constant({1}, &neg_two_val, DType::FLOAT32));

    float two_val = 2.0f;
    uint32_t two = ref_gelu_expand_scalar(graph, graph.constant({1}, &two_val, DType::FLOAT32));

    float e_val = 2.718281828459045f;
    uint32_t e_node = ref_gelu_expand_scalar(graph, graph.constant({1}, &e_val, DType::FLOAT32));

    float one_val = 1.0f;
    uint32_t one_node = ref_gelu_expand_scalar(graph, graph.constant({1}, &one_val, DType::FLOAT32));

    uint32_t neg_2x = graph.mul(term3, neg_two);
    uint32_t exp_neg_2x = graph.pow(e_node, neg_2x);

    uint32_t den = graph.add(one_node, exp_neg_2x);
    uint32_t quotient = graph.div(two, den);

    uint32_t neg_one = graph.neg(one_node);
    uint32_t tanh_result = graph.add(quotient, neg_one);

    uint32_t term4 = graph.add(one_node, tanh_result);

    float half_val = 0.5f;
    uint32_t half_node = ref_gelu_expand_scalar(graph, graph.constant({1}, &half_val, DType::FLOAT32));
    uint32_t term5 = graph.mul(x_id, half_node);

    return graph.mul(term5, term4);
}

REGISTER_KERNEL("Gelu", 1, matchGeluF32_ND, runGeluF32_ND, refFactoryGelu, {Backend::CPU}, {DType::FLOAT32}, {{1, 1, 2048}}, {true});