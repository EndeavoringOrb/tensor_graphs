// File: tensor_graphs_cpp/kernels/cpu/general/gelu/F32_3D_Threaded.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>

inline bool matchGeluF32_3D_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape().size() == 3 && isContiguous(inputs[0]) && isContiguous(output);
}

inline void runGeluF32_3D_Threaded(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                   const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    float *out = static_cast<float *>(outputs[0]);
    uint64_t n = countElements(inViews[0].getShape());

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    uint64_t chunk = (n + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint64_t start = t * chunk;
            uint64_t end = std::min(start + chunk, n);
            for (uint64_t i = start; i < end; ++i)
            {
                float x = in[i];
                float x3 = x * x * x;
                float inner = 0.79788456f * (x + 0.044715f * x3);
                float t_val = std::tanh(inner);
                out[i] = 0.5f * x * (1.0f + t_val);
            } });
    }
    for (auto &w : workers)
        w.join();
}

inline uint32_t ref_gelu_broadcast_scalar_th1(Graph &g, uint32_t scalar_id, const std::vector<uint32_t> &target_shape)
{
    std::vector<int32_t> ones(target_shape.size(), 1);
    uint32_t out = g.reshape(scalar_id, g.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));
    for (size_t i = 0; i < target_shape.size(); ++i)
    {
        if (target_shape[i] > 1)
        {
            int32_t rep = (int32_t)target_shape[i];
            int32_t axis = (int32_t)i;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &axis, DType::INT32));
        }
    }
    return out;
}

inline uint32_t refFactoryGelu_3D_Threaded(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t x_id = inputs[0];
    const auto &target_shape = graph.getNode(x_id).getShape();

    float c1_val = 0.044715f;
    uint32_t c1_node = ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &c1_val, DType::FLOAT32), target_shape);

    float c2_val = 0.79788456f;
    uint32_t c2_node = ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &c2_val, DType::FLOAT32), target_shape);

    uint32_t x_sq = graph.mul(x_id, x_id);
    uint32_t x_cube = graph.mul(x_sq, x_id);

    uint32_t term1 = graph.mul(x_cube, c1_node);
    uint32_t term2 = graph.add(x_id, term1);
    uint32_t term3 = graph.mul(term2, c2_node);

    float neg_two_val = -2.0f;
    uint32_t neg_two = ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &neg_two_val, DType::FLOAT32), target_shape);

    float two_val = 2.0f;
    uint32_t two = ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &two_val, DType::FLOAT32), target_shape);

    float e_val = 2.718281828459045f;
    uint32_t e_node = ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &e_val, DType::FLOAT32), target_shape);

    float one_val = 1.0f;
    uint32_t one_node = ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &one_val, DType::FLOAT32), target_shape);

    uint32_t neg_2x = graph.mul(term3, neg_two);
    uint32_t exp_neg_2x = graph.pow(e_node, neg_2x);

    uint32_t den = graph.add(one_node, exp_neg_2x);
    uint32_t quotient = graph.div(two, den);

    uint32_t neg_one = graph.neg(one_node);
    uint32_t tanh_result = graph.add(quotient, neg_one);

    uint32_t term4 = graph.add(one_node, tanh_result);

    float half_val = 0.5f;
    uint32_t half_node = ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &half_val, DType::FLOAT32), target_shape);
    uint32_t term5 = graph.mul(x_id, half_node);

    return graph.mul(term5, term4);
}

REGISTER_KERNEL("Gelu_3D_Threaded", 1, matchGeluF32_3D_Threaded, runGeluF32_3D_Threaded, refFactoryGelu_3D_Threaded, {Backend::CPU}, {DType::FLOAT32}, {{1, 1, 2048}}, {true}, {{Backend::CPU}});