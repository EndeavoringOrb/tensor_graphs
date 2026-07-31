// File: tensor_graphs_cpp/kernels/cpu/general/gelu/F32_3D_Threaded.hpp
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchGeluF32_3D_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape().size() == 3 && isContiguous(output);
}

inline void runGeluF32_3D_Threaded(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint64_t chunk = (n + num_threads - 1) / num_threads;
        uint64_t start = t * chunk;
        uint64_t end = std::min(start + chunk, n);
        for (uint64_t i = start; i < end; ++i)
        {
            float x = in[i];
            float x3 = x * x * x;
            float inner = 0.79788456f * (x + 0.044715f * x3);
            float t_val = std::tanh(inner);
            out[i] = 0.5f * x * (1.0f + t_val);
        }
    });
}

inline LogicalId ref_gelu_broadcast_scalar_th1(Graph &g, LogicalId scalar_id, const std::vector<uint32_t> &target_shape)
{
    std::vector<int32_t> ones(target_shape.size(), 1);
    LogicalId out = g.reshape(scalar_id, g.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));
    for (uint64_t i = 0; i < target_shape.size(); ++i)
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

inline LogicalId refFactoryGelu_3D_Threaded(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x_id = inputs[0];
    const auto &target_shape = graph.getNode(x_id).getShape();

    float c1_val = 0.044715f;
    LogicalId c1_node =
        ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &c1_val, DType::FLOAT32), target_shape);

    float c2_val = 0.79788456f;
    LogicalId c2_node =
        ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &c2_val, DType::FLOAT32), target_shape);

    LogicalId x_sq = graph.mul(x_id, x_id);
    LogicalId x_cube = graph.mul(x_sq, x_id);

    LogicalId term1 = graph.mul(x_cube, c1_node);
    LogicalId term2 = graph.add(x_id, term1);
    LogicalId term3 = graph.mul(term2, c2_node);

    float neg_two_val = -2.0f;
    LogicalId neg_two =
        ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &neg_two_val, DType::FLOAT32), target_shape);

    float two_val = 2.0f;
    LogicalId two = ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &two_val, DType::FLOAT32), target_shape);

    float e_val = 2.718281828459045f;
    LogicalId e_node = ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &e_val, DType::FLOAT32), target_shape);

    float one_val = 1.0f;
    LogicalId one_node =
        ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &one_val, DType::FLOAT32), target_shape);

    LogicalId neg_2x = graph.mul(term3, neg_two);
    LogicalId exp_neg_2x = graph.pow(e_node, neg_2x);

    LogicalId den = graph.add(one_node, exp_neg_2x);
    LogicalId quotient = graph.div(two, den);

    LogicalId neg_one = graph.neg(one_node);
    LogicalId tanh_result = graph.add(quotient, neg_one);

    LogicalId term4 = graph.add(one_node, tanh_result);

    float half_val = 0.5f;
    LogicalId half_node =
        ref_gelu_broadcast_scalar_th1(graph, graph.constant({1}, &half_val, DType::FLOAT32), target_shape);
    LogicalId term5 = graph.mul(x_id, half_node);

    return graph.mul(term5, term4);
}

REGISTER_KERNEL("Gelu_3D_Threaded", 1, 1, matchGeluF32_3D_Threaded, runGeluF32_3D_Threaded, refFactoryGelu_3D_Threaded,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 1, 2048}}, {true},
                {{MemSpace(1, HandleType::CPP)}});