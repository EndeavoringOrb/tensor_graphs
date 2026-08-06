#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchGeluF32_3D_Inplace_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape().size() == 3 && isContiguous(output);
}

inline void runGeluF32_3D_Inplace_Threaded(const KernelContext &ctx)
{
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint64_t chunk_size = (n + num_threads - 1) / num_threads;
        uint64_t start = t * chunk_size;
        uint64_t end = std::min(start + chunk_size, n);
        for (uint64_t i = start; i < end; ++i)
        {
            float x = out[i];
            float x3 = x * x * x;
            float inner = 0.79788456f * (x + 0.044715f * x3);
            float t_val = std::tanh(inner);
            out[i] = 0.5f * x * (1.0f + t_val);
        }
    });
}

inline LogicalId refFactoryGelu_Threaded(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x_id = inputs[0];
    const auto &target_shape = graph.getNode(x_id).getShape();

    float c1_val = 0.044715f;
    int32_t ones_arr[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    LogicalId ones_shape = graph.constant({(uint32_t)target_shape.size()}, ones_arr, DType::INT32);

    auto bcast = [&](float val) {
        LogicalId out = graph.reshape(graph.constant({1}, &val, DType::FLOAT32), ones_shape);
        for (uint64_t i = 0; i < target_shape.size(); ++i)
        {
            if (target_shape[i] > 1)
            {
                int32_t rep = (int32_t)target_shape[i];
                int32_t axis = (int32_t)i;
                out = graph.repeat(out, graph.constant({1}, &rep, DType::INT32),
                                   graph.constant({1}, &axis, DType::INT32));
            }
        }
        return out;
    };

    LogicalId c1_node = bcast(c1_val);
    LogicalId c2_node = bcast(0.79788456f);
    LogicalId neg_two = bcast(-2.0f);
    LogicalId two = bcast(2.0f);
    LogicalId e_node = bcast(2.718281828459045f);
    LogicalId one_node = bcast(1.0f);
    LogicalId half_node = bcast(0.5f);

    LogicalId x_sq = graph.mul(x_id, x_id);
    LogicalId x_cube = graph.mul(x_sq, x_id);
    LogicalId term1 = graph.mul(x_cube, c1_node);
    LogicalId term2 = graph.add(x_id, term1);
    LogicalId term3 = graph.mul(term2, c2_node);
    LogicalId neg_2x = graph.mul(term3, neg_two);
    LogicalId exp_neg_2x = graph.pow(e_node, neg_2x);
    LogicalId den = graph.add(one_node, exp_neg_2x);
    LogicalId quotient = graph.div(two, den);
    LogicalId neg_one = graph.neg(one_node);
    LogicalId tanh_result = graph.add(quotient, neg_one);
    LogicalId term4 = graph.add(one_node, tanh_result);
    LogicalId term5 = graph.mul(x_id, half_node);

    return graph.mul(term5, term4);
}

REGISTER_KERNEL("Gelu_3D_inplace_Threaded", 1, 1, matchGeluF32_3D_Inplace_Threaded, runGeluF32_3D_Inplace_Threaded,
                refFactoryGelu_Threaded, {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32}, {{1, 8, 2048}}, {true}, {{MemSpace(1, HandleType::CPP)}});