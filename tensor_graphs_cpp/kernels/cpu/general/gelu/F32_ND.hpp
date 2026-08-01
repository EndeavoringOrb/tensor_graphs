#pragma once
#include <cmath>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchGeluF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runGeluF32_ND(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());
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

// Helper to expand a scalar constant to match a specific target shape
inline LogicalId ref_gelu_broadcast_scalar(Graph &g, LogicalId scalar_id, const std::vector<uint32_t> &target_shape)
{
    // 1. Reshape to matching rank filled with 1s
    std::vector<int32_t> ones(target_shape.size(), 1);
    LogicalId out = g.reshape(scalar_id, g.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));

    // 2. Repeat for every dimension where target_shape > 1
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

inline LogicalId refFactoryGelu(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x_id = inputs[0];
    const auto &target_shape = graph.getNode(x_id).getShape();

    // Broadcast all constants to the input shape to satisfy atomic shape matching
    float c1_val = 0.044715f;
    LogicalId c1_node = ref_gelu_broadcast_scalar(graph, graph.constant({1}, &c1_val, DType::FLOAT32), target_shape);

    float c2_val = 0.79788456f;
    LogicalId c2_node = ref_gelu_broadcast_scalar(graph, graph.constant({1}, &c2_val, DType::FLOAT32), target_shape);

    LogicalId x_sq = graph.mul(x_id, x_id);
    LogicalId x_cube = graph.mul(x_sq, x_id);

    LogicalId term1 = graph.mul(x_cube, c1_node);
    LogicalId term2 = graph.add(x_id, term1);
    LogicalId term3 = graph.mul(term2, c2_node);

    float neg_two_val = -2.0f;
    LogicalId neg_two =
        ref_gelu_broadcast_scalar(graph, graph.constant({1}, &neg_two_val, DType::FLOAT32), target_shape);

    float two_val = 2.0f;
    LogicalId two = ref_gelu_broadcast_scalar(graph, graph.constant({1}, &two_val, DType::FLOAT32), target_shape);

    float e_val = 2.718281828459045f;
    LogicalId e_node = ref_gelu_broadcast_scalar(graph, graph.constant({1}, &e_val, DType::FLOAT32), target_shape);

    float one_val = 1.0f;
    LogicalId one_node = ref_gelu_broadcast_scalar(graph, graph.constant({1}, &one_val, DType::FLOAT32), target_shape);

    LogicalId neg_2x = graph.mul(term3, neg_two);
    LogicalId exp_neg_2x = graph.pow(e_node, neg_2x);

    LogicalId den = graph.add(one_node, exp_neg_2x);
    LogicalId quotient = graph.div(two, den);

    LogicalId neg_one = graph.neg(one_node);
    LogicalId tanh_result = graph.add(quotient, neg_one);

    LogicalId term4 = graph.add(one_node, tanh_result);

    float half_val = 0.5f;
    LogicalId half_node =
        ref_gelu_broadcast_scalar(graph, graph.constant({1}, &half_val, DType::FLOAT32), target_shape);
    LogicalId term5 = graph.mul(x_id, half_node);

    return graph.mul(term5, term4);
}

REGISTER_KERNEL("Gelu", 1, 1, matchGeluF32_ND, runGeluF32_ND, refFactoryGelu, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 1, 2048}}, {true},
                {{MemSpace(1, HandleType::CPP)}});
