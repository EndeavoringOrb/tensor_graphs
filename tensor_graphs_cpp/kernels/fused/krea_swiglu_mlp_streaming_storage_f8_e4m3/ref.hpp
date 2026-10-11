#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryKreaSwiGLU_MLP_StreamingStorage_F8_E4M3(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x = inputs[0];
    auto sX = graph.getNode(x).getShape();
    uint32_t S = sX[1];
    uint32_t K = sX[2];
    auto sWgate = graph.getNode(inputs[1]).getShape();
    uint32_t I = sWgate[0];

    int32_t perm[] = {1, 0};
    LogicalId perm_node = graph.constant({2}, perm, DType::INT32);

    LogicalId w_gate_copy = graph._copyto(inputs[1]);
    LogicalId w_gate_cast = graph.cast(w_gate_copy, DType::FLOAT32);
    LogicalId w_gate_t = graph.contiguous(graph.permute(w_gate_cast, perm_node));
    int32_t s3_gate[] = {1, (int32_t)K, (int32_t)I};
    LogicalId gate_mlp = graph.dot(x, graph.reshape(w_gate_t, graph.constant({3}, s3_gate, DType::INT32)));

    LogicalId w_up_copy = graph._copyto(inputs[2]);
    LogicalId w_up_cast = graph.cast(w_up_copy, DType::FLOAT32);
    LogicalId w_up_t = graph.contiguous(graph.permute(w_up_cast, perm_node));
    LogicalId up_mlp = graph.dot(x, graph.reshape(w_up_t, graph.constant({3}, s3_gate, DType::INT32)));

    LogicalId neg_one = graph.fill(-1.0f, {1, S, I});
    LogicalId neg_x = graph.mul(gate_mlp, neg_one);
    LogicalId exp_neg_x = graph.pow(graph.fill(TGConstants::E, {1, S, I}), neg_x);
    LogicalId one = graph.fill(1.0f, {1, S, I});
    LogicalId sig = graph.div(one, graph.add(one, exp_neg_x));
    LogicalId silu_gate = graph.mul(gate_mlp, sig);

    LogicalId swiglu = graph.mul(silu_gate, up_mlp);

    LogicalId w_down_copy = graph._copyto(inputs[3]);
    LogicalId w_down_cast = graph.cast(w_down_copy, DType::FLOAT32);
    LogicalId w_down_t = graph.contiguous(graph.permute(w_down_cast, perm_node));
    int32_t s3_down[] = {1, (int32_t)I, (int32_t)K};
    LogicalId mlp_out = graph.dot(swiglu, graph.reshape(w_down_t, graph.constant({3}, s3_down, DType::INT32)));

    return mlp_out;
}
