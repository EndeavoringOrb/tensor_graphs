#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryKreaGatedAttnProj_StreamingStorage_F8_E4M3(const std::vector<LogicalId> &inputs,
                                                                      Graph &graph)
{
    LogicalId h = inputs[0];
    LogicalId ctx_flat = inputs[1];
    auto sH = graph.getNode(h).getShape();
    uint32_t S = sH[1];
    uint32_t D = sH[2];

    int32_t perm[] = {1, 0};
    LogicalId perm_node = graph.constant({2}, perm, DType::INT32);
    int32_t s3[] = {1, (int32_t)D, (int32_t)D};
    LogicalId s3_node = graph.constant({3}, s3, DType::INT32);

    LogicalId w_gate_copy = graph._copyto(inputs[2]);
    LogicalId w_gate_cast = graph.cast(w_gate_copy, DType::FLOAT32);
    LogicalId w_gate_t = graph.contiguous(graph.permute(w_gate_cast, perm_node));
    LogicalId gate = graph.dot(h, graph.reshape(w_gate_t, s3_node));

    LogicalId neg_one = graph.fill(-1.0f, {1, S, D});
    LogicalId neg_gate = graph.mul(gate, neg_one);
    LogicalId exp_neg_gate = graph.pow(graph.fill(TGConstants::E, {1, S, D}), neg_gate);
    LogicalId one = graph.fill(1.0f, {1, S, D});
    LogicalId sig = graph.div(one, graph.add(one, exp_neg_gate));

    LogicalId gated_attn = graph.mul(ctx_flat, sig);

    LogicalId w_o_copy = graph._copyto(inputs[3]);
    LogicalId w_o_cast = graph.cast(w_o_copy, DType::FLOAT32);
    LogicalId w_o_t = graph.contiguous(graph.permute(w_o_cast, perm_node));
    LogicalId attn_proj = graph.dot(gated_attn, graph.reshape(w_o_t, s3_node));

    return attn_proj;
}
