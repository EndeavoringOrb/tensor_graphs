#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryHcPost(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x = inputs[0];
    LogicalId residual = inputs[1];
    LogicalId post = inputs[2];
    LogicalId comb = inputs[3];

    auto shapeX = graph.getNode(x).getShape();
    auto shapeRes = graph.getNode(residual).getShape();

    uint32_t seq_len = shapeX[1];
    uint32_t dim = shapeX[2];
    uint32_t hc_mult = shapeRes[2];

    int32_t sh4_x[] = {1, (int32_t)seq_len, 1, (int32_t)dim};
    LogicalId x_exp = graph.repeat(graph.reshape(x, graph.constant({4}, sh4_x, DType::INT32)), hc_mult, 2);

    int32_t sh4_p[] = {1, (int32_t)seq_len, (int32_t)hc_mult, 1};
    LogicalId post_exp = graph.repeat(graph.reshape(post, graph.constant({4}, sh4_p, DType::INT32)), dim, 3);

    LogicalId term1 = graph.mul(post_exp, x_exp);

    int32_t sh5_c[] = {1, (int32_t)seq_len, (int32_t)hc_mult, (int32_t)hc_mult, 1};
    LogicalId comb_exp = graph.repeat(graph.reshape(comb, graph.constant({5}, sh5_c, DType::INT32)), dim, 4);

    int32_t sh5_r[] = {1, (int32_t)seq_len, 1, (int32_t)hc_mult, (int32_t)dim};
    LogicalId res_exp = graph.repeat(graph.reshape(residual, graph.constant({5}, sh5_r, DType::INT32)), hc_mult, 2);

    int32_t ax_3 = 3;
    LogicalId term2_sum = graph.sum(graph.mul(comb_exp, res_exp), graph.constant({1}, &ax_3, DType::INT32));

    int32_t sh4_out[] = {1, (int32_t)seq_len, (int32_t)hc_mult, (int32_t)dim};
    return graph.add(term1, graph.reshape(term2_sum, graph.constant({4}, sh4_out, DType::INT32)));
}
