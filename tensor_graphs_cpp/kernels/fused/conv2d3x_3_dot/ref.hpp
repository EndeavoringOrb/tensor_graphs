#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryConv2d3x3Dot(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    LogicalId w = inputs[1]; // [1, out_c, in_c * k * k]

    int32_t k = 3;
    int32_t stride = 1;
    int32_t pad = 1;

    LogicalId k_node = g.constant({1}, &k, DType::INT32);
    LogicalId s_node = g.constant({1}, &stride, DType::INT32);
    LogicalId p_node = g.constant({1}, &pad, DType::INT32);

    LogicalId col = g.im2col(x, k_node, s_node, p_node);
    LogicalId out_flat = g.dot(w, col);

    auto sX = g.getNode(x).getShape();
    auto sW = g.getNode(w).getShape();
    uint32_t out_c = sW[1];
    uint32_t H = sX[2];
    uint32_t W = sX[3];

    int32_t sh4[] = {1, static_cast<int32_t>(out_c), static_cast<int32_t>(H), static_cast<int32_t>(W)};
    return g.reshape(out_flat, g.constant({4}, sh4, DType::INT32));
}
