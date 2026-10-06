#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryConv2dDot(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    LogicalId w = inputs[1];     // [1, out_c, in_c * k * k]
    LogicalId b_exp = inputs[2]; // [1, out_c, H_out, W_out]

    int32_t k = 3;
    int32_t stride = 1;
    int32_t pad = 1;

    LogicalId k_node = g.constant({1}, &k, DType::INT32);
    LogicalId s_node = g.constant({1}, &stride, DType::INT32);
    LogicalId p_node = g.constant({1}, &pad, DType::INT32);

    LogicalId col = g.im2col(x, k_node, s_node, p_node);
    LogicalId out_flat = g.dot(w, col);

    auto sO = g.getNode(b_exp).getShape();
    int32_t sh4[] = {1, static_cast<int32_t>(sO[1]), static_cast<int32_t>(sO[2]), static_cast<int32_t>(sO[3])};
    LogicalId out = g.reshape(out_flat, g.constant({4}, sh4, DType::INT32));

    return g.add(out, b_exp);
}
