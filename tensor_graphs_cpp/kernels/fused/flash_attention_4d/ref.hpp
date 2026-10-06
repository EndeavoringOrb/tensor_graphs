#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryFlashAttention4D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId Q = inputs[0], K = inputs[1], V = inputs[2];
    int32_t perm[] = {0, 1, 3, 2};
    LogicalId K_t = g.contiguous(g.permute(K, g.constant({4}, perm, DType::INT32)));
    LogicalId scores = g.dot(Q, K_t);

    int32_t axis = -1;
    LogicalId axis_node = g.constant({1}, &axis, DType::INT32);
    auto Q_shape = g.getNode(Q).getShape();
    auto K_shape = g.getNode(K).getShape();
    std::vector<uint32_t> s_shape = {Q_shape[0], Q_shape[1], Q_shape[2], K_shape[2]};
    int32_t S_val = (int32_t)s_shape[s_shape.size() - 1];
    LogicalId m_rep = g.constant({1}, &S_val, DType::INT32);
    LogicalId ax_rep = g.constant({1}, &axis, DType::INT32);

    LogicalId max_s = g.max(scores, axis_node);
    LogicalId max_expanded = g.repeat(max_s, m_rep, ax_rep);
    LogicalId shifted = g.add(scores, g.neg(max_expanded));

    float e_v = 2.718281828459045f;
    LogicalId e_n = g.constant({1}, &e_v, DType::FLOAT32);
    int32_t sh4[] = {1, 1, 1, 1};
    LogicalId e_b = g.reshape(e_n, g.constant({4}, sh4, DType::INT32));

    for (int i = 0; i < 4; ++i)
    {
        int32_t r = (int32_t)s_shape[i];
        if (r <= 1)
            continue;
        int32_t a = i;
        e_b = g.repeat(e_b, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }

    LogicalId exps = g.pow(e_b, shifted);
    LogicalId sums = g.repeat(g.sum(exps, axis_node), m_rep, ax_rep);
    LogicalId probs = g.div(exps, sums);

    return g.dot(probs, V);
}
