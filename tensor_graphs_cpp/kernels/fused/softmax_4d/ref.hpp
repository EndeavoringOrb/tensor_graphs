#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactorySoftmax4D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    auto s = g.getNode(x).getShape();
    int32_t ax = -1;
    LogicalId axis_node = g.constant({1}, &ax, DType::INT32);
    LogicalId m_rep = g.constant({1}, (int32_t *)&s[3], DType::INT32);
    LogicalId ax_rep = g.constant({1}, (int32_t *)&ax, DType::INT32);

    LogicalId max_s = g.repeat(g.max(x, axis_node), m_rep, ax_rep);
    LogicalId shifted = g.add(x, g.neg(max_s));

    float e_v = 2.718281828f;
    LogicalId e_n = g.constant({1}, &e_v, DType::FLOAT32);
    int32_t sh4[] = {1, 1, 1, 1};
    LogicalId e_b = g.reshape(e_n, g.constant({4}, sh4, DType::INT32));
    for (int i = 0; i < 4; ++i)
    {
        int32_t r = (int32_t)s[i];
        if (r <= 1)
            continue;
        int32_t a = i;
        e_b = g.repeat(e_b, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }

    LogicalId exps = g.pow(e_b, shifted);
    LogicalId sums = g.repeat(g.sum(exps, axis_node), m_rep, ax_rep);
    return g.div(exps, sums);
}
