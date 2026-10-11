#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryExpND(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    auto shape = g.getNode(x).getShape();
    float e_val = 2.7182818f;
    LogicalId e_node = g.constant({1}, &e_val, DType::FLOAT32);

    std::vector<int32_t> ones(shape.size(), 1);
    LogicalId current_e = g.reshape(e_node, g.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));

    for (uint64_t ax = 0; ax < shape.size(); ++ax)
    {
        if (shape[ax] > 1)
        {
            int32_t r = (int32_t)shape[ax];
            int32_t a = (int32_t)ax;
            current_e = g.repeat(current_e, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
    }
    return g.pow(current_e, x);
}
