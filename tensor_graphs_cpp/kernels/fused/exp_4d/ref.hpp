#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryExp4D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    auto shape = g.getNode(x).getShape();

    // 1. Create constant e
    float e_val = 2.7182818f;
    LogicalId e_node = g.constant({1}, &e_val, DType::FLOAT32);

    // 2. Reshape to 4D [1, 1, 1, 1]
    int32_t sh4[] = {1, 1, 1, 1};
    LogicalId e_4d = g.reshape(e_node, g.constant({4}, sh4, DType::INT32));

    // 3. Mirror expand_scalar_to_4d repeat logic
    LogicalId current_e = e_4d;
    for (int ax = 0; ax < 4; ++ax)
    {
        if (shape[ax] > 1)
        {
            int32_t r = (int32_t)shape[ax];
            int32_t a = ax;
            current_e = g.repeat(current_e, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
    }

    // 4. Return pow(e_expanded, x)
    return g.pow(current_e, x);
}
