#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryMul3D_Scalar(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Mul 3D+Scalar requires 2 inputs");

    const auto &shape3D = graph.getNode(inputs[0]).getShape();

    // 1. Reshape [1] -> [1, 1, 1]
    int32_t reshape_dims[] = {1, 1, 1};
    LogicalId out = graph.reshape(inputs[1], graph.constant({3}, reshape_dims, DType::INT32));

    // 2. Repeat for B, S, and D
    for (int i = 0; i < 3; ++i)
    {
        int32_t rep = (int32_t)shape3D[i];
        int32_t axis = i;
        out = graph.repeat(out, graph.constant({1}, &rep, DType::INT32), graph.constant({1}, &axis, DType::INT32));
    }

    return graph.mul(inputs[0], out);
}
