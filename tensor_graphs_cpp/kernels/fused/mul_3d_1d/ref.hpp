#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryMul3D_1D(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Mul 3D+1D requires 2 inputs");

    const auto &shape3D = graph.getNode(inputs[0]).getShape();
    const auto &shape1D = graph.getNode(inputs[1]).getShape();

    // 1. Reshape [D] -> [1, 1, D]
    int32_t reshape_dims[] = {1, 1, (int32_t)shape1D[0]};
    LogicalId out = graph.reshape(inputs[1], graph.constant({3}, reshape_dims, DType::INT32));

    // 2. Repeat axis 0 (Batch)
    int32_t b_rep = (int32_t)shape3D[0];
    int32_t b_axis = 0;
    out = graph.repeat(out, graph.constant({1}, &b_rep, DType::INT32), graph.constant({1}, &b_axis, DType::INT32));

    // 3. Repeat axis 1 (Sequence)
    int32_t s_rep = (int32_t)shape3D[1];
    int32_t s_axis = 1;
    out = graph.repeat(out, graph.constant({1}, &s_rep, DType::INT32), graph.constant({1}, &s_axis, DType::INT32));

    return graph.mul(inputs[0], out);
}
