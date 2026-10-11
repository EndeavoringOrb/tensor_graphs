#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryAdd3D1D(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Add 3D+1D requires 2 inputs");

    LogicalId id3D = inputs[0];
    LogicalId id1D = inputs[1];
    auto shape3D = graph.getNode(id3D).getShape();
    auto shape1D = graph.getNode(id1D).getShape();

    int32_t reshape_dims[] = {1, 1, (int32_t)shape1D[0]};
    LogicalId shape_node = graph.constant({3}, reshape_dims, DType::INT32);
    LogicalId reshaped = graph.reshape(id1D, shape_node);

    int32_t b_repeats[] = {(int32_t)shape3D[0]};
    int32_t b_axis[] = {0};
    LogicalId rep_b = graph.constant({1}, b_repeats, DType::INT32);
    LogicalId ax_b = graph.constant({1}, b_axis, DType::INT32);
    LogicalId repeated_b = graph.repeat(reshaped, rep_b, ax_b);

    int32_t s_repeats[] = {(int32_t)shape3D[1]};
    int32_t s_axis[] = {1};
    LogicalId rep_s = graph.constant({1}, s_repeats, DType::INT32);
    LogicalId ax_s = graph.constant({1}, s_axis, DType::INT32);
    LogicalId expanded = graph.repeat(repeated_b, rep_s, ax_s);

    return graph.add(id3D, expanded);
}
