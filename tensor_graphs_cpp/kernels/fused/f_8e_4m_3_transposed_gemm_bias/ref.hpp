#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryF8E4M3TransposedGEMMBias(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId w_cast = graph.cast(inputs[1], DType::FLOAT32);
    int32_t perm[] = {1, 0};
    LogicalId w_t = graph.contiguous(graph.permute(w_cast, graph.constant({2}, perm, DType::INT32)));
    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, static_cast<int32_t>(w_shape[1]), static_cast<int32_t>(w_shape[0])};
    LogicalId w_3d = graph.reshape(w_t, graph.constant({3}, s3, DType::INT32));
    LogicalId dot = graph.dot(inputs[0], w_3d);
    return graph.add(dot, inputs[2]);
}
