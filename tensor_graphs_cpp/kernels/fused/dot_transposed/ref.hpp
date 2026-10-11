#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryDotTransposedBF16(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId w_cast = graph.cast(inputs[1], DType::FLOAT32);

    int32_t perm_dims[] = {1, 0};
    LogicalId dims_node = graph.constant({2}, perm_dims, DType::INT32);
    LogicalId w_t = graph.permute(w_cast, dims_node);

    w_t = graph.contiguous(w_t);

    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, (int32_t)w_shape[1], (int32_t)w_shape[0]};
    LogicalId w_3d = graph.reshape(w_t, graph.constant({3}, s3, DType::INT32));

    return graph.dot(inputs[0], w_3d);
}
