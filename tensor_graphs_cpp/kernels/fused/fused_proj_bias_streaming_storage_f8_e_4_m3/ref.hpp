#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryFusedProjBiasStreamingStorage_F8_E4M3(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId w_copy = graph._copyto(inputs[1]);
    LogicalId w_cast = graph.cast(w_copy, DType::FLOAT32);
    int32_t perm[] = {1, 0};
    LogicalId w_t = graph.permute(w_cast, graph.constant({2}, perm, DType::INT32));
    LogicalId w_t_contig = graph.contiguous(w_t);
    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, static_cast<int32_t>(w_shape[1]), static_cast<int32_t>(w_shape[0])};
    LogicalId w_3d = graph.reshape(w_t_contig, graph.constant({3}, s3, DType::INT32));
    LogicalId dot = graph.dot(inputs[0], w_3d);
    return graph.add(dot, inputs[2]);
}
