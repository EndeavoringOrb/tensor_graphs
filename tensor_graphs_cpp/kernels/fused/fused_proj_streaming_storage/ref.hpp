#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryFusedProjStreamingStorage(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // inputs[0]: X [1, S, K] fp32 CPU
    // inputs[1]: W [N, K]    bf16 STORAGE (the raw on-disk weight node)

    // 1. Correctly model the COPY_TO from STORAGE to CPU bf16
    LogicalId w_copy = graph._copyto(inputs[1]);

    // 2. Perform the CAST on the CPU node: CPU bf16 -> CPU fp32
    LogicalId w_cast = graph.cast(w_copy, DType::FLOAT32);

    // [N, K] -> [K, N]  (this is the PERMUTE we are fusing away)
    int32_t perm[] = {1, 0};
    LogicalId w_t = graph.permute(w_cast, graph.constant({2}, perm, DType::INT32));

    // materialise  (this is the CONTIGUOUS we are fusing away)
    LogicalId w_t_contig = graph.contiguous(w_t);

    // [K, N] -> [1, K, N]  (this is the RESHAPE we are fusing away)
    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, static_cast<int32_t>(w_shape[1]), static_cast<int32_t>(w_shape[0])};
    LogicalId w_3d = graph.reshape(w_t_contig, graph.constant({3}, s3, DType::INT32));

    // The actual matmul (this is the DOT we are fusing away)
    return graph.dot(inputs[0], w_3d);
}
