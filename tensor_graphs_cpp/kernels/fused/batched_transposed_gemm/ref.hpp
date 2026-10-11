#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryBatchedTransposedGEMM(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // Reconstructs the unoptimized pattern: Dot(X, Contiguous(Permute(W)))
    int32_t perm[] = {0, 2, 1};
    LogicalId perm_node = graph.constant({3}, perm, DType::INT32);
    LogicalId transposed = graph.contiguous(graph.permute(inputs[1], perm_node));
    return graph.dot(inputs[0], transposed);
}
