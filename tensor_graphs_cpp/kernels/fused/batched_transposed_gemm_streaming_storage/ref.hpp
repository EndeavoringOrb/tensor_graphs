#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryBatchedTransposedGEMM_StreamingStorage(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // STORAGE bf16 -> CPU bf16 (this is the COPY_TO we are fusing away)
    LogicalId copy_w = graph._copyto(inputs[1]);
    // CPU bf16 -> CPU fp32 (this is the CAST we are fusing away)
    LogicalId cast_w = graph.cast(copy_w, DType::FLOAT32);
    // [E, O, H] -> [E, H, O]  (this is the PERMUTE+CONTIGUOUS we are fusing away)
    int32_t perm[] = {0, 2, 1};
    LogicalId perm_w = graph.permute(cast_w, graph.constant({3}, perm, DType::INT32));
    LogicalId contig_w = graph.contiguous(perm_w);
    // The actual batched dot
    return graph.dot(inputs[0], contig_w);
}
