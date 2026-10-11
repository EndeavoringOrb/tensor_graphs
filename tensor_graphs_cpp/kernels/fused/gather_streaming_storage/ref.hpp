#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryGatherStreamingStorage(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // inputs[0]: raw_weight_storage (STORAGE, BF16)
    // inputs[1]: indices (CPU, INT32)

    // 1. COPY_TO: STORAGE BF16 -> CPU BF16
    LogicalId w_cpu = graph._copyto(inputs[0]);

    // 2. CAST: CPU BF16 -> CPU FLOAT32
    LogicalId w_cast = graph.cast(w_cpu, DType::FLOAT32);

    // 3. GATHER: CPU FLOAT32
    return graph.gather(w_cast, inputs[1]);
}
