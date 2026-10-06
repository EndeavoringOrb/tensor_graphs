#pragma once

#include "core/kernels.hpp"

inline LogicalId ref_copy_cuda_cuda(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph._copyto(inputs[0]);
}
