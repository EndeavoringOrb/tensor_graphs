#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryIm2ColF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 4)
        Error::throw_err("Im2Col ND requires 4 inputs");

    return graph.im2col(inputs[0], inputs[1], inputs[2], inputs[3]);
}
