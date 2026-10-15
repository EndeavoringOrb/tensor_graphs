#pragma once
#include "kernels/fused/gemma_rms_norm_fill/ref.hpp"

inline LogicalId refFactoryGemmaRmsNormFillHeads(const std::vector<LogicalId> &inputs, Graph &g)
{
    return gemmaRmsNormFillPattern(inputs, g, true);
}

