#pragma once
#include "kernels/fused/contiguous_nd/ref.hpp"

inline bool matchContiguousTransposed4DLastAxes(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return matchContiguousTransposed4D<false>(inputs, output);
}

inline void runContiguousTransposed4DLastAxes(const KernelContext &ctx)
{
    runContiguousTransposed4D<false>(ctx);
}

REGISTER_KERNEL("Contiguous_Transposed_4D_LastAxes", 1, 1, matchContiguousTransposed4DLastAxes, runContiguousTransposed4DLastAxes,
                refFactoryContiguous, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32}, {{2, 3, 35, 37}}, {false}, {{MemSpace(1, HandleType::CPP)}});
