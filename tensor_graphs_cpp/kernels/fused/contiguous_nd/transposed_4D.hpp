#pragma once
#include <algorithm>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/contiguous_nd/ref.hpp"




inline bool matchContiguousTransposed4DMiddle(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return matchContiguousTransposed4D<true>(inputs, output);
}

inline void runContiguousTransposed4DMiddle(const KernelContext &ctx)
{
    runContiguousTransposed4D<true>(ctx);
}

REGISTER_KERNEL("Contiguous_Transposed_4D", 1, 1, matchContiguousTransposed4DMiddle, runContiguousTransposed4DMiddle,
                refFactoryContiguous, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32}, {{2, 3, 35, 37}}, {false}, {{MemSpace(1, HandleType::CPP)}});
