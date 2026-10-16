// Scatter of a contiguous float32 update block into a contiguous output, one
// memcpy per innermost row. Unit steps only (the planner's scatters write
// dense boxes); anything else fails loudly.
#pragma once

#include <cstring>

#include "core/kernels.hpp"
#include "kernels/fused/scatter_f32_nd/ref.hpp"



inline void runScatterF32Rows(const KernelContext &ctx)
{
    const uint64_t rows = validateScatterF32Rows(ctx);
    scatterF32RowsRange(ctx, 0, rows);
}

REGISTER_KERNEL("Scatter_F32_Rows", 5, 5, matchScatterF32Rows, runScatterF32Rows, refFactoryScatterF32_ND_CUDA, {},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32, DType::INT32},
                {{1, 8, 64}, {3}, {3}, {3}, {3}}, {true, false, false, false, false},
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}});
