// Threaded Scatter_F32_Rows: rows are split across the thread pool.
#pragma once

#include "core/common/thread_pool.hpp"
#include "kernels/fused/scatter_f32_nd/ref.hpp"

inline void runScatterF32RowsThreaded(const KernelContext &ctx)
{
    const uint64_t rows = validateScatterF32Rows(ctx);
    const uint32_t tasks = static_cast<uint32_t>(std::min<uint64_t>(rows, ThreadPool::get().get_num_threads()));
    ThreadPool::get().parallel_for(tasks, [&](uint32_t t) {
        scatterF32RowsRange(ctx, rows * t / tasks, rows * (t + 1) / tasks);
    });
}

REGISTER_KERNEL("Scatter_F32_Rows_Threaded", 5, 5, matchScatterF32Rows, runScatterF32RowsThreaded,
                refFactoryScatterF32_ND_CUDA, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32, DType::INT32},
                {{1, 8, 64}, {3}, {3}, {3}, {3}}, {true, false, false, false, false},
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}});
