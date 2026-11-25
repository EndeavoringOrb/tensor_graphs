// Max over the last axis, thread pool. See ref.hpp.
#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/reduce_last_axis/ref.hpp"
#if defined(TG_HAS_NEON)
#include "core/common/thread_pool.hpp"

inline void runMaxF32LastAxisNEONThreaded(const KernelContext &ctx)
{
    checkReduceAxis(ctx);
    const ReduceRows rows(ctx.inViews[0]);
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    ThreadPool &pool = ThreadPool::get();
    const uint32_t tasks = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(rows.rows, pool.get_num_threads() * 2)));
    const uint64_t per = (rows.rows + tasks - 1) / tasks;
    pool.parallel_for(tasks, [&](uint32_t t) {
        const uint64_t begin = static_cast<uint64_t>(t) * per;
        const uint64_t end = std::min(rows.rows, begin + per);
        if (begin < end)
            maxLastAxisRows(in, out, rows, begin, end);
    });
}

REGISTER_KERNEL("Max_F32_LastAxis_NEON_Threaded", 2, 2, matchReduceLastAxis, runMaxF32LastAxisNEONThreaded, refFactoryMaxF32_ND_CUDA, {}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::INT32}, {{4, 8, 64}, {1}}, {false, false},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif
