// Gemma RMS norm (fills as inputs, per-head weight repeat), thread pool. See ref.hpp.
#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/gemma_rms_norm_fill_heads/ref.hpp"
#if defined(TG_HAS_NEON)
#include <algorithm>
#include "core/common/thread_pool.hpp"

inline void runGemmaRMSNormFillHeadsF323DThreaded(const KernelContext &ctx)
{
    const auto &shape = ctx.inViews[0].getShape();
    const uint32_t D = shape[2];
    const uint64_t rows = static_cast<uint64_t>(shape[0]) * shape[1];
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    const float *w = static_cast<const float *>(ctx.inputs[1]);
    const float n = *static_cast<const float *>(ctx.inputs[2]);
    const float eps = *static_cast<const float *>(ctx.inputs[3]);
    const float half = *static_cast<const float *>(ctx.inputs[4]);
    const float one = *static_cast<const float *>(ctx.inputs[5]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    ThreadPool &pool = ThreadPool::get();
    const uint32_t tasks = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(rows, pool.get_num_threads() * 2)));
    const uint64_t per = (rows + tasks - 1) / tasks;
    pool.parallel_for(tasks, [&](uint32_t t) {
        const uint64_t begin = static_cast<uint64_t>(t) * per;
        const uint64_t end = std::min(rows, begin + per);
        if (begin < end)
            gemmaRmsNormFillRowsNeon(x, w, n, eps, half, one, out, D, begin, end);
    });
}

REGISTER_KERNEL("GemmaRMSNormFill_Heads_F32_3D_Threaded", 6, 6, matchGemmaRmsNormFill, runGemmaRMSNormFillHeadsF323DThreaded, refFactoryGemmaRmsNormFillHeads, {}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
                {{4, 8, 64}, {64}, {1}, {1}, {1}, {1}}, {true, true, true, true, true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif
