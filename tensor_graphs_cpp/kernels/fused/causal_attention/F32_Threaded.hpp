// Causal attention (diagonal offset input): (head, query row) tasks split
// across the thread pool.
#pragma once

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "kernels/fused/causal_attention/ref.hpp"

inline void runCausalAttentionF32Threaded(const KernelContext &ctx)
{
    const uint32_t total = causalAttentionTaskCount(ctx);
    const uint32_t tasks = std::min<uint32_t>(total, ThreadPool::get().get_num_threads());
    ThreadPool::get().parallel_for(tasks, [&](uint32_t t) {
        causalAttentionTasks(ctx, static_cast<uint32_t>(static_cast<uint64_t>(total) * t / tasks),
                             static_cast<uint32_t>(static_cast<uint64_t>(total) * (t + 1) / tasks));
    });
}

REGISTER_KERNEL("Causal_Attention_F32_Threaded", 4, 4, matchCausalAttention, runCausalAttentionF32Threaded,
                refFactoryCausalAttention, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::INT32},
                {{4, 16, 256}, {4, 16, 256}, {4, 16, 256}, {1}}, {true, true, true, true},
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}});

static const FusedRowRestrictionRegistrar kCausalAttentionF32ThreadedRowRestriction("Causal_Attention_F32_Threaded",
                                                                                   causalAttentionRowRestriction());
