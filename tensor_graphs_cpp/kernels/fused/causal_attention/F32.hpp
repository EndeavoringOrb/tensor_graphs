// Causal attention (diagonal offset input), single thread. Each query row
// reads only the keys left of the diagonal; the mask is never materialized.
#pragma once

#include "core/kernels.hpp"
#include "kernels/fused/causal_attention/ref.hpp"

inline void runCausalAttentionF32(const KernelContext &ctx)
{
    causalAttentionTasks(ctx, 0, causalAttentionTaskCount(ctx));
}

REGISTER_KERNEL("Causal_Attention_F32", 4, 4, matchCausalAttention, runCausalAttentionF32,
                refFactoryCausalAttention, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::INT32},
                {{4, 16, 256}, {4, 16, 256}, {4, 16, 256}, {1}}, {true, true, true, true},
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}});

static const FusedRowRestrictionRegistrar kCausalAttentionF32RowRestriction("Causal_Attention_F32",
                                                                           causalAttentionRowRestriction());
