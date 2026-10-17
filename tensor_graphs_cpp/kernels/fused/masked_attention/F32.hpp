// Masked attention (additive mask input), single thread. Keys masked with
// NEG_INF are skipped, so causal and windowed masks cost only their visible keys.
#pragma once

#include "core/kernels.hpp"
#include "kernels/fused/masked_attention/ref.hpp"

inline void runMaskedAttentionF32(const KernelContext &ctx)
{
    maskedAttentionTasks(ctx, 0, maskedAttentionTaskCount(ctx));
}

REGISTER_KERNEL("Masked_Attention_F32", 4, 4, matchMaskedAttention, runMaskedAttentionF32,
                refFactoryMaskedAttention, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
                {{4, 16, 256}, {4, 16, 256}, {4, 16, 256}, {1, 16, 16}}, {true, true, true, true},
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}});

static const FusedRowRestrictionRegistrar kMaskedAttentionF32RowRestriction("Masked_Attention_F32",
                                                                           maskedAttentionRowRestriction());
