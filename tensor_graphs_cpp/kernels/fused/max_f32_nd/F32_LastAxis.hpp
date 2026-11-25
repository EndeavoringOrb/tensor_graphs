// Max over the last axis, single thread. See ref.hpp.
#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/reduce_last_axis/ref.hpp"
#if defined(TG_HAS_NEON)

inline void runMaxF32LastAxisNEON(const KernelContext &ctx)
{
    checkReduceAxis(ctx);
    const ReduceRows rows(ctx.inViews[0]);
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    maxLastAxisRows(in, out, rows, 0, rows.rows);
}

REGISTER_KERNEL("Max_F32_LastAxis_NEON", 2, 2, matchReduceLastAxis, runMaxF32LastAxisNEON, refFactoryMaxF32_ND_CUDA, {}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::INT32}, {{4, 8, 64}, {1}}, {false, false},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif
