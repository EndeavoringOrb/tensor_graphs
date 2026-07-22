#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <thread>
#include <vector>
#include <algorithm>

// =============================================================================
// FUSED KERNEL: Negate F32 ND (NEON + Multi-threaded)
//
// Replaces the reference NEGATE kernel which uses getStridedIndex per element.
// For contiguous tensors (the common case), this uses NEON vnegq_f32 and
// multi-threading for near-linear scaling across all 12 cores.
//
// Expected savings: ~580ms remaining after softmax fusion (RoPE neg patterns)
// But also critical for enabling correct fusion of other subgraphs containing neg.
// =============================================================================

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchNegF32_ND_NEON_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    // Only handle contiguous for NEON path
    return isContiguous(output);
}

inline void runNegF32_ND_NEON_Threaded(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const uint64_t n = countElements(ctx.inViews[0].getShape());
    if (n == 0)
        return;

    const uint32_t num_threads = std::max(1u, std::thread::hardware_concurrency());
    const uint64_t chunk = (n + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            const uint64_t start = t * chunk;
            const uint64_t end = std::min(start + chunk, n);
            uint64_t i = start;

            // NEON path: negate 4 floats at a time
            for (; i + 4 <= end; i += 4)
            {
                float32x4_t vx = vld1q_f32(x + i);
                vst1q_f32(out + i, vnegq_f32(vx));
            }
            // Scalar tail
            for (; i < end; ++i)
            {
                out[i] = -x[i];
            } });
    }
    for (auto &w : workers)
        w.join();
}

// Reference factory: same as the reference negate - just graph.neg(x)
inline LogicalId refFactoryNegND_NEON_Threaded(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.neg(inputs[0]);
}

REGISTER_KERNEL("Neg_F32_ND_NEON_Threaded", 1, 1, matchNegF32_ND_NEON_Threaded, runNegF32_ND_NEON_Threaded, refFactoryNegND_NEON_Threaded, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1536}}, {true}, {{MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON