#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

#include <algorithm>
#include <vector>

#include "core/common/thread_pool.hpp"

/**
 * KERNEL: Mul_ND_NEON
 *
 * Optimized F32 multiplication for contiguous ND tensors.
 * Features:
 * 1. ARM NEON SIMD: Processes 4 elements per instruction using vmulq_f32.
 * 2. Multi-threading: Scales across all available CPU cores.
 */

inline bool matchMulF32_ND_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Shapes must match exactly for element-wise multiplication
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;

    // Enforce contiguity for this specific optimized kernel
    return isContiguous(output);
}

inline void runMulF32_ND_NEON(const KernelContext &ctx)
{
    const float *a_base = static_cast<const float *>(ctx.inputs[0]);
    const float *b_base = static_cast<const float *>(ctx.inputs[1]);
    float *out_base = static_cast<float *>(ctx.outputs[0]);

    const uint64_t n = countElements(ctx.inViews[0].getShape());
    if (n == 0)
        return;

    // Determine thread count
    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint64_t chunk_size = (n + num_threads - 1) / num_threads;
        uint64_t start = t * chunk_size;
        uint64_t end = std::min(start + chunk_size, n);

        uint64_t i = start;
        // 1. NEON SIMD Loop (4 elements per step)
        for (; i + 4 <= end; i += 4)
        {
            float32x4_t va = vld1q_f32(a_base + i);
            float32x4_t vb = vld1q_f32(b_base + i);
            vst1q_f32(out_base + i, vmulq_f32(va, vb));
        }
        // 2. Scalar Tail Loop
        for (; i < end; ++i)
        {
            out_base[i] = a_base[i] * b_base[i];
        }
    });
}

inline LogicalId refFactoryMulND_NEON(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // This allows the e-graph to identify this kernel as a valid implementation
    // for MUL
    return graph.mul(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Mul_ND_NEON", 2, 2, matchMulF32_ND_NEON, runMulF32_ND_NEON, refFactoryMulND_NEON, {0, 1},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 32, 512, 512}, {1, 32, 512, 512}}, // Target typical bottleneck shapes
                {true, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON