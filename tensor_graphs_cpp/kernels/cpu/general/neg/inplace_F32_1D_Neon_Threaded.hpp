#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>

/**
 * KERNEL: Neg_1D_NEON_inplace_Threaded
 *
 * High-performance, in-place negation kernel optimized for 1D float32 tensors.
 * Leverages ARM NEON SIMD and multi-threading for maximal throughput on high-core-count processors.
 */

inline bool matchNegF32_1D_NEON_Inplace_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // The engine's registration macro validates input count, dtypes, and backends.
    // We only verify that this is a 1D tensor mapping and output remains contiguous.
    if (inputs[0].getShape().size() != 1 || output.getShape().size() != 1)
        return false;

    if (inputs[0].getShape() != output.getShape())
        return false;

    return isContiguous(output);
}

inline void runNegF32_1D_NEON_Inplace_Threaded(const KernelContext &ctx)
{
    float *out = static_cast<float *>(ctx.outputs[0]);
    const uint64_t n = countElements(ctx.inViews[0].getShape());
    if (n == 0)
        return;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    // Fast-path: Skip thread spawning entirely for small workloads to eliminate overhead
    if (n < 16384)
    {
        num_threads = 1;
    }

    uint64_t chunk_size = (n + num_threads - 1) / num_threads;

    auto worker = [=](uint64_t start, uint64_t end)
    {
        uint64_t i = start;
        // Vector Loop: Process 4 elements at a time
        for (; i + 4 <= end; i += 4)
        {
            float32x4_t vx = vld1q_f32(out + i);
            vst1q_f32(out + i, vnegq_f32(vx));
        }
        // Tail Loop: Handle remaining elements
        for (; i < end; ++i)
        {
            out[i] = -out[i];
        }
    };

    if (num_threads == 1)
    {
        worker(0, n);
    }
    else
    {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (uint32_t t = 0; t < num_threads; ++t)
        {
            uint64_t start = t * chunk_size;
            uint64_t end = std::min(start + chunk_size, n);
            if (start < end)
            {
                threads.emplace_back(worker, start, end);
            }
        }
        for (auto &th : threads)
        {
            th.join();
        }
    }
}

inline uint32_t refFactoryNeg1D_NEON_Inplace_Threaded(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.neg(inputs[0]);
}

REGISTER_KERNEL_INPLACE(
    "Neg_1D_NEON_inplace_Threaded",
    1,
    matchNegF32_1D_NEON_Inplace_Threaded,
    runNegF32_1D_NEON_Inplace_Threaded,
    refFactoryNeg1D_NEON_Inplace_Threaded,
    {Backend::CPU},
    {DType::FLOAT32},
    {{2048}},
    {true},
    {{Backend::CPU}});

#endif // TG_HAS_NEON