#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>

/**
 * KERNEL: Div_1D_NEON_inplace_Threaded
 *
 * In-place element-wise division kernel optimized for 1D float32 tensors.
 * Uses ARM NEON vector instructions and parallelizes over available hardware threads.
 */

inline bool matchDivF32_1D_NEON_Inplace_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Validate rank and shapes to ensure structural compatibility
    if (inputs[0].getShape().size() != 1 || inputs[1].getShape().size() != 1 || output.getShape().size() != 1)
        return false;

    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;

    return isContiguous(output);
}

inline void runDivF32_1D_NEON_Inplace_Threaded(const KernelContext &ctx)
{
    float *out = static_cast<float *>(ctx.outputs[0]);
    const float *b = static_cast<const float *>(ctx.inputs[1]);
    uint64_t n = countElements(ctx.inViews[0].getShape());
    if (n == 0)
        return;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    // Use a single thread for small workloads to avoid thread creation overhead
    if (n < 16384)
    {
        num_threads = 1;
    }

    uint64_t chunk_size = (n + num_threads - 1) / num_threads;

    auto worker = [=](uint64_t start, uint64_t end)
    {
        uint64_t i = start;
        // Vectorized loop: Process 4 elements at a time
        for (; i + 4 <= end; i += 4)
        {
            float32x4_t va = vld1q_f32(out + i);
            float32x4_t vb = vld1q_f32(b + i);
            vst1q_f32(out + i, vdivq_f32(va, vb));
        }
        // Scalar cleanup loop
        for (; i < end; ++i)
        {
            out[i] /= b[i];
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

inline uint32_t refFactoryDiv1D_NEON_Inplace_Threaded(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.div(inputs[0], inputs[1]);
}

REGISTER_KERNEL_INPLACE(
    "Div_1D_NEON_inplace_Threaded",
    2,
    matchDivF32_1D_NEON_Inplace_Threaded,
    runDivF32_1D_NEON_Inplace_Threaded,
    refFactoryDiv1D_NEON_Inplace_Threaded,
    {Backend::CPU},
    {DType::FLOAT32, DType::FLOAT32},
    {{2048}, {2048}},
    {true, true},
    {{Backend::CPU}, {Backend::CPU}});

#endif // TG_HAS_NEON