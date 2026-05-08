#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <thread>
#include <vector>
#include <algorithm>

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
    if (inputs.size() != 2)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Shapes must match exactly for element-wise multiplication
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;

    // Enforce contiguity for this specific optimized kernel
    return isContiguous(output);
}

inline void runMulF32_ND_NEON(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *a_base = static_cast<const float *>(inputs[0]);
    const float *b_base = static_cast<const float *>(inputs[1]);
    float *out_base = static_cast<float *>(outputs[0]);

    const uint64_t n = countElements(inViews[0].getShape());
    if (n == 0)
        return;

    // Determine thread count
    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    uint64_t chunk_size = (n + num_threads - 1) / num_threads;

    auto worker = [=](uint64_t start, uint64_t end)
    {
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
    };

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
        th.join();
}

inline uint32_t refFactoryMulND_NEON(const std::vector<uint32_t> &inputs, Graph &graph)
{
    // This allows the e-graph to identify this kernel as a valid implementation for MUL
    return graph.mul(inputs[0], inputs[1]);
}

REGISTER_KERNEL(
    "Mul_ND_NEON",
    2,
    matchMulF32_ND_NEON,
    runMulF32_ND_NEON,
    refFactoryMulND_NEON,
    {Backend::CPU},
    {DType::FLOAT32, DType::FLOAT32},
    {{1, 32, 512, 512}, {1, 32, 512, 512}}, // Target typical bottleneck shapes
    {true, true},
    {{Backend::CPU}, {Backend::CPU}});

#endif // TG_HAS_NEON