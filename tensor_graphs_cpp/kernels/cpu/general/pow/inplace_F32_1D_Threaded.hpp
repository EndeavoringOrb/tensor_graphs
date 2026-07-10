#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>

/**
 * KERNEL: Pow_1D_inplace_Threaded
 * 
 * High-performance, in-place power kernel optimized for 1D float32 tensors.
 * Performs element-wise base[i] = pow(base[i], exponent[i]) across multiple CPU threads.
 * Designed with no fast paths, ensuring a uniform execution path for all exponent values.
 */

inline bool matchPowF32_1D_Inplace_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Validate that inputs and output have 1D shapes
    if (inputs[0].getShape().size() != 1 || inputs[1].getShape().size() != 1 || output.getShape().size() != 1)
        return false;

    // Enforce that shapes match exactly
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;

    return isContiguous(output);
}

inline void runPowF32_1D_Inplace_Threaded(const KernelContext &ctx)
{
    float *base_out = static_cast<float *>(ctx.outputs[0]);
    const float *exponent = static_cast<const float *>(ctx.inputs[1]);
    const uint64_t n = countElements(ctx.inViews[0].getShape());
    if (n == 0)
        return;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    // Skip thread overhead for small tensors
    if (n < 16384)
    {
        num_threads = 1;
    }

    uint64_t chunk_size = (n + num_threads - 1) / num_threads;

    auto worker = [=](uint64_t start, uint64_t end)
    {
        for (uint64_t i = start; i < end; ++i)
        {
            // Uniformly call std::pow with no fast paths for 0.5f, 2.0f, etc.
            base_out[i] = std::pow(base_out[i], exponent[i]);
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

inline uint32_t refFactoryPow1D_Inplace_Threaded(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.pow(inputs[0], inputs[1]);
}

REGISTER_KERNEL_INPLACE(
    "Pow_1D_inplace_Threaded",
    2,
    matchPowF32_1D_Inplace_Threaded,
    runPowF32_1D_Inplace_Threaded,
    refFactoryPow1D_Inplace_Threaded,
    {Backend::CPU},
    {DType::FLOAT32, DType::FLOAT32},
    {{2048}, {2048}},
    {true, true},
    {{Backend::CPU}, {Backend::CPU}}
);