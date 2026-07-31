#pragma once
#include <algorithm>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

/**
 * Highly optimized multi-threaded cache-blocked 3D Transposition / Contiguous
 * kernel. Replaces the slow, recursive fallback for [B, M, N] transposed
 * strides.
 */

inline bool matchContiguousTransposed3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || output.getShape().size() != 3)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (output.dtype != DType::FLOAT32)
        return false;
    if (!isContiguous(output))
        return false;

    const auto &shape = inputs[0].getShape();
    const auto &strides = inputs[0].strides;
    uint64_t B = shape[0];
    uint64_t M = shape[1];
    uint64_t N = shape[2];

    // Verify the input has transposed strides [M * N, 1, M]
    if (strides[0] != M * N)
        return false;
    if (strides[1] != 1)
        return false;
    if (strides[2] != M)
        return false;

    return true;
}

inline void runContiguousTransposed3D(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &view = ctx.inViews[0];
    const auto &shape = view.getShape();

    uint32_t B = shape[0];
    uint32_t M = shape[1];
    uint32_t N = shape[2];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    if (num_threads > B)
        num_threads = B;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t b_per_thread = (B + num_threads - 1) / num_threads;
        uint32_t b_start = t * b_per_thread;
        uint32_t b_end = std::min(b_start + b_per_thread, B);

        // 32x32 block tile size keeps memory chunks entirely within L1/L2 caches
        constexpr uint32_t BLOCK = 32;

        for (uint32_t b = b_start; b < b_end; ++b)
        {
            uint64_t batch_offset = (uint64_t)b * M * N;
            const float *in_batch = in + batch_offset;
            float *out_batch = out + batch_offset;

            for (uint32_t m_outer = 0; m_outer < M; m_outer += BLOCK)
            {
                uint32_t m_end = std::min(m_outer + BLOCK, M);
                for (uint32_t n_outer = 0; n_outer < N; n_outer += BLOCK)
                {
                    uint32_t n_end = std::min(n_outer + BLOCK, N);
                    for (uint32_t m = m_outer; m < m_end; ++m)
                    {
                        uint32_t n = n_outer;
                        for (; n < n_end; ++n)
                        {
                            out_batch[m * N + n] = in_batch[n * M + m];
                        }
                    }
                }
            }
        }
    });
}

inline LogicalId refFactoryContiguousTransposed3D(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.contiguous(inputs[0]);
}

REGISTER_KERNEL("Contiguous_Transposed_3D", 1, 1, matchContiguousTransposed3D, runContiguousTransposed3D,
                refFactoryContiguousTransposed3D, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32}, {{256, 2048, 1024}}, {false}, {{MemSpace(1, HandleType::CPP)}});