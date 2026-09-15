#pragma once
#include <algorithm>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchContiguousTransposed4D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 4 || output.getShape().size() != 4)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;

    const auto &shape = inputs[0].getShape();
    const auto &strides = inputs[0].strides;

    // Check for permute(0, 2, 1, 3): shape [B, H, S, D], input strides [H*S*D, D, H*D, 1]
    uint64_t B = shape[0];
    uint64_t H = shape[1];
    uint64_t S = shape[2];
    uint64_t D = shape[3];

    if (strides[3] == 1 && strides[1] == D && strides[2] == H * D)
        return true;

    // Check for permute(0, 1, 3, 2): shape [B, H, D, S], input strides [H*S*D, S*D, 1, D]
    if (strides[2] == 1 && strides[3] == D && strides[1] == S * D)
        return true;

    return false;
}

inline void runContiguousTransposed4D(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &shape = ctx.inViews[0].getShape();
    const auto &strides = ctx.inViews[0].strides;

    uint32_t B = shape[0];
    uint32_t H = shape[1];
    uint32_t S = shape[2];
    uint32_t D = shape[3];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    bool is_0213 = (strides[3] == 1 && strides[1] == D && strides[2] == H * D);

    if (is_0213)
    {
        // Out layout: [B, H, S, D] row-major
        // In layout: [B, S, H, D] row-major
        uint32_t total_tasks = B * H;
        num_threads = std::min(num_threads, total_tasks);

        ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
            uint32_t chunk = (total_tasks + num_threads - 1) / num_threads;
            uint32_t start_t = t * chunk;
            uint32_t end_t = std::min(start_t + chunk, total_tasks);

            for (uint32_t task = start_t; task < end_t; ++task)
            {
                uint32_t b = task / H;
                uint32_t h = task % H;

                for (uint32_t s = 0; s < S; ++s)
                {
                    const float *src = in + (static_cast<uint64_t>(b) * S * H + static_cast<uint64_t>(s) * H + h) * D;
                    float *dst = out + (static_cast<uint64_t>(b) * H * S + static_cast<uint64_t>(h) * S + s) * D;
                    std::memcpy(dst, src, D * sizeof(float));
                }
            }
        });
    }
    else
    {
        // Out layout: [B, H, S, D]
        // In layout: [B, H, D, S]
        uint32_t total_tasks = B * H;
        num_threads = std::min(num_threads, total_tasks);

        ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
            uint32_t chunk = (total_tasks + num_threads - 1) / num_threads;
            uint32_t start_t = t * chunk;
            uint32_t end_t = std::min(start_t + chunk, total_tasks);

            constexpr uint32_t BLOCK = 32;

            for (uint32_t task = start_t; task < end_t; ++task)
            {
                uint32_t b = task / H;
                uint32_t h = task % H;

                const float *src_head = in + (static_cast<uint64_t>(b) * H + h) * S * D;
                float *dst_head = out + (static_cast<uint64_t>(b) * H + h) * S * D;

                for (uint32_t s_outer = 0; s_outer < S; s_outer += BLOCK)
                {
                    uint32_t s_end = std::min(s_outer + BLOCK, S);
                    for (uint32_t d_outer = 0; d_outer < D; d_outer += BLOCK)
                    {
                        uint32_t d_end = std::min(d_outer + BLOCK, D);
                        for (uint32_t s = s_outer; s < s_end; ++s)
                        {
                            for (uint32_t d = d_outer; d < d_end; ++d)
                            {
                                dst_head[s * D + d] = src_head[d * S + s];
                            }
                        }
                    }
                }
            }
        });
    }
}

inline LogicalId refFactoryContiguousTransposed4D(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.contiguous(inputs[0]);
}

REGISTER_KERNEL("Contiguous_Transposed_4D", 1, 1, matchContiguousTransposed4D, runContiguousTransposed4D,
                refFactoryContiguousTransposed4D, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32}, {{1, 48, 4224, 128}}, {false}, {{MemSpace(1, HandleType::CPP)}});