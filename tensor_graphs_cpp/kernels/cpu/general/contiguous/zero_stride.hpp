#pragma once

#include <cstring>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

/**
 * KERNEL: ZeroStrideBroadcast_ND
 *
 * Highly optimized contiguous kernel for extreme edge cases where the input
 * tensor is effectively a broadcast (e.g. strides = [1, 0, 0]).
 * Optimized specifically for multi-core architecture using cache-line sized NEON vector unrolling and ThreadPool.
 */

inline bool matchZeroStrideBroadcast_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &in = inputs[0];

    if (in.dtype != output.dtype)
        return false;
    if (in.getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;

    const auto &strides = in.strides;
    if (strides.empty())
        return false;

    // Strict Match: Only triggers if the first stride is 1 and ALL subsequent
    // strides are 0.
    if (strides[0] != 1)
        return false;
    for (uint64_t i = 1; i < strides.size(); ++i)
    {
        if (strides[i] != 0)
            return false;
    }

    return true;
}

inline void runZeroStrideBroadcast_ND(const KernelContext &ctx)
{
    const uint8_t *src_base = static_cast<const uint8_t *>(ctx.inputs[0]);
    uint8_t *dst_base = static_cast<uint8_t *>(ctx.outputs[0]);

    const auto &view = ctx.inViews[0];
    const auto &shape = view.getShape();
    const uint64_t elementSize = getDTypeSize(view.dtype);

    if (shape.empty())
    {
        std::memcpy(dst_base, src_base, elementSize);
        return;
    }

    // Because strides are [1, 0, 0...], the first dimension reads linearly,
    // but the inner dimensions repeatedly read the EXACT same memory location.
    const uint64_t outer_elements = shape[0];
    uint64_t inner_elements = 1;
    for (uint64_t i = 1; i < shape.size(); ++i)
    {
        inner_elements *= shape[i];
    }

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    if (elementSize == 4)
    {
        const float *src_f32 = reinterpret_cast<const float *>(src_base);
        float *dst_f32 = reinterpret_cast<float *>(dst_base);

        for (uint64_t i = 0; i < outer_elements; ++i)
        {
            float val = src_f32[i];
            float *current_dst = dst_f32 + (i * inner_elements);

#if defined(__aarch64__) || defined(_M_ARM64)
            int64_t total_chunks = inner_elements / 16;
            ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
                int64_t chunk_size = (total_chunks + num_threads - 1) / num_threads;
                int64_t start_j = t * chunk_size;
                int64_t end_j = std::min(start_j + chunk_size, total_chunks);
                float32x4_t val_vec = vdupq_n_f32(val);
                for (int64_t j = start_j; j < end_j; ++j)
                {
                    int64_t offset = j * 16;
                    vst1q_f32(current_dst + offset, val_vec);
                    vst1q_f32(current_dst + offset + 4, val_vec);
                    vst1q_f32(current_dst + offset + 8, val_vec);
                    vst1q_f32(current_dst + offset + 12, val_vec);
                }
            });
            // Tail cleanup (single-threaded, mostly negligible size)
            for (int64_t j = (inner_elements / 16) * 16; j < (int64_t)inner_elements; ++j)
            {
                current_dst[j] = val;
            }
#else
            ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
                int64_t chunk_size = (inner_elements + num_threads - 1) / num_threads;
                int64_t start_j = t * chunk_size;
                int64_t end_j = std::min(start_j + chunk_size, (int64_t)inner_elements);
                for (int64_t j = start_j; j < end_j; ++j)
                {
                    current_dst[j] = val;
                }
            });
#endif
        }
    }
    else if (elementSize == 2)
    {
        const uint16_t *src_u16 = reinterpret_cast<const uint16_t *>(src_base);
        uint16_t *dst_u16 = reinterpret_cast<uint16_t *>(dst_base);

        for (uint64_t i = 0; i < outer_elements; ++i)
        {
            uint16_t val = src_u16[i];
            uint16_t *current_dst = dst_u16 + (i * inner_elements);

#if defined(__aarch64__) || defined(_M_ARM64)
            int64_t total_chunks = inner_elements / 32;
            ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
                int64_t chunk_size = (total_chunks + num_threads - 1) / num_threads;
                int64_t start_j = t * chunk_size;
                int64_t end_j = std::min(start_j + chunk_size, total_chunks);
                uint16x8_t val_vec = vdupq_n_u16(val);
                for (int64_t j = start_j; j < end_j; ++j)
                {
                    int64_t offset = j * 32;
                    vst1q_u16(current_dst + offset, val_vec);
                    vst1q_u16(current_dst + offset + 8, val_vec);
                    vst1q_u16(current_dst + offset + 16, val_vec);
                    vst1q_u16(current_dst + offset + 24, val_vec);
                }
            });
            for (int64_t j = (inner_elements / 32) * 32; j < (int64_t)inner_elements; ++j)
            {
                current_dst[j] = val;
            }
#else
            ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
                int64_t chunk_size = (inner_elements + num_threads - 1) / num_threads;
                int64_t start_j = t * chunk_size;
                int64_t end_j = std::min(start_j + chunk_size, (int64_t)inner_elements);
                for (int64_t j = start_j; j < end_j; ++j)
                {
                    current_dst[j] = val;
                }
            });
#endif
        }
    }
    else if (elementSize == 8)
    {
        const uint64_t *src_u64 = reinterpret_cast<const uint64_t *>(src_base);
        uint64_t *dst_u64 = reinterpret_cast<uint64_t *>(dst_base);

        for (uint64_t i = 0; i < outer_elements; ++i)
        {
            uint64_t val = src_u64[i];
            uint64_t *current_dst = dst_u64 + (i * inner_elements);

#if defined(__aarch64__) || defined(_M_ARM64)
            int64_t total_chunks = inner_elements / 8;
            ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
                int64_t chunk_size = (total_chunks + num_threads - 1) / num_threads;
                int64_t start_j = t * chunk_size;
                int64_t end_j = std::min(start_j + chunk_size, total_chunks);
                uint64x2_t val_vec = vdupq_n_u64(val);
                for (int64_t j = start_j; j < end_j; ++j)
                {
                    int64_t offset = j * 8;
                    vst1q_u64(current_dst + offset, val_vec);
                    vst1q_u64(current_dst + offset + 2, val_vec);
                    vst1q_u64(current_dst + offset + 4, val_vec);
                    vst1q_u64(current_dst + offset + 6, val_vec);
                }
            });
            for (int64_t j = (inner_elements / 8) * 8; j < (int64_t)inner_elements; ++j)
            {
                current_dst[j] = val;
            }
#else
            ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
                int64_t chunk_size = (inner_elements + num_threads - 1) / num_threads;
                int64_t start_j = t * chunk_size;
                int64_t end_j = std::min(start_j + chunk_size, (int64_t)inner_elements);
                for (int64_t j = start_j; j < end_j; ++j)
                {
                    current_dst[j] = val;
                }
            });
#endif
        }
    }
    else if (elementSize == 1)
    {
        for (uint64_t i = 0; i < outer_elements; ++i)
        {
            uint8_t val = src_base[i];
            uint8_t *current_dst = dst_base + (i * inner_elements);
            // System memset is generally perfectly optimized for 1 byte copies
            std::memset(current_dst, val, inner_elements);
        }
    }
}

inline LogicalId refFactoryZeroStrideBroadcast_ND(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 1)
        Error::throw_err("ZeroStrideBroadcast requires exactly 1 input");

    TensorNode &node = graph.getNode(inputs[0]);
    if (!node.getShape().empty())
    {
        std::vector<uint64_t> strides(node.getShape().size(), 0);
        strides[0] = 1;
        node.strides = strides; // TODO: add dummy input strides to REGISTER_KERNEL
                                // macro so we don't have to do this hack
    }

    return graph.contiguous(inputs[0]);
}

REGISTER_KERNEL("ZeroStrideBroadcast_ND", 1, 1, matchZeroStrideBroadcast_ND, runZeroStrideBroadcast_ND,
                refFactoryZeroStrideBroadcast_ND, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::ANY}, {{8, 32}}, {false}, {{MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON