#pragma once

#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>
#include <vector>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * KERNEL: ZeroStrideBroadcast_ND
 *
 * Highly optimized contiguous kernel for extreme edge cases where the input
 * tensor is effectively a broadcast (e.g. strides = [1, 0, 0]).
 * Optimized specifically for a 12-core ARM64 architecture (Snapdragon)
 * using cache-line sized NEON vector unrolling and OpenMP parallelization.
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

    // Strict Match: Only triggers if the first stride is 1 and ALL subsequent strides are 0.
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

    if (elementSize == 4)
    {
        const float *src_f32 = reinterpret_cast<const float *>(src_base);
        float *dst_f32 = reinterpret_cast<float *>(dst_base);

        for (uint64_t i = 0; i < outer_elements; ++i)
        {
            float val = src_f32[i];
            float *current_dst = dst_f32 + (i * inner_elements);

#if defined(__aarch64__) || defined(_M_ARM64)
// Parallelize across 12 ARM cores
#pragma omp parallel
            {
                // Vectorize the broadcast value
                float32x4_t val_vec = vdupq_n_f32(val);

// OpenMP static scheduling evenly chunks the loop to 12 cores
#pragma omp for schedule(static)
                for (int64_t j = 0; j < (int64_t)(inner_elements / 16); ++j)
                {
                    int64_t offset = j * 16;
                    // Unrolled 4x to write exactly one 64-byte ARM cache line per iteration
                    vst1q_f32(current_dst + offset, val_vec);
                    vst1q_f32(current_dst + offset + 4, val_vec);
                    vst1q_f32(current_dst + offset + 8, val_vec);
                    vst1q_f32(current_dst + offset + 12, val_vec);
                }
            }
            // Tail cleanup (single-threaded, mostly negligible size)
            for (int64_t j = (inner_elements / 16) * 16; j < (int64_t)inner_elements; ++j)
            {
                current_dst[j] = val;
            }
#else
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < (int64_t)inner_elements; ++j)
            {
                current_dst[j] = val;
            }
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
#pragma omp parallel
            {
                uint16x8_t val_vec = vdupq_n_u16(val);
#pragma omp for schedule(static)
                for (int64_t j = 0; j < (int64_t)(inner_elements / 32); ++j)
                {
                    int64_t offset = j * 32;
                    vst1q_u16(current_dst + offset, val_vec);
                    vst1q_u16(current_dst + offset + 8, val_vec);
                    vst1q_u16(current_dst + offset + 16, val_vec);
                    vst1q_u16(current_dst + offset + 24, val_vec);
                }
            }
            for (int64_t j = (inner_elements / 32) * 32; j < (int64_t)inner_elements; ++j)
            {
                current_dst[j] = val;
            }
#else
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < (int64_t)inner_elements; ++j)
            {
                current_dst[j] = val;
            }
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
#pragma omp parallel
            {
                uint64x2_t val_vec = vdupq_n_u64(val);
#pragma omp for schedule(static)
                for (int64_t j = 0; j < (int64_t)(inner_elements / 8); ++j)
                {
                    int64_t offset = j * 8;
                    vst1q_u64(current_dst + offset, val_vec);
                    vst1q_u64(current_dst + offset + 2, val_vec);
                    vst1q_u64(current_dst + offset + 4, val_vec);
                    vst1q_u64(current_dst + offset + 6, val_vec);
                }
            }
            for (int64_t j = (inner_elements / 8) * 8; j < (int64_t)inner_elements; ++j)
            {
                current_dst[j] = val;
            }
#else
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < (int64_t)inner_elements; ++j)
            {
                current_dst[j] = val;
            }
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
        node.strides = strides; // TODO: add dummy input strides to REGISTER_KERNEL macro so we don't have to do this hack
    }

    return graph.contiguous(inputs[0]);
}

REGISTER_KERNEL("ZeroStrideBroadcast_ND", 1, 1, matchZeroStrideBroadcast_ND, runZeroStrideBroadcast_ND, refFactoryZeroStrideBroadcast_ND, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
    {DType::ANY},
    {{8, 32}},
    {false},
    {{MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON