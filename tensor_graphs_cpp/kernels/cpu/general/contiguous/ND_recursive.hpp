#pragma once

#include <algorithm>
#include <cstring>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

/**
 * KERNEL: RecursiveContiguous_ND
 *
 * This kernel transforms a tensor with arbitrary strides into a contiguous
 * layout.
 *
 * Optimized Implementation Details:
 * 1. Recursive Descent: Replaced the "odometer" coordinate system with
 * recursive pointer arithmetic. This eliminates O(Rank) multiplications per
 * block.
 * 2. Allocation-Free: Removed std::vector<uint32_t> coords from the hot path.
 * 3. Small-Block Optimization: Uses direct assignment instead of std::memcpy
 *    when block_size == 1, bypassing memcpy's call overhead.
 * 4. Parallelization: Utilizes ThreadPool to parallelize the outermost dimension
 *    for large tensors.
 * 5. Contiguity Analysis: Identifies the largest contiguous suffix to maximize
 *    the size of memcpy calls.
 */

namespace detail
{

// Recursive helper to traverse the tensor dimensions and copy data
// shape is vector<uint32_t>, strides is vector<uint64_t> to match types.hpp
template <typename T>
void copy_recursive_fast(int dim, const T *src, T *&dst, const std::vector<uint32_t> &shape,
                         const std::vector<uint64_t> &strides, int outer_rank, uint64_t block_size)
{
    // Base Case: We have reached the contiguous block
    if (dim == outer_rank)
    {
        if (block_size == 1)
        {
            *dst = *src;
        }
        else
        {
            std::memcpy(dst, src, block_size * sizeof(T));
        }
        dst += block_size;
        return;
    }

    const uint32_t dim_size = shape[dim];
    const uint64_t dim_stride = strides[dim];

    for (uint32_t i = 0; i < dim_size; ++i)
    {
        // Advance src by the stride of the current dimension
        // dst is passed by reference and advanced linearly by the base case
        copy_recursive_fast<T>(dim + 1, src + (i * dim_stride), dst, shape, strides, outer_rank, block_size);
    }
}
} // namespace detail

inline bool matchRecursiveContiguous_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &in = inputs[0];

    if (in.dtype != output.dtype)
        return false;
    if (in.getShape() != output.getShape())
        return false;

    // The output of a contiguous operation must be contiguous
    if (!isContiguous(output))
        return false;

    return true;
}

inline void runRecursiveContiguous_ND(const KernelContext &ctx)
{
    const auto &view = ctx.inViews[0];
    const auto &shape = view.getShape();
    const auto &strides = view.strides;

    const uint8_t *src_base = static_cast<const uint8_t *>(ctx.inputs[0]);
    uint8_t *dst_base = static_cast<uint8_t *>(ctx.outputs[0]);
    const uint64_t elementSize = getDTypeSize(view.dtype);

    if (shape.empty())
    {
        if (countElements(shape) == 1)
        {
            std::memcpy(dst_base, src_base, elementSize);
        }
        return;
    }

    // --- Step 1: Contiguity Analysis ---
    int rank = static_cast<int>(shape.size());
    int contig_dim_start = rank;
    uint64_t contig_elements = 1;

    for (int d = rank - 1; d >= 0; --d)
    {
        if (d == rank - 1)
        {
            if (strides[d] == 1)
            {
                contig_dim_start = d;
                contig_elements = shape[d];
            }
            else
                break;
        }
        else
        {
            // Match the logic: current stride == next_stride * next_dim_size
            if (strides[d] == static_cast<uint64_t>(strides[d + 1]) * shape[d + 1])
            {
                contig_dim_start = d;
                contig_elements *= shape[d];
            }
            else
                break;
        }
    }

    // Case A: The entire tensor is already contiguous.
    if (contig_dim_start == 0)
    {
        std::memcpy(dst_base, src_base, countElements(shape) * elementSize);
        return;
    }

    // Case B: Partially contiguous or totally strided.
    int outer_rank = contig_dim_start;
    uint64_t block_size = (outer_rank == rank) ? 1 : contig_elements;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    if (elementSize == 4)
    {
        uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(dst_base);
        const uint32_t *src_ptr = reinterpret_cast<const uint32_t *>(src_base);
        if (outer_rank > 0 && shape[0] > 1)
        {
            const uint32_t dim0_size = shape[0];
            const uint64_t dim0_stride = strides[0];
            const uint64_t inner_elements = countElements(shape) / dim0_size;
            ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
                uint32_t chunk = (dim0_size + num_threads - 1) / num_threads;
                uint32_t start_i = t * chunk;
                uint32_t end_i = std::min(start_i + chunk, dim0_size);
                for (uint32_t i = start_i; i < end_i; ++i)
                {
                    uint32_t *local_dst = dst_ptr + (i * inner_elements);
                    const uint32_t *local_src = src_ptr + (i * dim0_stride);
                    detail::copy_recursive_fast<uint32_t>(1, local_src, local_dst, shape, strides, outer_rank,
                                                          block_size);
                }
            });
        }
        else
        {
            uint32_t *temp_dst = dst_ptr;
            detail::copy_recursive_fast<uint32_t>(0, src_ptr, temp_dst, shape, strides, outer_rank, block_size);
        }
    }
    else if (elementSize == 2)
    {
        uint16_t *dst_ptr = reinterpret_cast<uint16_t *>(dst_base);
        const uint16_t *src_ptr = reinterpret_cast<const uint16_t *>(src_base);
        if (outer_rank > 0 && shape[0] > 1)
        {
            const uint32_t dim0_size = shape[0];
            const uint64_t dim0_stride = strides[0];
            const uint64_t inner_elements = countElements(shape) / dim0_size;
            ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
                uint32_t chunk = (dim0_size + num_threads - 1) / num_threads;
                uint32_t start_i = t * chunk;
                uint32_t end_i = std::min(start_i + chunk, dim0_size);
                for (uint32_t i = start_i; i < end_i; ++i)
                {
                    uint16_t *local_dst = dst_ptr + (i * inner_elements);
                    const uint16_t *local_src = src_ptr + (i * dim0_stride);
                    detail::copy_recursive_fast<uint16_t>(1, local_src, local_dst, shape, strides, outer_rank,
                                                          block_size);
                }
            });
        }
        else
        {
            uint16_t *temp_dst = dst_ptr;
            detail::copy_recursive_fast<uint16_t>(0, src_ptr, temp_dst, shape, strides, outer_rank, block_size);
        }
    }
    else if (elementSize == 8)
    {
        uint64_t *dst_ptr = reinterpret_cast<uint64_t *>(dst_base);
        const uint64_t *src_ptr = reinterpret_cast<const uint64_t *>(src_base);
        if (outer_rank > 0 && shape[0] > 1)
        {
            const uint32_t dim0_size = shape[0];
            const uint64_t dim0_stride = strides[0];
            const uint64_t inner_elements = countElements(shape) / dim0_size;
            ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
                uint32_t chunk = (dim0_size + num_threads - 1) / num_threads;
                uint32_t start_i = t * chunk;
                uint32_t end_i = std::min(start_i + chunk, dim0_size);
                for (uint32_t i = start_i; i < end_i; ++i)
                {
                    uint64_t *local_dst = dst_ptr + (i * inner_elements);
                    const uint64_t *local_src = src_ptr + (i * dim0_stride);
                    detail::copy_recursive_fast<uint64_t>(1, local_src, local_dst, shape, strides, outer_rank,
                                                          block_size);
                }
            });
        }
        else
        {
            uint64_t *temp_dst = dst_ptr;
            detail::copy_recursive_fast<uint64_t>(0, src_ptr, temp_dst, shape, strides, outer_rank, block_size);
        }
    }
    else
    {
        uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(dst_base);
        const uint8_t *src_ptr = reinterpret_cast<const uint8_t *>(src_base);
        if (outer_rank > 0 && shape[0] > 1)
        {
            const uint32_t dim0_size = shape[0];
            const uint64_t dim0_stride = strides[0];
            const uint64_t inner_elements = countElements(shape) / dim0_size;
            ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
                uint32_t chunk = (dim0_size + num_threads - 1) / num_threads;
                uint32_t start_i = t * chunk;
                uint32_t end_i = std::min(start_i + chunk, dim0_size);
                for (uint32_t i = start_i; i < end_i; ++i)
                {
                    uint8_t *local_dst = dst_ptr + (i * inner_elements);
                    const uint8_t *local_src = src_ptr + (i * dim0_stride);
                    detail::copy_recursive_fast<uint8_t>(1, local_src, local_dst, shape, strides, outer_rank,
                                                         block_size);
                }
            });
        }
        else
        {
            uint8_t *temp_dst = dst_ptr;
            detail::copy_recursive_fast<uint8_t>(0, src_ptr, temp_dst, shape, strides, outer_rank, block_size);
        }
    }
}

inline LogicalId refFactoryRecursiveContiguous_ND(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 1)
        Error::throw_err("RecursiveContiguous requires exactly 1 input");

    return graph.contiguous(inputs[0]);
}

REGISTER_KERNEL("RecursiveContiguous_ND", 1, 1, matchRecursiveContiguous_ND, runRecursiveContiguous_ND,
                refFactoryRecursiveContiguous_ND, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::ANY}, {{8, 32}}, {false}, {{MemSpace(1, HandleType::CPP)}});
