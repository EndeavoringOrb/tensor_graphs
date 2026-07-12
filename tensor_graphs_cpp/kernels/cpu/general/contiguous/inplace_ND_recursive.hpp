#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace inplace_recursive_detail
{
    template <typename T>
    void copy_recursive_fast(int dim, const T *src, T *&dst,
                             const std::vector<uint32_t> &shape,
                             const std::vector<uint64_t> &strides,
                             int outer_rank, uint64_t block_size)
    {
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
            copy_recursive_fast<T>(dim + 1, src + (i * dim_stride), dst,
                                   shape, strides, outer_rank, block_size);
        }
    }
}

inline bool matchRecursiveContiguous_ND_Inplace(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;

    // Output layout must be contiguous
    return isContiguous(output);
}

inline void runRecursiveContiguous_ND_Inplace(const KernelContext &ctx)
{
    const auto &view = ctx.inViews[0];
    const auto &shape = view.getShape();
    const auto &strides = view.strides;

    const uint8_t *src_base = static_cast<const uint8_t *>(ctx.inputs[0]);
    uint8_t *dst_base = static_cast<uint8_t *>(ctx.outputs[0]);
    const uint64_t elementSize = getDTypeSize(view.dtype);

    if (shape.empty() || countElements(shape) == 0) return;
    if (src_base != dst_base)
    {
        Error::throw_err("[runRecursiveContiguous_ND_Inplace] src_base != dst_base");
    }

    // Contiguity analysis
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
            if (strides[d] == static_cast<uint64_t>(strides[d + 1]) * shape[d + 1])
            {
                contig_dim_start = d;
                contig_elements *= shape[d];
            }
            else
                break;
        }
    }

    // Already contiguous inside the same buffer
    if (contig_dim_start == 0)
    {
        return;
    }

    // Strided inplace copy requires a shadow buffer to prevent self-overwriting
    uint64_t totalElements = countElements(shape);
    uint64_t totalBytes = totalElements * elementSize;
    std::vector<uint8_t> temp_buf(totalBytes);
    uint8_t *temp_dst_base = temp_buf.data();

    int outer_rank = contig_dim_start;
    uint64_t block_size = (outer_rank == rank) ? 1 : contig_elements;

    if (elementSize == 4)
    {
        uint32_t *dst_ptr = reinterpret_cast<uint32_t *>(temp_dst_base);
        const uint32_t *src_ptr = reinterpret_cast<const uint32_t *>(src_base);
        if (outer_rank > 0 && shape[0] > 1)
        {
            const uint32_t dim0_size = shape[0];
            const uint64_t dim0_stride = strides[0];
            const uint64_t inner_elements = totalElements / dim0_size;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < (int)dim0_size; ++i)
            {
                uint32_t *local_dst = dst_ptr + (i * inner_elements);
                const uint32_t *local_src = src_ptr + (i * dim0_stride);
                inplace_recursive_detail::copy_recursive_fast<uint32_t>(1, local_src, local_dst, shape, strides, outer_rank, block_size);
            }
        }
        else
        {
            uint32_t *temp_dst = dst_ptr;
            inplace_recursive_detail::copy_recursive_fast<uint32_t>(0, src_ptr, temp_dst, shape, strides, outer_rank, block_size);
        }
    }
    else if (elementSize == 2)
    {
        uint16_t *dst_ptr = reinterpret_cast<uint16_t *>(temp_dst_base);
        const uint16_t *src_ptr = reinterpret_cast<const uint16_t *>(src_base);
        if (outer_rank > 0 && shape[0] > 1)
        {
            const uint32_t dim0_size = shape[0];
            const uint64_t dim0_stride = strides[0];
            const uint64_t inner_elements = totalElements / dim0_size;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < (int)dim0_size; ++i)
            {
                uint16_t *local_dst = dst_ptr + (i * inner_elements);
                const uint16_t *local_src = src_ptr + (i * dim0_stride);
                inplace_recursive_detail::copy_recursive_fast<uint16_t>(1, local_src, local_dst, shape, strides, outer_rank, block_size);
            }
        }
        else
        {
            uint16_t *temp_dst = dst_ptr;
            inplace_recursive_detail::copy_recursive_fast<uint16_t>(0, src_ptr, temp_dst, shape, strides, outer_rank, block_size);
        }
    }
    else if (elementSize == 8)
    {
        uint64_t *dst_ptr = reinterpret_cast<uint64_t *>(temp_dst_base);
        const uint64_t *src_ptr = reinterpret_cast<const uint64_t *>(src_base);
        if (outer_rank > 0 && shape[0] > 1)
        {
            const uint32_t dim0_size = shape[0];
            const uint64_t dim0_stride = strides[0];
            const uint64_t inner_elements = totalElements / dim0_size;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < (int)dim0_size; ++i)
            {
                uint64_t *local_dst = dst_ptr + (i * inner_elements);
                const uint64_t *local_src = src_ptr + (i * dim0_stride);
                inplace_recursive_detail::copy_recursive_fast<uint64_t>(1, local_src, local_dst, shape, strides, outer_rank, block_size);
            }
        }
        else
        {
            uint64_t *temp_dst = dst_ptr;
            inplace_recursive_detail::copy_recursive_fast<uint64_t>(0, src_ptr, temp_dst, shape, strides, outer_rank, block_size);
        }
    }
    else
    {
        uint8_t *dst_ptr = temp_dst_base;
        const uint8_t *src_ptr = src_base;
        if (outer_rank > 0 && shape[0] > 1)
        {
            const uint32_t dim0_size = shape[0];
            const uint64_t dim0_stride = strides[0];
            const uint64_t inner_elements = totalElements / dim0_size;
#pragma omp parallel for schedule(static)
            for (int i = 0; i < (int)dim0_size; ++i)
            {
                uint8_t *local_dst = dst_ptr + (i * inner_elements);
                const uint8_t *local_src = src_ptr + (i * dim0_stride);
                inplace_recursive_detail::copy_recursive_fast<uint8_t>(1, local_src, local_dst, shape, strides, outer_rank, block_size);
            }
        }
        else
        {
            uint8_t *temp_dst = dst_ptr;
            inplace_recursive_detail::copy_recursive_fast<uint8_t>(0, src_ptr, temp_dst, shape, strides, outer_rank, block_size);
        }
    }

    std::memcpy(dst_base, temp_dst_base, totalBytes);
}

inline uint32_t refFactoryRecursiveContiguous_ND_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.contiguous(inputs[0]);
}

REGISTER_KERNEL_INPLACE(
    "RecursiveContiguous_ND_inplace",
    1,
    matchRecursiveContiguous_ND_Inplace,
    runRecursiveContiguous_ND_Inplace,
    refFactoryRecursiveContiguous_ND_Inplace,
    {Backend::CPU},
    {DType::ANY},
    {{8, 32}},
    {false},
    {{Backend::CPU}});