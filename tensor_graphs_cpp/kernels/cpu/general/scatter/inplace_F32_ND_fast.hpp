#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * KERNEL: SCATTER_inplace (Fast Version)
 *
 * Optimized Implementation Details:
 * 1. O(1) Coordinate Increment: Uses incremental coordinate updates instead of
 *    O(Rank) division/modulo per element for index unraveling.
 * 2. Contiguity Analysis: Identifies contiguous suffixes in updates to avoid
 *    getStridedIndex calls and enable direct indexing.
 * 3. Pre-computed Multipliers: Output index computed as base_offset + sum(coord * multiplier),
 *    avoiding repeated multiplication of (start + coord * step) * stride.
 * 4. Fast Copy Paths: Uses memcpy for contiguous tensors, recursive copy for
 *    partially contiguous, and incremental copy for fully strided.
 * 5. Parallelization: OpenMP parallelization on outermost dimension for large updates.
 * 6. Small-Block Optimization: Direct assignment for block_size == 1 in copy.
 */

namespace scatter_detail
{

    // Find start dimension of contiguous suffix. Returns rank if no contiguity.
    inline int findContiguousSuffix(const std::vector<uint32_t> &shape, const std::vector<uint64_t> &strides)
    {
        int rank = static_cast<int>(shape.size());
        if (rank == 0)
            return 0;
        if (strides[rank - 1] != 1)
            return rank;

        int contig_start = rank - 1;
        for (int d = rank - 2; d >= 0; --d)
        {
            if (strides[d] == strides[d + 1] * shape[d + 1])
            {
                contig_start = d;
            }
            else
            {
                break;
            }
        }
        return contig_start;
    }

    // Recursive copy for strided source to contiguous destination
    inline void copyStridedToContiguous(int dim, const float *src, float *&dst,
                                        const std::vector<uint32_t> &shape,
                                        const std::vector<uint64_t> &src_strides,
                                        int contig_start, uint64_t block_size)
    {
        if (dim == contig_start)
        {
            if (block_size == 1)
            {
                *dst = *src;
            }
            else
            {
                std::memcpy(dst, src, block_size * sizeof(float));
            }
            dst += block_size;
            return;
        }

        uint32_t dim_size = shape[dim];
        uint64_t src_stride = src_strides[dim];

        for (uint32_t i = 0; i < dim_size; ++i)
        {
            copyStridedToContiguous(dim + 1, src + i * src_stride, dst,
                                    shape, src_strides, contig_start, block_size);
        }
    }

    // General strided copy using O(1) incremental coordinates
    inline void copyStridedGeneral(const float *src, float *dst,
                                   const std::vector<uint32_t> &shape,
                                   const std::vector<uint64_t> &src_strides,
                                   const std::vector<uint64_t> &dst_strides)
    {
        int rank = static_cast<int>(shape.size());
        if (rank == 0)
        {
            *dst = *src;
            return;
        }

        uint64_t total = countElements(shape);
        uint32_t coords[8] = {0};
        uint64_t src_idx = 0;
        uint64_t dst_idx = 0;

        for (uint64_t i = 0; i < total; ++i)
        {
            dst[dst_idx] = src[src_idx];

            int d = rank - 1;
            while (d >= 0)
            {
                uint32_t old_coord = coords[d];
                coords[d]++;

                if (coords[d] < shape[d])
                {
                    src_idx += src_strides[d];
                    dst_idx += dst_strides[d];
                    break;
                }

                coords[d] = 0;
                src_idx -= (uint64_t)old_coord * src_strides[d];
                dst_idx -= (uint64_t)old_coord * dst_strides[d];
                d--;
            }
        }
    }

} // namespace scatter_detail

inline bool matchScatterF32_ND_Inplace_Fast(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 5)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;
    return true;
}

inline void runInplaceScatterF32_ND_Fast(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *target = static_cast<const float *>(inputs[0]);
    const float *updates = static_cast<const float *>(inputs[1]);
    const int32_t *starts = static_cast<const int32_t *>(inputs[2]);
    const int32_t *steps = static_cast<const int32_t *>(inputs[4]);
    float *out = static_cast<float *>(outputs[0]);

    const auto &out_shape = outViews[0].getShape();
    const auto &upd_shape = inViews[1].getShape();
    const auto &upd_strides = inViews[1].strides;
    const auto &out_strides = outViews[0].strides;
    const auto &tgt_strides = inViews[0].strides;

    uint64_t n_target = countElements(out_shape);
    uint64_t n_updates = countElements(upd_shape);
    int ndim = static_cast<int>(upd_shape.size());

    // --- Step 1: Copy target to out if different buffers ---
    if (target != out)
    {
        int tgt_contig = scatter_detail::findContiguousSuffix(out_shape, tgt_strides);
        int out_contig = scatter_detail::findContiguousSuffix(out_shape, out_strides);

        if (tgt_contig == 0 && out_contig == 0)
        {
            // Both contiguous - fast path
            std::memcpy(out, target, n_target * sizeof(float));
        }
        else if (out_contig == 0)
        {
            // Destination contiguous, source may be strided
            uint64_t block_size = 1;
            for (int d = tgt_contig; d < ndim; ++d)
            {
                block_size *= out_shape[d];
            }
            float *temp_dst = out;
            scatter_detail::copyStridedToContiguous(0, target, temp_dst, out_shape,
                                                    tgt_strides, tgt_contig, block_size);
        }
        else
        {
            // Both strided - use incremental copy
            scatter_detail::copyStridedGeneral(target, out, out_shape,
                                               tgt_strides, out_strides);
        }
    }

    // --- Step 2: Scatter updates ---
    if (n_updates == 0)
        return;

    // Scalar case
    if (ndim == 0)
    {
        int32_t s = inViews[2].getShape().empty() ? 0 : starts[0];
        if (s < 0)
            s += out_shape.empty() ? 1 : out_shape[0];
        out[s * out_strides[0]] = updates[0];
        return;
    }

    // Pre-compute adjusted starts, steps, and stride multipliers
    // out_idx = base_offset + sum(coords[d] * stride_multipliers[d])
    uint32_t starts_size = inViews[2].getShape().empty() ? 0 : inViews[2].getShape()[0];
    uint32_t steps_size = inViews[4].getShape().empty() ? 0 : inViews[4].getShape()[0];

    uint64_t stride_multipliers[8];
    uint64_t base_offset = 0;

    for (int d = 0; d < ndim; ++d)
    {
        int32_t adj_start = (d < (int)starts_size) ? starts[d] : 0;
        if (adj_start < 0)
            adj_start += out_shape[d];
        int32_t adj_step = (d < (int)steps_size) ? steps[d] : 1;

        stride_multipliers[d] = (uint64_t)adj_step * out_strides[d];
        base_offset += (uint64_t)adj_start * out_strides[d];
    }

    // Check if updates are fully contiguous
    bool updates_contig = (scatter_detail::findContiguousSuffix(upd_shape, upd_strides) == 0);

    // Parallelization threshold
    const uint64_t PARALLEL_THRESHOLD = 4096;

    if (ndim > 0 && upd_shape[0] > 1 && n_updates >= PARALLEL_THRESHOLD)
    {
        // Parallelize on outermost dimension
        uint64_t inner_elements = 1;
        for (int d = 1; d < ndim; ++d)
        {
            inner_elements *= upd_shape[d];
        }

#pragma omp parallel for schedule(static)
        for (uint32_t i0 = 0; i0 < upd_shape[0]; ++i0)
        {
            uint32_t coords[8] = {0};
            coords[0] = i0;

            uint64_t out_idx = base_offset + (uint64_t)i0 * stride_multipliers[0];
            uint64_t upd_idx = (uint64_t)i0 * upd_strides[0];
            uint64_t flat_idx = i0 * inner_elements;

            for (uint64_t j = 0; j < inner_elements; ++j)
            {
                out[out_idx] = updates[updates_contig ? flat_idx++ : upd_idx];

                int d = ndim - 1;
                while (d >= 1)
                {
                    uint32_t old_coord = coords[d];
                    coords[d]++;

                    if (coords[d] < upd_shape[d])
                    {
                        out_idx += stride_multipliers[d];
                        if (!updates_contig)
                            upd_idx += upd_strides[d];
                        break;
                    }

                    coords[d] = 0;
                    out_idx -= (uint64_t)old_coord * stride_multipliers[d];
                    if (!updates_contig)
                        upd_idx -= (uint64_t)old_coord * upd_strides[d];
                    d--;
                }
            }
        }
    }
    else
    {
        // Sequential version with O(1) coordinate updates
        uint32_t coords[8] = {0};
        uint64_t out_idx = base_offset;
        uint64_t upd_idx = 0;

        for (uint64_t i = 0; i < n_updates; ++i)
        {
            out[out_idx] = updates[updates_contig ? i : upd_idx];

            int d = ndim - 1;
            while (d >= 0)
            {
                uint32_t old_coord = coords[d];
                coords[d]++;

                if (coords[d] < upd_shape[d])
                {
                    out_idx += stride_multipliers[d];
                    if (!updates_contig)
                        upd_idx += upd_strides[d];
                    break;
                }

                coords[d] = 0;
                out_idx -= (uint64_t)old_coord * stride_multipliers[d];
                if (!updates_contig)
                    upd_idx -= (uint64_t)old_coord * upd_strides[d];
                d--;
            }
        }
    }
}

uint32_t refFactoryScatterF32_ND_Inplace_Fast(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.scatter(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
}

REGISTER_KERNEL_INPLACE("SCATTER_inplace_fast", 5, matchScatterF32_ND_Inplace_Fast, runInplaceScatterF32_ND_Fast, refFactoryScatterF32_ND_Inplace_Fast, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32}, {{8, 32}, {8, 32}, {8}, {8}, {8}}, {false, false, false, false, false}, {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});