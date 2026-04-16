#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>
#include <vector>
#include <algorithm>

/**
 * KERNEL: FastContiguous_ND
 *
 * Optimization Strategy:
 * Instead of element-wise copying, this kernel analyzes the tensor strides to find the
 * largest contiguous "suffix" of the tensor.
 *
 * If the innermost dimensions are contiguous, it collapses those dimensions into
 * a single large std::memcpy call, significantly reducing overhead.
 */

inline bool matchFastContiguous_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &in = inputs[0];

    if (in.dtype != output.dtype)
        return false;
    if (in.getShape() != output.getShape())
        return false;

    // Target must be contiguous
    if (!isContiguous(output))
        return false;

    return true;
}

inline void runFastContiguous_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                 const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const uint8_t *src_base = static_cast<const uint8_t *>(inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(outputs[0]);

    const auto &view = inViews[0];
    const auto &shape = view.getShape();
    const auto &strides = view.strides;
    const uint64_t elementSize = getDTypeSize(view.dtype);

    if (shape.empty())
    {
        if (countElements(shape) == 1)
        {
            std::memcpy(dst, src_base, elementSize);
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
            // Innermost dimension must have stride 1 to be contiguous
            if (strides[d] == 1)
            {
                contig_dim_start = d;
                contig_elements = shape[d];
            }
            else
            {
                break;
            }
        }
        else
        {
            // Dimension is contiguous if its stride matches the product of the
            // stride and size of the dimension to its right.
            if (strides[d] == static_cast<int64_t>(strides[d + 1]) * shape[d + 1])
            {
                contig_dim_start = d;
                contig_elements *= shape[d];
            }
            else
            {
                break;
            }
        }
    }

    // --- Step 2: Execution ---

    // Case A: The entire tensor is contiguous. Single massive memcpy.
    if (contig_dim_start == 0)
    {
        uint64_t totalBytes = countElements(shape) * elementSize;
        std::memcpy(dst, src_base, totalBytes);
        return;
    }

    // Case B: Partially contiguous or totally strided.
    int outer_rank = contig_dim_start;

    // If the innermost dimension wasn't stride 1, outer_rank == rank.
    // We treat the 'block' as a single element.
    bool fallback_to_element = (outer_rank == rank);
    uint64_t block_size = fallback_to_element ? 1 : contig_elements;
    int iteration_rank = fallback_to_element ? rank : outer_rank;

    std::vector<uint32_t> coords(iteration_rank, 0);
    uint64_t outer_iterations = 1;
    for (int i = 0; i < iteration_rank; ++i)
        outer_iterations *= shape[i];

    for (uint64_t i = 0; i < outer_iterations; ++i)
    {
        uint64_t offset = 0;
        for (int d = 0; d < iteration_rank; ++d)
        {
            offset += static_cast<uint64_t>(coords[d]) * strides[d];
        }

        // Copy the largest possible contiguous chunk
        std::memcpy(dst, src_base + (offset * elementSize), block_size * elementSize);

        dst += (block_size * elementSize);

        // Odometer increment
        for (int d = iteration_rank - 1; d >= 0; --d)
        {
            coords[d]++;
            if (coords[d] < shape[d])
                break;
            coords[d] = 0;
        }
    }
}

inline uint32_t refFactoryFastContiguous_ND(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 1)
        Error::throw_err("FastContiguous requires exactly 1 input");

    return graph.contiguous(inputs[0]);
}

REGISTER_KERNEL(
    "FastContiguous_ND",
    1,
    matchFastContiguous_ND,
    runFastContiguous_ND,
    refFactoryFastContiguous_ND,
    {Backend::CPU},
    {DType::FLOAT32},
    {{8, 32}},
    {false},
    {{Backend::CPU}});

