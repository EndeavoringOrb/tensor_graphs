#pragma once
#include "core/common/partial.hpp"

/**
 * KERNEL: Partial_DOT_inplace
 *
 * Computes a partial DOT (matrix multiplication) for only a slice of the output,
 * writing the result directly into the target buffer at the correct strided positions.
 *
 * This is the DOT equivalent of Partial_Add_inplace / Partial_Mul_inplace.
 * It is critical for decode (autoregressive generation) where only a single
 * row of the DOT output is needed (the new token position), avoiding the
 * full sequence-length DOT computation.
 *
 * Inputs (6):
 *   [0] target  - F32 tensor, the existing output buffer to update in-place
 *   [1] A       - F32 tensor, the left matrix  [B, M', K]  (sliced to the needed rows)
 *   [2] B       - F32 tensor, the right matrix [B, K, N']  (sliced to the needed cols)
 *   [3] starts  - I32 1-D array, slice start per dimension
 *   [4] ends    - I32 1-D array, slice end per dimension
 *   [5] steps   - I32 1-D array, slice step per dimension
 *
 * Semantics:
 *   out[starts:ends:steps] = A @ B
 *   (the rest of `target` is preserved)
 *
 * Reference decomposition:
 *   scatter(target, contiguous(dot(contiguous(slice(A,...)), contiguous(slice(B,...)))), starts, ends, steps)
 */

inline bool matchPartialDotInplace(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Must have exactly 6 inputs: target, A, B, starts, ends, steps
    if (inputs.size() != 6)
        return false;

    // target, A, B must be F32; starts, ends, steps must be I32
    if (inputs[0].dtype != DType::FLOAT32)
        return false;
    if (inputs[1].dtype != DType::FLOAT32)
        return false;
    if (inputs[2].dtype != DType::FLOAT32)
        return false;
    if (inputs[3].dtype != DType::INT32)
        return false;
    if (inputs[4].dtype != DType::INT32)
        return false;
    if (inputs[5].dtype != DType::INT32)
        return false;

    // A and B must be rank 2 or 3 (DOT only supports these ranks)
    if (inputs[1].getShape().size() < 2 || inputs[1].getShape().size() > 3)
        return false;
    if (inputs[2].getShape().size() < 2 || inputs[2].getShape().size() > 3)
        return false;

    // Target must not be PERSISTENT (can't write into weights/constants)
    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;

    return true;
}

inline uint32_t refFactoryPartialDot(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t target = inputs[0];
    uint32_t A = inputs[1];
    uint32_t B = inputs[2];
    uint32_t st = inputs[3];
    uint32_t en = inputs[4];
    uint32_t step = inputs[5];

    // scatter(target, contiguous(dot(contiguous(slice(A, ...)), contiguous(slice(B, ...)))), starts, ends, steps)
    uint32_t op_res = graph.dot(graph.contiguous(graph.slice(A, st, en, step)),
                                graph.contiguous(graph.slice(B, st, en, step)));
    return graph.scatter(target, graph.contiguous(op_res), st, en, step);
}

inline void runPartialDotInplace(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                 const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *target = (const float *)inputs[0];
    const float *A = (const float *)inputs[1];
    const float *B = (const float *)inputs[2];
    const int32_t *starts = (const int32_t *)inputs[3];
    const int32_t *ends = (const int32_t *)inputs[4];
    const int32_t *steps = (const int32_t *)inputs[5];
    float *out = (float *)outputs[0];

    const auto &out_shape = outViews[0].getShape();
    const auto &out_strides = outViews[0].strides;

    // Step 1: Copy target to output if different buffers (inplace copy-on-write)
    partial_ops::copy_target_if_needed(target, out, out_shape, inViews[0].strides, out_strides);

    // Step 2: Compute slice shape and adjusted starts/steps for the output region
    uint32_t starts_size = inViews[3].getShape().empty() ? 0 : inViews[3].getShape()[0];
    uint32_t ends_size = inViews[4].getShape().empty() ? 0 : inViews[4].getShape()[0];
    uint32_t steps_size = inViews[5].getShape().empty() ? 0 : inViews[5].getShape()[0];

    std::vector<uint32_t> slice_shape(out_shape.size());
    std::vector<int32_t> adj_starts(out_shape.size());
    std::vector<int32_t> adj_steps(out_shape.size());

    for (size_t d = 0; d < out_shape.size(); ++d)
    {
        int32_t st = (d < starts_size) ? starts[d] : 0;
        int32_t en = (d < ends_size) ? ends[d] : (int32_t)out_shape[d];
        int32_t sp = (d < steps_size) ? steps[d] : 1;
        if (st < 0)
            st += out_shape[d];
        if (en < 0)
            en += out_shape[d];
        adj_starts[d] = st;
        adj_steps[d] = sp;
        slice_shape[d] = static_cast<uint32_t>(std::max(0, (en - st + sp - 1) / sp));
    }

    // Step 3: Get A and B dimensions
    const auto &a_shape = inViews[1].getShape();
    const auto &a_strides = inViews[1].strides;
    const auto &b_shape = inViews[2].getShape();
    const auto &b_strides = inViews[2].strides;

    size_t rank = a_shape.size();
    uint32_t B_count = (rank == 3) ? a_shape[0] : 1;
    uint32_t M = a_shape[rank - 2];
    uint32_t K = a_shape[rank - 1];
    uint32_t N = b_shape[rank - 1];

    // Step 4: Compute DOT for each element in the slice region
    // For each (b, m, n) in the slice, compute:
    //   out[slice_index(b,m,n)] = sum_k A[b,m,k] * B[b,k,n]
    //
    // We iterate over the slice coordinates and map them to the output buffer positions.

    // Compute the iteration ranges for the slice
    // The slice covers dimensions [d0_start:d0_end:d0_step, d1_start:d1_end:d1_step, d2_start:d2_end:d2_step]
    // For 3D DOT output [B_batch, M, N], the slice maps to:
    //   batch index = slice coord for dim 0
    //   M index = slice coord for dim 1
    //   N index = slice coord for dim 2

    uint32_t slice_B = slice_shape[0];
    uint32_t slice_M = slice_shape[1];
    uint32_t slice_N = slice_shape[2];

    // Iterate over the slice elements
    for (uint32_t sb = 0; sb < slice_B; ++sb)
    {
        uint32_t b = adj_starts[0] + sb * adj_steps[0];

        for (uint32_t sm = 0; sm < slice_M; ++sm)
        {
            uint32_t m = adj_starts[1] + sm * adj_steps[1];

            // Compute the base offset for A[b, m, :]
            uint64_t a_row_offset = 0;
            if (rank == 3)
            {
                a_row_offset = (uint64_t)b * a_strides[0] + (uint64_t)m * a_strides[1];
            }
            else
            {
                a_row_offset = (uint64_t)m * a_strides[0];
            }

            for (uint32_t sn = 0; sn < slice_N; ++sn)
            {
                uint32_t n = adj_starts[2] + sn * adj_steps[2];

                // Compute the output index
                uint64_t out_idx = 0;
                if (rank == 3)
                {
                    out_idx = (uint64_t)b * out_strides[0] + (uint64_t)m * out_strides[1] + (uint64_t)n * out_strides[2];
                }
                else
                {
                    out_idx = (uint64_t)m * out_strides[0] + (uint64_t)n * out_strides[1];
                }

                // Compute dot product: sum_k A[b,m,k] * B[b,k,n]
                float sum = 0.0f;
                if (rank == 3)
                {
                    uint64_t b_col_offset = (uint64_t)b * b_strides[0] + (uint64_t)n * b_strides[2];
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        sum += A[a_row_offset + (uint64_t)k * a_strides[rank - 1]] * B[b_col_offset + (uint64_t)k * b_strides[1]];
                    }
                }
                else
                {
                    uint64_t b_col_offset = (uint64_t)n * b_strides[1];
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        sum += A[a_row_offset + (uint64_t)k * a_strides[1]] * B[b_col_offset + (uint64_t)k * b_strides[0]];
                    }
                }

                out[out_idx] = sum;
            }
        }
    }
}

REGISTER_KERNEL_INPLACE(
    "Partial_DOT_inplace",
    6,
    matchPartialDotInplace,
    runPartialDotInplace,
    refFactoryPartialDot,
    {Backend::CPU},
    {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32},
    {{8, 32}, {8, 8, 8}, {8, 8, 8}, {8}, {8}, {8}},
    {false, false, false, false, false, false},
    {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});