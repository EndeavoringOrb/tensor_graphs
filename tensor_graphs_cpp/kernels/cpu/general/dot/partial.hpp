// File: tensor_graphs_cpp/kernels/cpu/general/dot/partial.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <thread>
#include <algorithm>
#include <cstring>

/**
 * KERNEL: Partial Dot F32 3D
 *
 * Fuses: SLICE -> CONTIGUOUS -> DOT -> SCATTER
 *
 * Purpose: Partial evaluation of cached tensors. Instead of allocating a temporary
 * buffer for the partial result and scattering it back, this kernel computes the
 * dot product directly into the correct cache locations using the provided slice offsets.
 *
 * Inputs:
 * 0: cache        [B, M, N]  (Target tensor, mutable, TRANSIENT/PINNED)
 * 1: A_partial    [B, M, K]  (Sliced & contiguous input)
 * 2: B_partial    [B, K, N]  (Sliced & contiguous weight matrix)
 * 3: starts       [3]        (Slice start indices)
 * 4: ends         [3]        (Slice end indices)
 * 5: steps        [3]        (Slice steps, typically 1)
 *
 * Output:
 * 0: cache        [B, M, N]  (Accumulated result in-place)
 */

inline bool matchPartialDotF32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 6)
        return false;
    // 0: cache, 1: A_partial, 2: B_partial
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || inputs[2].dtype != DType::FLOAT32)
        return false;

    // All tensors in this fused chain are 3D in the current model layout.
    // Note: B_partial is 3D because the planner slices the reshaped 3D weight matrix.
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 3 || inputs[2].getShape().size() != 3)
        return false;

    // 3: starts, 4: ends, 5: steps
    if (inputs[3].dtype != DType::INT32 || inputs[4].dtype != DType::INT32 || inputs[5].dtype != DType::INT32)
        return false;

    // Cache must be mutable (TRANSIENT or PINNED)
    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;

    // Output must match cache shape
    if (inputs[0].getShape() != output.getShape())
        return false;

    // Verify DOT dimensions match: A[..., K] x B[K, ...] -> Cache[..., N]
    if (inputs[1].getShape().back() != inputs[2].getShape()[1])
        return false;
    if (inputs[0].getShape().back() != inputs[2].getShape().back())
        return false;

    return true;
}

inline void runPartialDotF32_3D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *cache = static_cast<const float *>(inputs[0]);
    const float *A = static_cast<const float *>(inputs[1]);
    const float *B = static_cast<const float *>(inputs[2]);
    const int32_t *starts = static_cast<const int32_t *>(inputs[3]);
    const int32_t *steps = static_cast<const int32_t *>(inputs[5]);
    float *out = static_cast<float *>(outputs[0]); // Aliases cache

    uint32_t Bp = inViews[1].getShape()[0];
    uint32_t Mp = inViews[1].getShape()[1];
    uint32_t K = inViews[1].getShape()[2];
    uint32_t N_dim = inViews[0].getShape()[2];

    // Extract slice parameters
    int64_t b_start = starts[0];
    int64_t m_start = starts[1];
    int64_t n_start = starts[2];
    int64_t b_step = steps[0];
    int64_t m_step = steps[1];
    int64_t n_step = steps[2];

    // Strides
    int64_t cB_stride = outViews[0].strides[0];
    int64_t cM_stride = outViews[0].strides[1];
    int64_t cN_stride = outViews[0].strides[2];
    int64_t bK_stride = inViews[2].strides[1];
    int64_t bN_stride = inViews[2].strides[2];

    // Fast path for unit steps
    if (b_step == 1 && m_step == 1 && n_step == 1)
    {
        for (uint32_t bp = 0; bp < Bp; ++bp)
        {
            for (uint32_t mp = 0; mp < Mp; ++mp)
            {
                const float *a_row = A + (bp * Mp + mp) * K;
                float *c_row = out + (b_start + bp) * cB_stride + (m_start + mp) * cM_stride + n_start * cN_stride;

                for (uint32_t n = 0; n < N_dim; ++n)
                {
                    float sum = 0.0f;
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        sum += a_row[k] * B[k * bK_stride + n * bN_stride];
                    }
                    c_row[n] = sum;
                }
            }
        }
    }
    else
    {
        // General strided accumulation
        for (uint32_t bp = 0; bp < Bp; ++bp)
        {
            for (uint32_t mp = 0; mp < Mp; ++mp)
            {
                const float *a_row = A + (bp * Mp + mp) * K;
                float *c_row = out + (b_start + bp * b_step) * cB_stride +
                               (m_start + mp * m_step) * cM_stride +
                               n_start * cN_stride;

                for (uint32_t n = 0; n < N_dim; ++n)
                {
                    float sum = 0.0f;
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        sum += a_row[k] * B[k * bK_stride + n * bN_stride];
                    }
                    c_row[n * n_step] = sum;
                }
            }
        }
    }
}

inline uint32_t refFactoryPartialDot(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t cache = inputs[0];
    uint32_t A_partial = inputs[1];
    uint32_t B_partial = inputs[2];
    uint32_t starts = inputs[3];
    uint32_t ends = inputs[4];
    uint32_t steps = inputs[5];

    uint32_t dot_res = graph.dot(A_partial, B_partial);
    return graph.scatter(cache, dot_res, starts, ends, steps);
}

REGISTER_KERNEL_INPLACE("PartialDot_F32_3D", 6, matchPartialDotF32_3D, runPartialDotF32_3D, refFactoryPartialDot,
                        {Backend::CPU},
                        {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32},
                        {{1, 8, 262144}, {1, 8, 640}, {1, 640, 2048}, {3}, {3}, {3}},
                        {false, true, true, false, false, false},
                        {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});

#endif