// File: tensor_graphs_cpp/kernels/cpu/reference/permute/F32_ND_naive.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>
#include <cstring> // For memcpy

/**
 * KERNEL: PERMUTE F32 ND (NAIVE REFERENCE)
 * Reorders the dimensions of the input tensor using simple linear index mapping.
 * No recursion, no stride precomputation, no inner-loop unrolling.
 */

inline bool matchPermuteF32_ND_naive(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    // Inputs: Data (0), Permutation Indices (1)
    if (inputs.size() != 2)
        return false;

    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[1].dtype != DType::INT32)
        return false;

    // Check permutation tensor shape matches data rank
    if (inputs[1].shape.size() != 1 || inputs[1].shape[0] != inputs[0].shape.size())
        return false;

    return true;
}

inline void runPermuteF32_ND_naive(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                   const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *src = static_cast<const float *>(inputs[0]);
    const int32_t *perm = static_cast<const int32_t *>(inputs[1]);
    float *dst = static_cast<float *>(outputs[0]);

    const auto &inShape = inViews[0].shape;
    const auto &outShape = outViews[0].shape;

    uint64_t numElements = countElements(outShape);
    if (numElements == 0)
        return;
    if (outShape.empty())
    {
        *dst = *src;
        return;
    }

    // Naive approach: for each element in output, compute its corresponding index in input
    std::vector<int64_t> inStrides(inShape.size(), 1);
    for (int i = (int)inShape.size() - 2; i >= 0; --i)
        inStrides[i] = inStrides[i + 1] * inShape[i + 1];

    std::vector<int64_t> outStrides(outShape.size(), 1);
    for (int i = (int)outShape.size() - 2; i >= 0; --i)
        outStrides[i] = outStrides[i + 1] * outShape[i + 1];

    for (uint64_t outIdx = 0; outIdx < numElements; ++outIdx)
    {
        // Convert linear index to output coordinates
        std::vector<int64_t> coord(outShape.size());
        uint64_t tmp = outIdx;
        for (size_t i = 0; i < outShape.size(); ++i)
        {
            coord[i] = tmp / outStrides[i];
            tmp %= outStrides[i];
        }

        // Map to input coordinates using permutation
        uint64_t inIdx = 0;
        for (size_t i = 0; i < coord.size(); ++i) {
            int32_t p_dim = perm[i];
            if (p_dim < 0 || (size_t)p_dim >= inShape.size()) {
                Error::throw_err("Invalid permutation index: " + std::to_string(p_dim) + " for rank " + std::to_string(inShape.size()));
            }
            inIdx += coord[i] * inStrides[p_dim];
        }

        dst[outIdx] = src[inIdx];
    }
}

REGISTER_REF_KERNEL(OpType::PERMUTE, matchPermuteF32_ND_naive, runPermuteF32_ND_naive, {Backend::CPU});