// File: tensor_graphs_cpp/kernels/cpu/reference/permute/F32_ND.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>
#include <numeric>

/**
 * KERNEL: PERMUTE F32 ND
 * Reorders the dimensions of the input tensor.
 */

inline bool matchPermuteF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    // Inputs: Data (0), Permutation Indices (1)
    if (inputs.size() != 2)
        return false;
    
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[1].dtype != DType::INT32)
        return false;

    // Contiguity check: Permutation often results in non-contiguous memory access patterns logically,
    // but this reference implementation assumes the input is contiguous physically,
    // and writes sequentially to a contiguous output buffer.
    if (!inputs[0].view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

inline void runPermuteF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                             const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *src = static_cast<const float *>(inputs[0]);
    const int32_t *perm = static_cast<const int32_t *>(inputs[1]);
    float *dst = static_cast<float *>(outputs[0]);

    const auto &inShape = inViews[0].shape;
    uint32_t ndim = static_cast<uint32_t>(inShape.size());

    // Calculate input strides
    std::vector<int64_t> inStrides(ndim);
    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        inStrides[i] = stride;
        stride *= inShape[i];
    }

    uint64_t numElements = countElements(outViews[0].shape);
    const auto &outShape = outViews[0].shape;

    // Precompute output strides
    std::vector<int64_t> outStrides(ndim);
    stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        outStrides[i] = stride;
        stride *= outShape[i];
    }

    // Map permuted dimensions
    std::vector<uint32_t> permuted(ndim);
    for(uint32_t i = 0; i < ndim; ++i) {
        permuted[i] = static_cast<uint32_t>(perm[i]);
    }

    // Naive implementation: Iterate over output space
    for (uint64_t i = 0; i < numElements; ++i) {
        uint64_t src_idx = 0;

        // Decompose output linear index into coordinates
        // and reconstruct source linear index
        for (uint32_t d = 0; d < ndim; ++d) {
            uint32_t coord = (i / outStrides[d]) % outShape[d];
            src_idx += coord * inStrides[permuted[d]];
        }
        
        dst[i] = src[src_idx];
    }
}

REGISTER_REF_KERNEL(OpType::PERMUTE, Backend::CPU, matchPermuteF32_ND, runPermuteF32_ND);