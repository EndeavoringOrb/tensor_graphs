#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

/**
 * CUDA KERNEL: PERMUTE ND (Contiguous)
 * ---------------------------------------------------------
 * This kernel materializes a permutation. It assumes the input is stored
 * contiguously and writes the result into a contiguous output buffer.
 */

namespace PermuteCUDA
{
    constexpr int MAX_RANK = 8;

    struct PermuteParams
    {
        uint32_t rank;
        uint64_t out_strides[MAX_RANK];
        uint64_t in_strides_permuted[MAX_RANK]; // Input strides reordered to match output dims
    };

    __global__ void permute_kernel(const float *__restrict__ src, float *__restrict__ dst, uint64_t numElements, PermuteParams p)
    {
        uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numElements)
            return;

        uint64_t temp = idx;
        uint64_t src_idx = 0;

// Traverse dimensions to calculate source index
// We logically "unravel" the output index into coordinates,
// then "ravel" them using input strides reordered by the permutation.
#pragma unroll
        for (int i = 0; i < MAX_RANK; ++i)
        {
            if (i >= p.rank)
                break;
            uint64_t coord = temp / p.out_strides[i];
            temp %= p.out_strides[i];
            src_idx += coord * p.in_strides_permuted[i];
        }

        dst[idx] = src[src_idx];
    }
}

/**
 * Match Function:
 * Requires Rank <= 8, both tensors contiguous, and CUDA backend.
 */
inline bool matchPermute_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    const auto &data = inputs[0];
    const auto &perm = inputs[1];

    if (data.backend != Backend::CUDA || output.backend != Backend::CUDA)
        return false;
    if (data.dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (perm.dtype != DType::INT32)
        return false;

    // Check rank limits
    if (data.getShape().size() > PermuteCUDA::MAX_RANK || data.getShape().size() == 0)
        return false;

    // Check permutation tensor shape matches data rank
    if (perm.getShape().size() != 1 || perm.getShape()[0] != data.getShape().size())
        return false;

    if (!isContiguous(output))
        return false;

    return true;
}

/**
 * Run Function:
 * Prepares the stride metadata on host and launches the GPU kernel.
 */
inline void runPermute_CUDA_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                               const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *src = static_cast<const float *>(inputs[0]);
    const int32_t *h_perm = static_cast<const int32_t *>(inputs[1]);
    float *dst = static_cast<float *>(outputs[0]);

    uint64_t numElements = countElements(outViews[0].getShape());
    if (numElements == 0)
        return;

    PermuteCUDA::PermuteParams p;
    p.rank = (uint32_t)outViews[0].getShape().size();

    // Calculate logical strides for a contiguous input
    std::vector<uint64_t> in_strides_contig = calcContiguousStrides(inViews[0].getShape());
    // Calculate logical strides for a contiguous output
    std::vector<uint64_t> out_strides_contig = calcContiguousStrides(outViews[0].getShape());

    for (uint32_t i = 0; i < p.rank; ++i)
    {
        p.out_strides[i] = (uint64_t)out_strides_contig[i];

        // The i-th dimension of the output corresponds to the h_perm[i]-th dimension of the input
        int32_t input_dim_idx = h_perm[i];
        
        // Safety check for permutation indices
        if (input_dim_idx < 0 || (uint32_t)input_dim_idx >= p.rank) {
            Error::throw_err("Invalid permutation index: " + std::to_string(input_dim_idx) + " for rank " + std::to_string(p.rank));
        }

        p.in_strides_permuted[i] = (uint64_t)in_strides_contig[input_dim_idx];
    }

    int blockSize = 256;
    uint32_t gridSize = (uint32_t)((numElements + blockSize - 1) / blockSize);

    PermuteCUDA::permute_kernel<<<gridSize, blockSize>>>(src, dst, numElements, p);
    
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        Error::throw_err("CUDA kernel launch failed in Permute_CUDA_Contiguous: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * Reference Factory:
 * Maps this kernel to the standard graph permute operation.
 */
inline uint32_t refFactoryPermute_CUDA_ND(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.permute(inputs[0], inputs[1]);
}

// Register as a named general kernel
REGISTER_KERNEL("Permute_CUDA_Contiguous", 2, matchPermute_CUDA_ND, runPermute_CUDA_ND, refFactoryPermute_CUDA_ND, {Backend::CUDA},
    {DType::FLOAT32, DType::INT32}, // Data, Indices
    {{1024, 640}, {2}},             // Dummy shapes
    {true, true},                   // Contiguity requirements for match
    {{Backend::CUDA}, {Backend::CPU}}   // Input backends
);

#endif