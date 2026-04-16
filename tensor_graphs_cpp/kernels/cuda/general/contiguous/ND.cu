#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

/**
 * CUDA KERNEL: CONTIGUOUS ND (FLOAT32)
 * ---------------------------------------------------------
 * This kernel ensures a tensor is contiguous in memory.
 * It reads from a potentially strided source and writes to a 
 * linear (contiguous) destination.
 */

namespace ContiguousCUDA
{
    constexpr int MAX_RANK = 8;

    struct ContiguousParams
    {
        uint32_t rank;
        uint32_t shape[MAX_RANK];
        int64_t in_strides[MAX_RANK];
    };

    __global__ void contiguous_kernel(const float *__restrict__ src, float *__restrict__ dst, uint64_t numElements, ContiguousParams p)
    {
        uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numElements)
            return;

        uint64_t temp = idx;
        uint64_t src_idx = 0;

        // Unravel the linear index 'idx' into N-D coordinates based on the shape,
        // then ravel back into a physical index using the input strides.
#pragma unroll
        for (int i = MAX_RANK - 1; i >= 0; --i)
        {
            if (i >= (int)p.rank)
                continue;
            
            uint32_t coord = temp % p.shape[i];
            temp /= p.shape[i];
            src_idx += (uint64_t)coord * p.in_strides[i];
        }

        dst[idx] = src[src_idx];
    }
}

/**
 * Match Function:
 * Requires Rank <= 8, Input backend CUDA, Output backend CUDA, and Output must be contiguous.
 */
inline bool matchContiguous_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    const auto &in = inputs[0];

    if (in.backend != Backend::CUDA || output.backend != Backend::CUDA)
        return false;
    if (in.dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Check rank limits
    if (in.getShape().size() > ContiguousCUDA::MAX_RANK || in.getShape().empty())
        return false;

    if (in.getShape() != output.getShape())
        return false;

    // Output must be contiguous
    if (!isContiguous(output))
        return false;

    return true;
}

/**
 * Run Function:
 * Prepares the stride and shape metadata and launches the GPU kernel.
 */
inline void runContiguous_CUDA_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                 const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *src = static_cast<const float *>(inputs[0]);
    float *dst = static_cast<float *>(outputs[0]);

    uint64_t numElements = countElements(outViews[0].getShape());
    if (numElements == 0)
        return;

    ContiguousCUDA::ContiguousParams p;
    p.rank = (uint32_t)outViews[0].getShape().size();

    for (uint32_t i = 0; i < p.rank; ++i)
    {
        p.shape[i] = outViews[0].getShape()[i];
        p.in_strides[i] = inViews[0].strides[i];
    }

    int blockSize = 256;
    uint32_t gridSize = (uint32_t)((numElements + blockSize - 1) / blockSize);

    ContiguousCUDA::contiguous_kernel<<<gridSize, blockSize>>>(src, dst, numElements, p);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        Error::throw_err("CUDA kernel launch failed in Contiguous_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * Reference Factory:
 * Maps this kernel to the standard graph contiguous operation.
 */
inline uint32_t refFactoryContiguous_CUDA_ND(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.contiguous(inputs[0]);
}

// Register as a named general kernel for CUDA
REGISTER_KERNEL("Contiguous_CUDA_ND", 1, matchContiguous_CUDA_ND, runContiguous_CUDA_ND, refFactoryContiguous_CUDA_ND, {Backend::CUDA},
    {DType::FLOAT32},      // Input DType
    {{1024, 640}},         // Dummy shape
    {false},               // Input does NOT require contiguity (that's the point of this kernel)
    {{Backend::CUDA}}      // Input backends
);

#endif