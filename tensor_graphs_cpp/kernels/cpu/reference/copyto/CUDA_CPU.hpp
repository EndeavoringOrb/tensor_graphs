#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <cstring>

// ------------------------------------------------------------
// GPU gather kernel (strided source → contiguous destination)
// ------------------------------------------------------------
__global__ void gatherKernelBytes(const uint8_t *__restrict__ src,
                                  uint8_t *__restrict__ dst,
                                  uint64_t numElements,
                                  uint64_t elemSize,
                                  int ndim,
                                  const uint64_t *__restrict__ dims,
                                  const uint64_t *__restrict__ strides)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElements)
        return;

    uint64_t tmp = idx;
    uint64_t srcOffset = 0;

    // Calculate strided index from flat index
    for (int i = ndim - 1; i >= 0; --i)
    {
        uint64_t coord = tmp % dims[i];
        tmp /= dims[i];
        srcOffset += coord * strides[i];
    }

    const uint8_t *srcPtr = src + srcOffset * elemSize;
    uint8_t *dstPtr = dst + idx * elemSize;

    for (uint64_t b = 0; b < elemSize; ++b)
        dstPtr[b] = srcPtr[b];
}

// ------------------------------------------------------------
// Matcher
// ------------------------------------------------------------
inline bool matchCopyTo_CUDA_CPU(const std::vector<TensorNode> &inputs,
                                 const TensorNode &output,
                                 const std::unordered_map<uint32_t, uint32_t> &)
{
    if (inputs.size() != 1)
        return false;

    if (inputs[0].backend != Backend::CUDA || output.backend != Backend::CPU)
        return false;

    if (inputs[0].dtype != output.dtype)
        return false;

    if (inputs[0].getShape() != output.getShape())
        return false;

    return true;
}

// ------------------------------------------------------------
// Runner
// ------------------------------------------------------------
inline void runCopyTo_CUDA_CPU(const std::vector<const void *> &inputs,
                               const std::vector<void *> &outputs,
                               const std::vector<TensorView> &inViews,
                               const std::vector<TensorView> &outViews)
{
    const uint8_t *src = static_cast<const uint8_t *>(inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(outputs[0]);

    uint64_t numElements = countElements(inViews[0].getShape());
    uint64_t elemSize = getDTypeSize(inViews[0].dtype);
    size_t bytes = numElements * elemSize;

    bool srcContig = isContiguous(inViews[0]);
    bool dstContig = isContiguous(outViews[0]);

    // -------------------------
    // Fast path: Both contiguous
    // -------------------------
    if (srcContig && dstContig)
    {
        cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Error::throw_err(cudaGetErrorString(err));
        return;
    }

    // -------------------------
    // Step 1: Handle strided source on GPU
    // -------------------------
    const uint8_t *d_contiguousSrc = src;
    uint8_t *d_tempBuffer = nullptr;

    if (!srcContig)
    {
        // Allocate temporary contiguous buffer on device
        cudaMalloc(&d_tempBuffer, bytes);
        d_contiguousSrc = d_tempBuffer;

        int ndim = static_cast<int>(inViews[0].getShape().size());

        // Prepare metadata for the GPU (convert to uint64_t)
        std::vector<uint64_t> h_dims(ndim);
        std::vector<uint64_t> h_strides(ndim);
        for (int i = 0; i < ndim; ++i)
        {
            h_dims[i] = static_cast<uint64_t>(inViews[0].getShape()[i]);
            h_strides[i] = static_cast<uint64_t>(inViews[0].strides[i]);
        }

        uint64_t *d_dims, *d_strides;
        cudaMalloc(&d_dims, ndim * sizeof(uint64_t));
        cudaMalloc(&d_strides, ndim * sizeof(uint64_t));

        cudaMemcpy(d_dims, h_dims.data(), ndim * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, h_strides.data(), ndim * sizeof(uint64_t), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = static_cast<int>((numElements + blockSize - 1) / blockSize);

        gatherKernelBytes<<<numBlocks, blockSize>>>(
            src,
            d_tempBuffer,
            numElements,
            elemSize,
            ndim,
            d_dims,
            d_strides);

        cudaFree(d_dims);
        cudaFree(d_strides);
    }

    // -------------------------
    // Step 2: Copy to host
    // -------------------------
    if (dstContig)
    {
        // Copy directly to destination
        cudaMemcpy(dst, d_contiguousSrc, bytes, cudaMemcpyDeviceToHost);
    }
    else
    {
        // Copy to a temporary host buffer, then manually unpack to strided CPU destination
        std::vector<uint8_t> hostContiguous(bytes);
        cudaMemcpy(hostContiguous.data(), d_contiguousSrc, bytes, cudaMemcpyDeviceToHost);

        for (uint64_t i = 0; i < numElements; ++i)
        {
            uint64_t dstIdx = getStridedIndex(i, outViews[0].getShape(), outViews[0].strides);
            std::memcpy(dst + dstIdx * elemSize,
                        hostContiguous.data() + i * elemSize,
                        elemSize);
        }
    }

    // Cleanup
    if (d_tempBuffer)
    {
        cudaFree(d_tempBuffer);
    }
}

REGISTER_REF_KERNEL(OpType::COPY_TO, 1, matchCopyTo_CUDA_CPU, runCopyTo_CUDA_CPU, {Backend::CPU}, {DType::FLOAT32}, {{8, 32}}, {false}, {Backend::CUDA});

#endif