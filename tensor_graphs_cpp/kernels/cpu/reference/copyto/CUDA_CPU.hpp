#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

// ------------------------------------------------------------
// GPU gather kernel (strided → contiguous)
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

    if (inputs[0].shape != output.shape)
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

    uint64_t numElements = countElements(inViews[0].shape);
    uint64_t elemSize = getDTypeSize(inViews[0].dtype);
    size_t bytes = numElements * elemSize;

    auto isContiguous = [](const TensorView &v)
    {
        uint64_t expected = 1;
        for (int i = v.shape.ndim - 1; i >= 0; --i)
        {
            if (v.strides[i] != expected)
                return false;
            expected *= v.shape.dims[i];
        }
        return true;
    };

    bool srcContig = isContiguous(inViews[0]);
    bool dstContig = isContiguous(outViews[0]);

    // Fast path
    if (srcContig && dstContig)
    {
        cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
        return;
    }

    uint8_t *d_temp = (uint8_t *)src;

    if (!srcContig)
    {
        cudaMalloc(&d_temp, bytes);

        size_t metaBytes = inViews[0].shape.ndim * sizeof(uint64_t);

        uint64_t *d_dims;
        uint64_t *d_strides;

        cudaMalloc(&d_dims, metaBytes);
        cudaMalloc(&d_strides, metaBytes);

        cudaMemcpy(d_dims, inViews[0].shape.dims, metaBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, inViews[0].strides, metaBytes, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (numElements + blockSize - 1) / blockSize;

        gatherKernelBytes<<<numBlocks, blockSize>>>(
            src,
            d_temp,
            numElements,
            elemSize,
            inViews[0].shape.ndim,
            d_dims,
            d_strides);

        cudaFree(d_dims);
        cudaFree(d_strides);
    }

    std::vector<uint8_t> hostBuffer(bytes);

    cudaMemcpy(hostBuffer.data(), d_temp, bytes, cudaMemcpyDeviceToHost);

    if (!srcContig)
        cudaFree(d_temp);

    if (dstContig)
    {
        memcpy(dst, hostBuffer.data(), bytes);
        return;
    }

    for (uint64_t i = 0; i < numElements; ++i)
    {
        uint64_t dstIdx = getStridedIndex(i, outViews[0].shape, outViews[0].strides);

        memcpy(dst + dstIdx * elemSize,
               hostBuffer.data() + i * elemSize,
               elemSize);
    }
}

REGISTER_REF_KERNEL(OpType::COPY_TO, Backend::CPU,
                    matchCopyTo_CUDA_CPU,
                    runCopyTo_CUDA_CPU);

#endif