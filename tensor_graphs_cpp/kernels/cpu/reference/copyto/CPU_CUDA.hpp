#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

// ------------------------------------------------------------
// CUDA Kernel: unpack contiguous buffer → strided destination
// ------------------------------------------------------------
__global__ void unpackKernelBytes(const uint8_t *__restrict__ src,
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
    uint64_t dstOffset = 0;

    for (int i = ndim - 1; i >= 0; --i)
    {
        uint64_t coord = tmp % dims[i];
        tmp /= dims[i];
        dstOffset += coord * strides[i];
    }

    const uint8_t *srcPtr = src + idx * elemSize;
    uint8_t *dstPtr = dst + dstOffset * elemSize;

    for (uint64_t b = 0; b < elemSize; ++b)
        dstPtr[b] = srcPtr[b];
}

// ------------------------------------------------------------
// Matcher
// ------------------------------------------------------------
inline bool matchCopyTo_CPU_CUDA(const std::vector<TensorNode> &inputs,
                                 const TensorNode &output,
                                 const std::unordered_map<uint32_t, uint32_t> &)
{
    if (inputs.size() != 1)
        return false;

    if (inputs[0].backend != Backend::CPU || output.backend != Backend::CUDA)
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
inline void runCopyTo_CPU_CUDA(const std::vector<const void *> &inputs,
                               const std::vector<void *> &outputs,
                               const std::vector<TensorView> &inViews,
                               const std::vector<TensorView> &outViews)
{
    const uint8_t *src = static_cast<const uint8_t *>(inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(outputs[0]);

    uint64_t numElements = countElements(inViews[0].shape);
    uint64_t elemSize = getDTypeSize(inViews[0].dtype);
    size_t bytes = numElements * elemSize;

    // -------------------------
    // Contiguity checks
    // -------------------------
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

    // -------------------------
    // Fast path
    // -------------------------
    if (srcContig && dstContig)
    {
        cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(err));
        return;
    }

    // -------------------------
    // Step 1: CPU gather
    // -------------------------
    std::vector<uint8_t> hostBuffer;
    const uint8_t *packedSrc = src;

    if (!srcContig)
    {
        hostBuffer.resize(bytes);
        packedSrc = hostBuffer.data();

        for (uint64_t i = 0; i < numElements; ++i)
        {
            uint64_t srcIdx = getStridedIndex(i, inViews[0].shape, inViews[0].strides);

            memcpy(hostBuffer.data() + i * elemSize,
                   src + srcIdx * elemSize,
                   elemSize);
        }
    }

    // -------------------------
    // Step 2: copy to device
    // -------------------------
    uint8_t *d_temp = dst;

    if (!dstContig)
    {
        cudaMalloc(&d_temp, bytes);
    }

    cudaMemcpy(d_temp, packedSrc, bytes, cudaMemcpyHostToDevice);

    // -------------------------
    // Step 3: GPU unpack
    // -------------------------
    if (!dstContig)
    {
        size_t metaBytes = outViews[0].shape.ndim * sizeof(uint64_t);

        uint64_t *d_dims;
        uint64_t *d_strides;

        cudaMalloc(&d_dims, metaBytes);
        cudaMalloc(&d_strides, metaBytes);

        cudaMemcpy(d_dims, outViews[0].shape.dims, metaBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_strides, outViews[0].strides, metaBytes, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (numElements + blockSize - 1) / blockSize;

        unpackKernelBytes<<<numBlocks, blockSize>>>(
            d_temp,
            dst,
            numElements,
            elemSize,
            outViews[0].shape.ndim,
            d_dims,
            d_strides);

        cudaFree(d_temp);
        cudaFree(d_dims);
        cudaFree(d_strides);
    }
}

REGISTER_REF_KERNEL(OpType::COPY_TO, Backend::CUDA,
                    matchCopyTo_CPU_CUDA,
                    runCopyTo_CPU_CUDA);

#endif