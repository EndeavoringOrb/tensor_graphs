#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <stdexcept>
#include <cuda_runtime.h>

inline bool matchCopyTo_CUDA_CPU_F32(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 1)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    if (inputs[0].backend != Backend::CUDA || output.backend != Backend::CPU)
        return false;
    if (inputs[0].shape != output.shape)
        return false;

    return true;
}

inline void runCopyTo_CUDA_CPU_F32(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                   const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *src = static_cast<const float *>(inputs[0]);
    float *dst = static_cast<float *>(outputs[0]);
    uint64_t numElements = countElements(inViews[0].shape);

    for (uint64_t i = 0; i < numElements; ++i)
    {
        uint64_t srcIdx = getStridedIndex(i, inViews[0].shape, inViews[0].strides);
        uint64_t dstIdx = getStridedIndex(i, outViews[0].shape, outViews[0].strides);

        cudaError_t err = cudaMemcpy(dst + dstIdx, src + srcIdx, sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("CUDA strided copyto device->host failed: ") + cudaGetErrorString(err));
        }
    }
}

REGISTER_REF_KERNEL(OpType::COPY_TO, Backend::CPU, matchCopyTo_CUDA_CPU_F32, runCopyTo_CUDA_CPU_F32);
#endif