#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <stdexcept>
#include <cuda_runtime.h>

inline bool matchCopyTo_CPU_CUDA_F32(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 1)
        return false;

    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // CPU -> CUDA copy matches the CUDA backend because its output is physically mapped to CUDA
    if (inputs[0].backend != Backend::CPU || output.backend != Backend::CUDA)
        return false;

    if (inputs[0].shape != output.shape)
        return false;

    return true;
}

inline void runCopyTo_CPU_CUDA_F32(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                   const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const void *src = inputs[0];
    void *dst = outputs[0];
    uint64_t sizeBytes = countElements(inViews[0].shape) * getDTypeSize(DType::FLOAT32);

    cudaError_t err = cudaMemcpy(dst, src, sizeBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA copyto host->device failed: ") + cudaGetErrorString(err));
    }
}

REGISTER_REF_KERNEL(OpType::COPY_TO, Backend::CUDA, matchCopyTo_CPU_CUDA_F32, runCopyTo_CPU_CUDA_F32);
#endif