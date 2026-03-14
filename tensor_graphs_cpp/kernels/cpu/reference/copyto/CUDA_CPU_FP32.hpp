// File: tensor_graphs_cpp/kernels/cpu/reference/copyto/CUDA_CPU_FP32.hpp
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

    // CUDA -> CPU copy matches the CPU backend because its output is physically mapped to CPU
    if (inputs[0].backend != Backend::CUDA || output.backend != Backend::CPU)
        return false;

    if (inputs[0].shape != output.shape)
        return false;

    if (!inputs[0].view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

inline void runCopyTo_CUDA_CPU_F32(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                   const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const void *src = inputs[0];
    void *dst = outputs[0];
    uint64_t sizeBytes = countElements(inViews[0].shape) * getDTypeSize(DType::FLOAT32);

    cudaError_t err = cudaMemcpy(dst, src, sizeBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA copyto device->host failed: ") + cudaGetErrorString(err));
    }
}

REGISTER_REF_KERNEL(OpType::COPY_TO, Backend::CPU, matchCopyTo_CUDA_CPU_F32, runCopyTo_CUDA_CPU_F32);
#endif