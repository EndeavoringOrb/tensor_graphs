#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

inline bool matchCopyTo_CPU_CUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &in = inputs[0];
    return (in.dtype == output.dtype && in.getShape() == output.getShape() && in.strides == output.strides && isContiguous(output));
}

inline void runCopyTo_CPU_CUDA(const std::vector<const void *> &inputs, const std::vector<void *> &outputs, const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const uint8_t *src = static_cast<const uint8_t *>(inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(outputs[0]);
    uint64_t numElements = countElements(inViews[0].getShape());
    uint64_t elemSize = getDTypeSize(inViews[0].dtype);
    size_t bytes = numElements * elemSize;

    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        Error::throw_err(cudaGetErrorString(err));
}

REGISTER_REF_KERNEL(OpType::COPY_TO, 1, matchCopyTo_CPU_CUDA, runCopyTo_CPU_CUDA, {Backend::CUDA}, {DType::ANY}, {{8, 32}}, {true}, {{Backend::CPU}});
#endif