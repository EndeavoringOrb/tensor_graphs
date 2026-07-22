#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <cstring>

// ------------------------------------------------------------
// Matcher
// ------------------------------------------------------------
inline bool matchCopyTo_CUDA_CPU(const std::vector<TensorNode> &inputs,
                                 const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;

    if (inputs[0].strides != output.strides)
        return false;

    if (!isContiguous(output))
        return false;

    return true;
}

// ------------------------------------------------------------
// Runner
// ------------------------------------------------------------
inline void runCopyTo_CUDA_CPU(const KernelContext &ctx)
{
    const uint8_t *src = static_cast<const uint8_t *>(ctx.inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(ctx.outputs[0]);

    uint64_t numElements = countElements(ctx.inViews[0].getShape());
    uint64_t elemSize = getDTypeSize(ctx.inViews[0].dtype);
    uint64_t bytes = numElements * elemSize;

    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        Error::throw_err(cudaGetErrorString(err));
}

REGISTER_REF_KERNEL(OpType::COPY_TO, 1, 1, matchCopyTo_CUDA_CPU, runCopyTo_CUDA_CPU, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::ANY}, {{8, 32}}, {false}, {{MemSpace(1, HandleType::CUDA)}});

#endif