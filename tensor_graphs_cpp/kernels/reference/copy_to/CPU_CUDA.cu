#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

inline bool matchCopyTo_CPU_CUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &in = inputs[0];
    return (in.dtype == output.dtype && in.getShape() == output.getShape() && in.strides == output.strides && isContiguous(output));
}

inline void runCopyTo_CPU_CUDA(const KernelContext &ctx)
{
    const uint8_t *src = static_cast<const uint8_t *>(ctx.inputs[0]);
    uint8_t *dst = static_cast<uint8_t *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());
    uint64_t elemSize = getDTypeSize(ctx.inViews[0].dtype);
    uint64_t bytes = numElements * elemSize;

    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        Error::throw_err(cudaGetErrorString(err));
}

REGISTER_REF_KERNEL(OpType::COPY_TO, 1, 1, matchCopyTo_CPU_CUDA, runCopyTo_CPU_CUDA, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CPU)}, {DType::ANY}, {{8, 32}}, {true}, {{MemSpace(1, HandleType::CPP)}});
#endif