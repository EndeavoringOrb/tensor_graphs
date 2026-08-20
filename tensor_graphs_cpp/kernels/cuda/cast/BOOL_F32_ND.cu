#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void cast_bool_f32_nd_kernel(const bool* A, float* Out, uint64_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Out[idx] = A[idx] ? 1.0f : 0.0f;
    }
}

inline bool matchCastBool_F32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::FLOAT32) return false;
    if (inputs[0].getShape() != output.getShape()) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runCastBool_F32_CUDA_ND(const KernelContext &ctx) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    const bool *A = static_cast<const bool *>(ctx.inputs[0]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cast_bool_f32_nd_kernel<<<numBlocks, blockSize, 0, stream>>>(A, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Cast_Bool_F32_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryCastBool_F32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.cast(inputs[0], DType::FLOAT32);
}

REGISTER_KERNEL("Cast_Bool_F32_ND_CUDA", 1, 1, matchCastBool_F32_CUDA_ND, runCastBool_F32_CUDA_ND, refFactoryCastBool_F32_ND_CUDA,{}, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::BOOL}, {{1024}}, {true}, {{MemSpace(2, HandleType::CUDA)}});

#endif