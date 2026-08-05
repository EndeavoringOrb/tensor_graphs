#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void cast_i32_bool_nd_kernel(const int32_t* A, bool* Out, uint64_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Out[idx] = (A[idx] != 0);
    }
}

inline bool matchCastI32_Bool_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::BOOL) return false;
    if (inputs[0].getShape() != output.getShape()) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runCastI32_Bool_CUDA_ND(const KernelContext &ctx) {
    const int32_t *A = static_cast<const int32_t *>(ctx.inputs[0]);
    bool *Out = static_cast<bool *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cast_i32_bool_nd_kernel<<<numBlocks, blockSize>>>(A, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Cast_I32_Bool_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryCastI32_Bool_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.cast(inputs[0], DType::BOOL);
}

REGISTER_KERNEL("Cast_I32_Bool_ND_CUDA", 1, 1, matchCastI32_Bool_CUDA_ND, runCastI32_Bool_CUDA_ND, refFactoryCastI32_Bool_ND_CUDA, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::INT32}, {{1024}}, {true}, {{MemSpace(2, HandleType::CUDA)}});

#endif