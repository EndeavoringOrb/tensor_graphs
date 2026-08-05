#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void cast_bool_i32_nd_kernel(const bool* A, int32_t* Out, uint64_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Out[idx] = A[idx] ? 1 : 0;
    }
}

inline bool matchCastBool_I32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::INT32) return false;
    if (inputs[0].getShape() != output.getShape()) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runCastBool_I32_CUDA_ND(const KernelContext &ctx) {
    const bool *A = static_cast<const bool *>(ctx.inputs[0]);
    int32_t *Out = static_cast<int32_t *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cast_bool_i32_nd_kernel<<<numBlocks, blockSize>>>(A, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Cast_Bool_I32_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryCastBool_I32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.cast(inputs[0], DType::INT32);
}

REGISTER_KERNEL("Cast_Bool_I32_ND_CUDA", 1, 1, matchCastBool_I32_CUDA_ND, runCastBool_I32_CUDA_ND, refFactoryCastBool_I32_ND_CUDA, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::BOOL}, {{1024}}, {true}, {{MemSpace(2, HandleType::CUDA)}});

#endif