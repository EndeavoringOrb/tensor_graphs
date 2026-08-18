#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void eq_i32_nd_kernel(const int32_t* A, const int32_t* B, bool* Out, uint64_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Out[idx] = (A[idx] == B[idx]);
    }
}

inline bool matchEqI32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::BOOL) return false;
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape()) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runEqI32_CUDA_ND(const KernelContext &ctx) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    const int32_t *A = static_cast<const int32_t *>(ctx.inputs[0]);
    const int32_t *B = static_cast<const int32_t *>(ctx.inputs[1]);
    bool *Out = static_cast<bool *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    eq_i32_nd_kernel<<<numBlocks, blockSize, 0, stream>>>(A, B, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Eq_I32_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryEqI32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.eq(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Eq_I32_ND_CUDA", 2, 2, matchEqI32_CUDA_ND, runEqI32_CUDA_ND, refFactoryEqI32_ND_CUDA,{0,1}, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::INT32, DType::INT32}, {{1024}, {1024}}, {true, true}, {{MemSpace(2, HandleType::CUDA)}, {MemSpace(2, HandleType::CUDA)}});

#endif