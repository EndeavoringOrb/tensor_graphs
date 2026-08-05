#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void not_bool_nd_kernel(const bool* A, bool* Out, uint64_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Out[idx] = !A[idx];
    }
}

inline bool matchNotBool_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::BOOL) return false;
    if (inputs[0].getShape() != output.getShape()) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runNotBool_CUDA_ND(const KernelContext &ctx) {
    const bool *A = static_cast<const bool *>(ctx.inputs[0]);
    bool *Out = static_cast<bool *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    not_bool_nd_kernel<<<numBlocks, blockSize>>>(A, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Not_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryNotBool_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.logical_not(inputs[0]);
}

REGISTER_KERNEL("Not_Bool_ND_CUDA", 1, 1, matchNotBool_CUDA_ND, runNotBool_CUDA_ND, refFactoryNotBool_ND_CUDA, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::BOOL}, {{1024}}, {true}, {{MemSpace(2, HandleType::CUDA)}});

#endif