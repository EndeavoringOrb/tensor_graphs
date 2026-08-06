#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void eq_f32_nd_kernel(const float* A, const float* B, bool* Out, uint64_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Out[idx] = (A[idx] == B[idx]);
    }
}

inline bool matchEqF32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::BOOL) return false;
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape()) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runEqF32_CUDA_ND(const KernelContext &ctx) {
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    const float *B = static_cast<const float *>(ctx.inputs[1]);
    bool *Out = static_cast<bool *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    eq_f32_nd_kernel<<<numBlocks, blockSize>>>(A, B, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Eq_F32_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryEqF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.eq(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Eq_F32_ND_CUDA", 2, 2, matchEqF32_CUDA_ND, runEqF32_CUDA_ND, refFactoryEqF32_ND_CUDA,{0,1}, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::FLOAT32, DType::FLOAT32}, {{1024}, {1024}}, {true, true}, {{MemSpace(2, HandleType::CUDA)}, {MemSpace(2, HandleType::CUDA)}});

#endif