#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void mul_f32_nd_kernel(const float* A, const float* B, float* Out, uint64_t n) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Out[idx] = A[idx] * B[idx];
    }
}

inline bool matchMulF32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::FLOAT32) return false;
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape()) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runMulF32_CUDA_ND(const KernelContext &ctx) {
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    const float *B = static_cast<const float *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    mul_f32_nd_kernel<<<numBlocks, blockSize>>>(A, B, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Mul_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * Reference Factory
 */
inline LogicalId refFactoryMulF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Mul ND requires 2 inputs");

    return graph.mul(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Mul_F32_ND_CUDA", 2, 2, matchMulF32_CUDA_ND, runMulF32_CUDA_ND, refFactoryMulF32_ND_CUDA, MemSpace(1, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::FLOAT32, DType::FLOAT32}, {{1024}, {1024}}, {true, true}, {{MemSpace(1, HandleType::CUDA)}, {MemSpace(1, HandleType::CUDA)}});

#endif
