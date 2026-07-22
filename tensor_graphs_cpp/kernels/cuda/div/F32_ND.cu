#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void div_f32_nd_kernel(const float* A, const float* B, float* Out, uint64_t n) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Out[idx] = A[idx] / B[idx];
    }
}

inline bool matchDivF32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::FLOAT32) return false;
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape()) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runDivF32_CUDA_ND(const KernelContext &ctx) {
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    const float *B = static_cast<const float *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    div_f32_nd_kernel<<<numBlocks, blockSize>>>(A, B, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Div_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * Reference Factory
 */
inline LogicalId refFactoryDivF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Divide ND requires 2 inputs");

    return graph.div(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Div_F32_ND_CUDA", 2, 2, matchDivF32_CUDA_ND, runDivF32_CUDA_ND, refFactoryDivF32_ND_CUDA, MemSpace(1, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::FLOAT32, DType::FLOAT32}, {{1024}, {1024}}, {true, true}, {{MemSpace(1, HandleType::CUDA)}, {MemSpace(1, HandleType::CUDA)}});

#endif
