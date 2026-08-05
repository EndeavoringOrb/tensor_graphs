#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__device__ __constant__ float FP4_TABLE_CUDA[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__global__ void cast_e2m1_f32_nd_kernel(const uint8_t* A, float* Out, uint64_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint8_t val = A[idx] & 0x0F;
        Out[idx] = FP4_TABLE_CUDA[val];
    }
}

inline bool matchCastE2M1_F32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::FLOAT32) return false;
    if (inputs[0].getShape() != output.getShape()) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runCastE2M1_F32_CUDA_ND(const KernelContext &ctx) {
    const uint8_t *A = static_cast<const uint8_t *>(ctx.inputs[0]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cast_e2m1_f32_nd_kernel<<<numBlocks, blockSize>>>(A, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Cast_E2M1_F32_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryCastE2M1_F32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.cast(inputs[0], DType::FLOAT32);
}

REGISTER_KERNEL("Cast_E2M1_F32_ND_CUDA", 1, 1, matchCastE2M1_F32_CUDA_ND, runCastE2M1_F32_CUDA_ND, refFactoryCastE2M1_F32_ND_CUDA, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::E2M1}, {{1024}}, {true}, {{MemSpace(2, HandleType::CUDA)}});

#endif