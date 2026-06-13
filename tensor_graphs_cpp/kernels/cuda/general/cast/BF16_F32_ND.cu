#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void cast_bf16_f32_nd_kernel(const uint16_t* A, float* Out, uint64_t n) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t bits = static_cast<uint32_t>(A[idx]) << 16;
        Out[idx] = __uint_as_float(bits);
    }
}

inline bool matchCastBF16_F32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::FLOAT32) return false;
    if (inputs[0].getShape() != output.getShape()) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runCastBF16_F32_CUDA_ND(const KernelContext &ctx) {
    const uint16_t *A = static_cast<const uint16_t *>(ctx.inputs[0]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    cast_bf16_f32_nd_kernel<<<numBlocks, blockSize>>>(A, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Cast_BF16_F32_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * Reference Factory
 */
inline uint32_t refFactoryCastBF16_F32_ND_CUDA(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 1)
        Error::throw_err("Cast BF16->F32 ND requires 1 input");

    return graph.cast(inputs[0], DType::FLOAT32);
}

REGISTER_KERNEL("Cast_BF16_F32_ND_CUDA", 1, matchCastBF16_F32_CUDA_ND, runCastBF16_F32_CUDA_ND, refFactoryCastBF16_F32_ND_CUDA, {Backend::CUDA}, {DType::BF16}, {{1024}}, {true}, {{Backend::CUDA}});

#endif
