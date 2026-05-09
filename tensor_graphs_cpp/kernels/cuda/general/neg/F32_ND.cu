#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void neg_f32_nd_kernel(const float* A, float* Out, uint64_t n) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Out[idx] = -A[idx];
    }
}

inline bool matchNegF32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (inputs.size() != 1) return false;
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32) return false;
    if (inputs[0].getShape() != output.getShape()) return false;
    if (!isContiguous(inputs[0]) || !isContiguous(output)) return false;
    return true;
}

inline void runNegF32_CUDA_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews) {
    const float *A = static_cast<const float *>(inputs[0]);
    float *Out = static_cast<float *>(outputs[0]);

    uint64_t n = countElements(outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    neg_f32_nd_kernel<<<numBlocks, blockSize>>>(A, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Neg_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * Reference Factory
 */
inline uint32_t refFactoryNegF32_ND_CUDA(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 1)
        Error::throw_err("Negate ND requires 1 input");

    return graph.neg(inputs[0]);
}

REGISTER_KERNEL("Neg_F32_ND_CUDA", 1, matchNegF32_CUDA_ND, runNegF32_CUDA_ND, refFactoryNegF32_ND_CUDA, {Backend::CUDA}, {DType::FLOAT32}, {{1024}}, {true}, {{Backend::CUDA}});

#endif
