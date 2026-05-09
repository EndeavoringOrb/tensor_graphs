#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>
#include <math_functions.h>

__global__ void pow_f32_nd_kernel(const float *A, const float *B, float *Out, uint64_t n)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        Out[idx] = powf(A[idx], B[idx]);
    }
}

inline bool matchPowF32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runPowF32_CUDA_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *A = static_cast<const float *>(inputs[0]);
    const float *B = static_cast<const float *>(inputs[1]);
    float *Out = static_cast<float *>(outputs[0]);

    uint64_t n = countElements(outViews[0].getShape());
    if (n == 0)
        return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    pow_f32_nd_kernel<<<numBlocks, blockSize>>>(A, B, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        Error::throw_err("CUDA kernel launch failed in Pow_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * Reference Factory
 */
inline uint32_t refFactoryPowF32_ND_CUDA(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Pow ND requires 2 inputs");

    return graph.pow(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Pow_F32_ND_CUDA", 2, matchPowF32_CUDA_ND, runPowF32_CUDA_ND, refFactoryPowF32_ND_CUDA, {Backend::CUDA}, {DType::FLOAT32, DType::FLOAT32}, {{1024}, {1024}}, {true, true}, {{Backend::CUDA}, {Backend::CUDA}});

#endif
