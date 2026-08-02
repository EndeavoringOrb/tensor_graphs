#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void sum_f32_nd_kernel(const float* A, float* Out, uint64_t O, uint64_t R, uint64_t I) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < O * I) {
        uint64_t o = idx / I;
        uint64_t i = idx % I;
        float sum = 0.0f;
        for (uint64_t r = 0; r < R; ++r) {
            sum += A[o * (R * I) + r * I + i];
        }
        Out[idx] = sum;
    }
}

inline bool matchSumF32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::FLOAT32) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runSumF32_CUDA_ND(const KernelContext &ctx) {
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    auto inShape = ctx.inViews[0].getShape();
    if (axis < 0) axis += inShape.size();

    uint64_t O = 1, R = 1, I = 1;
    for (int i = 0; i < inShape.size(); ++i) {
        if (i < axis) O *= inShape[i];
        else if (i == axis) R = inShape[i];
        else I *= inShape[i];
    }

    uint64_t n = O * I;
    if (n == 0) return;
    
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sum_f32_nd_kernel<<<numBlocks, blockSize>>>(A, Out, O, R, I);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Sum_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * Reference Factory
 */
inline LogicalId refFactorySumF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Sum ND requires 2 inputs");

    return graph.sum(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Sum_F32_ND_CUDA", 2, 2, matchSumF32_CUDA_ND, runSumF32_CUDA_ND, refFactorySumF32_ND_CUDA, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::FLOAT32, DType::INT32}, {{1024, 1024}, {1}}, {true, false}, {{MemSpace(2, HandleType::CUDA)}, {MemSpace(1, HandleType::CPP)}});

#endif
