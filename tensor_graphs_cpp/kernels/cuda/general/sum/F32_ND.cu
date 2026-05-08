#ifdef USE_CUDA
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
    if (inputs.size() != 2) return false;
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32 || inputs[1].dtype != DType::INT32) return false;
    if (!isContiguous(inputs[0]) || !isContiguous(output)) return false;
    return true;
}

inline void runSumF32_CUDA_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews) {
    const float *A = static_cast<const float *>(inputs[0]);
    int32_t axis = *static_cast<const int32_t *>(inputs[1]);
    float *Out = static_cast<float *>(outputs[0]);

    auto inShape = inViews[0].getShape();
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

REGISTER_REF_KERNEL(OpType::SUM, 2, matchSumF32_CUDA_ND, runSumF32_CUDA_ND, {Backend::CUDA}, {DType::FLOAT32, DType::INT32}, {{1024, 1024}, {1}}, {true, false}, {{Backend::CUDA}, {Backend::CPU}});

#endif
