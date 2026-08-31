#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void triu_f32_nd_kernel(const float* A, int32_t k, float* Out, uint64_t rows, uint64_t cols, uint64_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint64_t r = (idx / cols) % rows;
        uint64_t c = idx % cols;
        Out[idx] = ((int64_t)c >= (int64_t)r + k) ? A[idx] : 0.0f;
    }
}

inline bool matchTriuF32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::FLOAT32) return false;
    if (inputs[0].getShape().size() < 2) return false;
    if (inputs[0].getShape() != output.getShape()) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runTriuF32_CUDA_ND(const KernelContext &ctx) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    int32_t k = *static_cast<const int32_t *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const auto &shape = ctx.outViews[0].getShape();
    uint64_t cols = shape.back();
    uint64_t rows = shape[shape.size() - 2];
    uint64_t n = countElements(shape);

    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    triu_f32_nd_kernel<<<numBlocks, blockSize, 0, stream>>>(A, k, Out, rows, cols, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Triu_F32_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryTriuF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.triu(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Triu_F32_ND_CUDA", 2, 2, matchTriuF32_CUDA_ND, runTriuF32_CUDA_ND, refFactoryTriuF32_ND_CUDA, {0}, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::FLOAT32, DType::INT32}, {{8, 32}, {1}}, {true, false}, {{MemSpace(2, HandleType::CUDA)}, {MemSpace(1, HandleType::CPP)}});

#endif