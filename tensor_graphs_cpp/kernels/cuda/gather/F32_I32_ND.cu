#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void gather_f32_i32_nd_kernel(const float* data, const int32_t* indices, float* Out, uint64_t vocabSize, uint64_t rowSize, uint64_t numIndices) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = numIndices * rowSize;
    if (idx < total) {
        uint64_t i = idx / rowSize;
        uint64_t r = idx % rowSize;
        int32_t row_idx = indices[i];
        if (row_idx >= 0 && (uint64_t)row_idx < vocabSize) {
            Out[idx] = data[(uint64_t)row_idx * rowSize + r];
        } else {
            Out[idx] = 0.0f;
        }
    }
}

inline bool matchGatherF32_I32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::FLOAT32) return false;
    if (inputs[0].getShape().empty()) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runGatherF32_I32_CUDA_ND(const KernelContext &ctx) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    const float *data = static_cast<const float *>(ctx.inputs[0]);
    const int32_t *indices = static_cast<const int32_t *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const auto &dataShape = ctx.inViews[0].getShape();
    const auto &idxShape = ctx.inViews[1].getShape();

    uint64_t vocabSize = dataShape[0];
    uint64_t rowSize = 1;
    for (uint64_t i = 1; i < dataShape.size(); ++i)
        rowSize *= dataShape[i];

    uint64_t numIndices = countElements(idxShape);
    uint64_t total = numIndices * rowSize;

    if (total == 0) return;

    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    gather_f32_i32_nd_kernel<<<numBlocks, blockSize, 0, stream>>>(data, indices, Out, vocabSize, rowSize, numIndices);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Gather_F32_I32_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryGatherF32_I32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.gather(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Gather_F32_I32_ND_CUDA", 2, 2, matchGatherF32_I32_CUDA_ND, runGatherF32_I32_CUDA_ND, refFactoryGatherF32_I32_ND_CUDA,{}, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::FLOAT32, DType::INT32}, {{8, 32}, {8}}, {true, true}, {{MemSpace(2, HandleType::CUDA)}, {MemSpace(2, HandleType::CUDA)}});

#endif