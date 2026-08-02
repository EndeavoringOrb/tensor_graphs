#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void concat_f32_nd_kernel(const float* A, float* Out, uint64_t O, uint64_t C_in, uint64_t C_out, uint64_t I, uint64_t C_offset) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = O * C_in * I;
    if (idx >= total) return;

    uint64_t i = idx % I;
    uint64_t c = (idx / I) % C_in;
    uint64_t o = idx / (I * C_in);

    uint64_t out_idx = o * (C_out * I) + (C_offset + c) * I + i;
    Out[out_idx] = A[idx];
}

inline bool matchConcatF32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::FLOAT32) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runConcatF32_CUDA_ND(const KernelContext &ctx) {
    float *Out = static_cast<float *>(ctx.outputs[0]);
    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[0]);
    
    auto outShape = ctx.outViews[0].getShape();
    if (axis < 0) axis += outShape.size();

    uint64_t O = 1, C_out = outShape[axis], I = 1;
    for (int i = 0; i < outShape.size(); ++i) {
        if (i < axis) O *= outShape[i];
        else if (i > axis) I *= outShape[i];
    }

    uint64_t c_offset = 0;
    int blockSize = 256;

    for (uint64_t n = 1; n < ctx.inputs.size(); ++n) {
        const float *A = static_cast<const float *>(ctx.inputs[n]);
        uint64_t C_in = ctx.inViews[n].getShape()[axis];
        uint64_t total = O * C_in * I;
        if (total > 0) {
            int numBlocks = (total + blockSize - 1) / blockSize;
            concat_f32_nd_kernel<<<numBlocks, blockSize>>>(A, Out, O, C_in, C_out, I, c_offset);
        }
        c_offset += C_in;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Concat_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * Reference Factory
 */
inline LogicalId refFactoryConcatF32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() < 2)
        Error::throw_err("Concat ND requires at least 2 inputs");

    std::vector<LogicalId> tensors(inputs.begin() + 1, inputs.end());
    LogicalId axis = inputs[0];
    return graph.concat(tensors, axis);
}

REGISTER_KERNEL("Concat_F32_ND_CUDA", 2, UINT32_MAX, matchConcatF32_CUDA_ND, runConcatF32_CUDA_ND, refFactoryConcatF32_ND_CUDA, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::INT32, DType::FLOAT32}, {{1}, {1024}}, {false, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(2, HandleType::CUDA)}});

#endif
