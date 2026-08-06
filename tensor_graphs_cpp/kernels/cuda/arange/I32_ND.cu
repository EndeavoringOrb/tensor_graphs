#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void arange_i32_nd_kernel(int32_t start, int32_t step, int32_t* Out, uint64_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Out[idx] = start + (int32_t)idx * step;
    }
}

inline bool matchArangeI32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::INT32) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runArangeI32_CUDA_ND(const KernelContext &ctx) {
    int32_t start = *static_cast<const int32_t *>(ctx.inputs[0]);
    int32_t step = *static_cast<const int32_t *>(ctx.inputs[2]);
    int32_t *Out = static_cast<int32_t *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    arange_i32_nd_kernel<<<numBlocks, blockSize>>>(start, step, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Arange_I32_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryArangeI32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.arange(inputs[0], inputs[1], inputs[2]);
}

REGISTER_KERNEL("Arange_I32_ND_CUDA", 3, 3, matchArangeI32_CUDA_ND, runArangeI32_CUDA_ND, refFactoryArangeI32_ND_CUDA, {},MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::INT32, DType::INT32, DType::INT32}, {{1}, {1}, {1}}, {false, false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif