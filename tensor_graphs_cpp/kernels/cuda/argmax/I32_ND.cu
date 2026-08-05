#ifdef TG_USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void argmax_i32_nd_kernel(const float* in, int32_t* Out, uint64_t outer, uint64_t mid, uint64_t inner, int32_t k) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < outer * inner) {
        uint64_t o = idx / inner;
        uint64_t i = idx % inner;

        for (int32_t j = 0; j < k; ++j) {
            float max_val = -3.402823466e+38f;
            int32_t max_idx = -1;

            for (uint64_t m = 0; m < mid; ++m) {
                bool already_selected = false;
                for (int32_t prev = 0; prev < j; ++prev) {
                    uint64_t prev_out_idx = (o * k + prev) * inner + i;
                    if (Out[prev_out_idx] == (int32_t)m) {
                        already_selected = true;
                        break;
                    }
                }
                if (already_selected) continue;

                uint64_t in_idx = (o * mid + m) * inner + i;
                float val = in[in_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = (int32_t)m;
                }
            }

            uint64_t out_idx = (o * k + j) * inner + i;
            Out[out_idx] = max_idx;
        }
    }
}

inline bool matchArgmaxI32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (output.dtype != DType::INT32) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runArgmaxI32_CUDA_ND(const KernelContext &ctx) {
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[1]);
    int32_t k = *static_cast<const int32_t *>(ctx.inputs[2]);
    int32_t *Out = static_cast<int32_t *>(ctx.outputs[0]);

    const auto &inShape = ctx.inViews[0].getShape();
    int ndim = static_cast<int>(inShape.size());
    if (axis < 0) axis += ndim;

    uint64_t outer = 1, mid = inShape[axis], inner = 1;
    for (int i = 0; i < axis; ++i) outer *= inShape[i];
    for (int i = axis + 1; i < ndim; ++i) inner *= inShape[i];

    uint64_t total = outer * inner;
    if (total == 0) return;

    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    argmax_i32_nd_kernel<<<numBlocks, blockSize>>>(in, Out, outer, mid, inner, k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Argmax_I32_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryArgmaxI32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.argmax(inputs[0], inputs[1], inputs[2]);
}

REGISTER_KERNEL("Argmax_I32_ND_CUDA", 3, 3, matchArgmaxI32_CUDA_ND, runArgmaxI32_CUDA_ND, refFactoryArgmaxI32_ND_CUDA, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::FLOAT32, DType::INT32, DType::INT32}, {{8, 32}, {1}, {1}}, {true, false, false}, {{MemSpace(2, HandleType::CUDA)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif