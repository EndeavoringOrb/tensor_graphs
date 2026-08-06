#ifdef TG_USE_CUDA
#pragma once
#include <cuda_runtime.h>
#include "core/types.hpp"
#include "core/kernels.hpp"

__global__ void clamp_f32_nd_kernel(const float* A, float min_val, float max_val, float* Out, uint64_t n) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = A[idx];
        Out[idx] = fminf(fmaxf(val, min_val), max_val);
    }
}

inline bool matchClampF32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (inputs[0].getShape() != output.getShape()) return false;
    if (inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != 1) return false;
    if (inputs[2].getShape().size() != 1 || inputs[2].getShape()[0] != 1) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runClampF32_CUDA_ND(const KernelContext &ctx) {
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    float min_val = *static_cast<const float *>(ctx.inputs[1]);
    float max_val = *static_cast<const float *>(ctx.inputs[2]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0) return;

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    clamp_f32_nd_kernel<<<numBlocks, blockSize>>>(A, min_val, max_val, Out, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Clamp_F32_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * Reference Factory
 * Reconstructs the exact atomic subgraph created by DeepSeekV4FlashModel::clamp:
 *   1. clamp_max:
 *      max_node = g.fill(max_val, shape)
 *      is_less = g.lt(x, max_node)
 *      is_less_f = g.cast(is_less, DType::FLOAT32)
 *      not_less_f = g.add(g.fill(1.0f, shape), g.neg(is_less_f))
 *      clamp_max_res = g.add(g.mul(x, is_less_f), g.mul(max_node, not_less_f))
 * 
 *   2. clamp_min:
 *      min_node = g.fill(min_val, shape)
 *      is_greater = g.lt(min_node, clamp_max_res)
 *      is_greater_f = g.cast(is_greater, DType::FLOAT32)
 *      not_greater_f = g.add(g.fill(1.0f, shape), g.neg(is_greater_f))
 *      clamp_res = g.add(g.mul(clamp_max_res, is_greater_f), g.mul(min_node, not_greater_f))
 */
inline LogicalId refFactoryClamp_F32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x = inputs[0];
    LogicalId min_val = inputs[1];
    LogicalId max_val = inputs[2];

    const auto &shape = graph.getNode(x).getShape();

    // --- clamp_max ---
    LogicalId max_node = graph.fill(max_val, shape);
    LogicalId is_less = graph.lt(x, max_node);
    LogicalId is_less_f = graph.cast(is_less, DType::FLOAT32);

    float one_val = 1.0f;
    LogicalId fill_one_1 = graph.fill(one_val, shape);
    LogicalId not_less_f = graph.add(fill_one_1, graph.neg(is_less_f));

    LogicalId clamp_max_res = graph.add(graph.mul(x, is_less_f), graph.mul(max_node, not_less_f));

    // --- clamp_min ---
    LogicalId min_node = graph.fill(min_val, shape);
    LogicalId is_greater = graph.lt(min_node, clamp_max_res);
    LogicalId is_greater_f = graph.cast(is_greater, DType::FLOAT32);

    LogicalId fill_one_2 = graph.fill(one_val, shape);
    LogicalId not_greater_f = graph.add(fill_one_2, graph.neg(is_greater_f));

    return graph.add(graph.mul(clamp_max_res, is_greater_f), graph.mul(min_node, not_greater_f));
}

REGISTER_KERNEL("Clamp_F32_ND_CUDA", 3, 3, matchClampF32_CUDA_ND, runClampF32_CUDA_ND, refFactoryClamp_F32_ND_CUDA,
                MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
                {{1024}, {1}, {1}}, {true, false, false},
                {{MemSpace(2, HandleType::CUDA)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif