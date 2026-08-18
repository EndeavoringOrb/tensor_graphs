#ifdef TG_USE_CUDA
#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

/**
 * Fused CUDA Kernel for DeepSeekV4FlashModel::hc_post
 * Computes:
 *   Out[0, s, m, d] = Post[0, s, m] * X[0, s, d] + sum_{k=0}^{M-1} (Comb[0, s, m, k] * Residual[0, s, k, d])
 *
 * Input shapes:
 *   X:        [1, S, D]
 *   Residual: [1, S, M, D]
 *   Post:     [1, S, M]
 *   Comb:     [1, S, M, M]
 * Output shape:
 *   Out:      [1, S, M, D]
 */
__global__ void hc_post_f32_cuda_kernel(const float *__restrict__ X,
                                        const float *__restrict__ Residual,
                                        const float *__restrict__ Post,
                                        const float *__restrict__ Comb,
                                        float *__restrict__ Out,
                                        uint32_t S, uint32_t M, uint32_t D,
                                        uint64_t total_elements)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
    {
        uint32_t d = idx % D;
        uint32_t temp = idx / D;
        uint32_t m = temp % M;
        uint32_t s = temp / M;

        float x_val = X[s * D + d];
        float post_val = Post[s * M + m];
        float term1 = post_val * x_val;

        float term2_sum = 0.0f;
        uint32_t comb_base = s * M * M + m * M;
        uint32_t res_base = s * M * D + d;

        for (uint32_t k = 0; k < M; ++k)
        {
            term2_sum += Comb[comb_base + k] * Residual[res_base + k * D];
        }

        Out[idx] = term1 + term2_sum;
    }
}

inline bool matchHcPost_CUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &sX = inputs[0].getShape();
    const auto &sRes = inputs[1].getShape();
    const auto &sPost = inputs[2].getShape();
    const auto &sComb = inputs[3].getShape();
    const auto &sOut = output.getShape();

    if (sX.size() != 3 || sRes.size() != 4 || sPost.size() != 3 || sComb.size() != 4 || sOut.size() != 4)
        return false;

    uint32_t S = sX[1];
    uint32_t D = sX[2];
    uint32_t M = sRes[2];

    if (sX[0] != 1 || sRes[0] != 1 || sPost[0] != 1 || sComb[0] != 1 || sOut[0] != 1)
        return false;

    if (sRes[1] != S || sRes[3] != D)
        return false;

    if (sPost[1] != S || sPost[2] != M)
        return false;

    if (sComb[1] != S || sComb[2] != M || sComb[3] != M)
        return false;

    if (sOut[1] != S || sOut[2] != M || sOut[3] != D)
        return false;

    if (!isContiguous(output))
        return false;

    return true;
}

inline void runHcPost_CUDA(const KernelContext &ctx)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const float *Residual = static_cast<const float *>(ctx.inputs[1]);
    const float *Post = static_cast<const float *>(ctx.inputs[2]);
    const float *Comb = static_cast<const float *>(ctx.inputs[3]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const auto &shapeX = ctx.inViews[0].getShape();
    const auto &shapeRes = ctx.inViews[1].getShape();

    uint32_t S = shapeX[1];
    uint32_t D = shapeX[2];
    uint32_t M = shapeRes[2];

    uint64_t total_elements = (uint64_t)S * M * D;
    if (total_elements == 0)
        return;

    int blockSize = 256;
    int numBlocks = (total_elements + blockSize - 1) / blockSize;

    hc_post_f32_cuda_kernel<<<numBlocks, blockSize, 0, stream>>>(X, Residual, Post, Comb, Out, S, M, D, total_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        Error::throw_err("CUDA kernel launch failed in HcPost_CUDA: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * Reference Factory: Exactly mirrors DeepSeekV4FlashModel::hc_post graph structure.
 */
inline LogicalId refFactoryHcPost(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x = inputs[0];
    LogicalId residual = inputs[1];
    LogicalId post = inputs[2];
    LogicalId comb = inputs[3];

    auto shapeX = graph.getNode(x).getShape();
    auto shapeRes = graph.getNode(residual).getShape();

    uint32_t seq_len = shapeX[1];
    uint32_t dim = shapeX[2];
    uint32_t hc_mult = shapeRes[2];

    int32_t sh4_x[] = {1, (int32_t)seq_len, 1, (int32_t)dim};
    LogicalId x_exp = graph.repeat(graph.reshape(x, graph.constant({4}, sh4_x, DType::INT32)), hc_mult, 2);

    int32_t sh4_p[] = {1, (int32_t)seq_len, (int32_t)hc_mult, 1};
    LogicalId post_exp = graph.repeat(graph.reshape(post, graph.constant({4}, sh4_p, DType::INT32)), dim, 3);

    LogicalId term1 = graph.mul(post_exp, x_exp);

    int32_t sh5_c[] = {1, (int32_t)seq_len, (int32_t)hc_mult, (int32_t)hc_mult, 1};
    LogicalId comb_exp = graph.repeat(graph.reshape(comb, graph.constant({5}, sh5_c, DType::INT32)), dim, 4);

    int32_t sh5_r[] = {1, (int32_t)seq_len, 1, (int32_t)hc_mult, (int32_t)dim};
    LogicalId res_exp = graph.repeat(graph.reshape(residual, graph.constant({5}, sh5_r, DType::INT32)), hc_mult, 2);

    int32_t ax_3 = 3;
    LogicalId term2_sum = graph.sum(graph.mul(comb_exp, res_exp), graph.constant({1}, &ax_3, DType::INT32));

    int32_t sh4_out[] = {1, (int32_t)seq_len, (int32_t)hc_mult, (int32_t)dim};
    return graph.add(term1, graph.reshape(term2_sum, graph.constant({4}, sh4_out, DType::INT32)));
}

REGISTER_KERNEL(
    "HcPost_F32_CUDA",
    4, 4,
    matchHcPost_CUDA,
    runHcPost_CUDA,
    refFactoryHcPost,
    {},
    MemSpace(2, HandleType::CUDA),
    {Engine(0, EngineType::CUDA_GPU)},
    {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
    {{1, 8, 4096}, {1, 8, 4, 4096}, {1, 8, 4}, {1, 8, 4, 4}},
    {true, true, true, true},
    {{MemSpace(2, HandleType::CUDA)},
     {MemSpace(2, HandleType::CUDA)},
     {MemSpace(2, HandleType::CUDA)},
     {MemSpace(2, HandleType::CUDA)}});

#endif