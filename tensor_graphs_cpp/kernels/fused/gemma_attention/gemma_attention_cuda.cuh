#pragma once

#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "core/types.hpp"
#include "core/kernels.hpp"

namespace gemma_attention_cuda {

constexpr uint32_t MAX_SEQ_LEN = 128;

static __global__ void gemmaAttentionKernel(
    const float *__restrict__ q,
    const float *__restrict__ k,
    const float *__restrict__ v,
    const float *__restrict__ mask,
    float *__restrict__ ctx_flat,
    uint32_t n_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    float scale)
{
    __shared__ float s_warp_sums[8];
    __shared__ float s_scores[MAX_SEQ_LEN];
    __shared__ float s_probs[MAX_SEQ_LEN];

    const uint32_t q_row = blockIdx.x; // in [0, n_heads * seq_len)
    const uint32_t h = q_row / seq_len;
    const uint32_t i = q_row % seq_len;
    const uint32_t t = threadIdx.x;
    const uint32_t warp_id = t >> 5;
    const uint32_t lane = t & 31u;

    const float *q_vec = q + static_cast<uint64_t>(h * seq_len + i) * head_dim;
    const float *mask_row = mask + static_cast<uint64_t>(i) * seq_len;
    float *out_row = ctx_flat + static_cast<uint64_t>(i) * (n_heads * head_dim) + (h * head_dim);

    const float q_val = (t < head_dim) ? q_vec[t] * scale : 0.0f;

    for (uint32_t j = 0; j < seq_len; ++j)
    {
        const float *k_vec = k + static_cast<uint64_t>(h * seq_len + j) * head_dim;
        float prod = (t < head_dim) ? q_val * k_vec[t] : 0.0f;

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            prod += __shfl_down_sync(0xffffffffu, prod, offset);
        }

        if (lane == 0)
        {
            s_warp_sums[warp_id] = prod;
        }
        __syncthreads();

        float s = (t < 8) ? s_warp_sums[t] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
        {
            s += __shfl_down_sync(0xffffffffu, s, offset);
        }
        if (t == 0)
        {
            s_scores[j] = s + mask_row[j];
        }
        __syncthreads();
    }

    if (t == 0)
    {
        float max_val = s_scores[0];
        for (uint32_t j = 1; j < seq_len; ++j)
        {
            if (s_scores[j] > max_val)
                max_val = s_scores[j];
        }

        float sum_exp = 0.0f;
        for (uint32_t j = 0; j < seq_len; ++j)
        {
            float exp_val = __expf(s_scores[j] - max_val);
            s_probs[j] = exp_val;
            sum_exp += exp_val;
        }

        float inv_sum = 1.0f / sum_exp;
        for (uint32_t j = 0; j < seq_len; ++j)
        {
            s_probs[j] *= inv_sum;
        }
    }
    __syncthreads();

    if (t < head_dim)
    {
        float acc = 0.0f;
        for (uint32_t j = 0; j < seq_len; ++j)
        {
            const float *v_vec = v + static_cast<uint64_t>(h * seq_len + j) * head_dim;
            acc += s_probs[j] * v_vec[t];
        }
        out_row[t] = acc;
    }
}

inline void runGemmaAttentionCUDAImpl(const KernelContext &ctx)
{
    if (ctx.inputs.size() < 4 || ctx.outputs.empty())
    {
        Error::throw_err("Gemma_Attention_CUDA: insufficient inputs or outputs");
    }

    const auto &q_shape = ctx.inViews[0].getShape();
    if (q_shape.size() != 3)
    {
        Error::throw_err("Gemma_Attention_CUDA: expected 3D input tensor for Q");
    }

    uint32_t n_heads = q_shape[0];
    uint32_t seq_len = q_shape[1];
    uint32_t head_dim = q_shape[2];

    if (n_heads != 4 || head_dim != 256)
    {
        Error::throw_err("Gemma_Attention_CUDA: expected n_heads=4, head_dim=256");
    }
    if (seq_len > MAX_SEQ_LEN)
    {
        Error::throw_err("Gemma_Attention_CUDA: seq_len exceeds MAX_SEQ_LEN");
    }

    const float *q = static_cast<const float *>(ctx.inputs[0]);
    const float *k = static_cast<const float *>(ctx.inputs[1]);
    const float *v = static_cast<const float *>(ctx.inputs[2]);
    const float *mask = static_cast<const float *>(ctx.inputs[3]);
    float *ctx_flat = static_cast<float *>(ctx.outputs[0]);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    uint32_t num_blocks = n_heads * seq_len;
    uint32_t block_dim = 256;

    gemmaAttentionKernel<<<num_blocks, block_dim, 0, stream>>>(q, k, v, mask, ctx_flat, n_heads, seq_len, head_dim, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        Error::throw_err("Gemma_Attention_CUDA launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

} // namespace gemma_attention_cuda
#endif
