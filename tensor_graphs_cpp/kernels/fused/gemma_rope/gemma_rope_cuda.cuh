#pragma once

#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <vector>
#include "core/types.hpp"
#include "core/kernels.hpp"

namespace gemma_rope_cuda {

static __global__ void gemmaRoPEKernel(
    const float *__restrict__ x,
    const float *__restrict__ cos_tab,
    const float *__restrict__ sin_tab,
    float *__restrict__ out,
    uint32_t n_groups,
    uint32_t seq_len,
    uint32_t head_dim)
{
    const uint32_t half_dim = head_dim >> 1;
    const uint32_t row = blockIdx.x; // in [0, n_groups * seq_len)
    const uint32_t s = row % seq_len;
    const uint32_t t = threadIdx.x;

    const float *row_x = x + static_cast<uint64_t>(row) * head_dim;
    float *row_out = out + static_cast<uint64_t>(row) * head_dim;
    const float *row_cos = cos_tab + static_cast<uint64_t>(s) * head_dim;
    const float *row_sin = sin_tab + static_cast<uint64_t>(s) * head_dim;

    for (uint32_t c = t; c < head_dim; c += blockDim.x)
    {
        float x_val = row_x[c];
        float rot_val = (c < half_dim) ? -row_x[c + half_dim] : row_x[c - half_dim];
        row_out[c] = x_val * row_cos[c] + rot_val * row_sin[c];
    }
}

inline void runGemmaRoPECUDABase(const KernelContext &ctx, const char *kernel_name, uint32_t expected_groups, uint32_t expected_head_dim)
{
    if (ctx.inputs.size() < 3 || ctx.outputs.empty())
    {
        Error::throw_err(std::string(kernel_name) + ": insufficient inputs or outputs");
    }

    const auto &x_shape = ctx.inViews[0].getShape();
    if (x_shape.size() != 3)
    {
        Error::throw_err(std::string(kernel_name) + ": expected 3D input tensor");
    }
    uint32_t n_groups = x_shape[0];
    uint32_t seq_len = x_shape[1];
    uint32_t head_dim = x_shape[2];

    if (n_groups != expected_groups || head_dim != expected_head_dim)
    {
        Error::throw_err(std::string(kernel_name) + ": unexpected shape dimension");
    }

    const float *x = static_cast<const float *>(ctx.inputs[0]);
    const float *cos_tab = static_cast<const float *>(ctx.inputs[1]);
    const float *sin_tab = static_cast<const float *>(ctx.inputs[2]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    uint32_t num_rows = n_groups * seq_len;
    uint32_t block_dim = (head_dim < 256) ? head_dim : 256;

    gemmaRoPEKernel<<<num_rows, block_dim, 0, stream>>>(x, cos_tab, sin_tab, out, n_groups, seq_len, head_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        Error::throw_err(std::string(kernel_name) + " CUDA launch failed: " + cudaGetErrorString(err));
    }
}

} // namespace gemma_rope_cuda
#endif
