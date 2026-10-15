#pragma once

#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "core/types.hpp"
#include "core/kernels.hpp"

namespace gemma_rms_norm_cuda {

static __global__ void gemmaRmsNormWarpKernel(
    const float *__restrict__ x,
    const float *__restrict__ w,
    float *__restrict__ out,
    uint32_t rows,
    uint32_t cols,
    float eps)
{
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + warp;
    if (row >= rows)
        return;

    const float *rx = x + row * cols;
    float *rout = out + row * cols;

    float sum_sq = 0.0f;
    for (uint32_t c = lane; c < cols; c += 32)
    {
        float val = rx[c];
        sum_sq += val * val;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
    }

    float inv_std;
    if (lane == 0)
    {
        inv_std = rsqrtf((sum_sq / static_cast<float>(cols)) + eps);
    }
    inv_std = __shfl_sync(0xffffffffu, inv_std, 0);

    for (uint32_t c = lane; c < cols; c += 32)
    {
        rout[c] = rx[c] * inv_std * (w[c] + 1.0f);
    }
}

inline void runGemmaRMSNormCUDAImpl(const KernelContext &ctx, const char *kernel_name)
{
    if (ctx.inputs.size() < 2 || ctx.outputs.empty())
    {
        Error::throw_err(std::string(kernel_name) + ": insufficient inputs or outputs");
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    const float *w = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &shape = ctx.inViews[0].getShape();
    if (shape.size() != 3)
    {
        Error::throw_err(std::string(kernel_name) + ": expected 3D input tensor");
    }

    uint32_t rows = shape[0] * shape[1];
    uint32_t cols = shape[2];
    if (rows == 0 || cols == 0)
        return;

    const float eps = 1e-6f;
    const uint32_t warps_per_block = 8;
    const uint32_t block_size = warps_per_block * 32;
    uint32_t grid_size = (rows + warps_per_block - 1) / warps_per_block;

    gemmaRmsNormWarpKernel<<<grid_size, block_size, 0, stream>>>(x, w, out, rows, cols, eps);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        Error::throw_err("CUDA kernel launch failed in " + std::string(kernel_name) + ": " + std::string(cudaGetErrorString(err)));
    }
}

} // namespace gemma_rms_norm_cuda

#endif
