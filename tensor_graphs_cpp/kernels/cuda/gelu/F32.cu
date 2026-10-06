#ifdef TG_USE_CUDA
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "core/common/constants.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

__global__ void gelu_tanh_fill_f32_cuda_kernel(const float *__restrict__ in, float *__restrict__ out, uint64_t n)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = idx; i < n; i += stride)
    {
        float x = in[i];
        float x3 = x * x * x;
        float inner = 0.79788456f * (x + 0.044715f * x3);
        float tanh_val = tanhf(inner);
        out[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

__global__ void gelu_tanh_fill_f32_vec4_cuda_kernel(const float4 *__restrict__ in, float4 *__restrict__ out, uint64_t n4)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    for (uint64_t i = idx; i < n4; i += stride)
    {
        float4 v = in[i];
        float4 r;

#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            float x = (k == 0) ? v.x : (k == 1) ? v.y : (k == 2) ? v.z : v.w;
            float x3 = x * x * x;
            float inner = 0.79788456f * (x + 0.044715f * x3);
            float tanh_val = tanhf(inner);
            float res = 0.5f * x * (1.0f + tanh_val);
            if (k == 0)
                r.x = res;
            else if (k == 1)
                r.y = res;
            else if (k == 2)
                r.z = res;
            else
                r.w = res;
        }
        out[i] = r;
    }
}

inline bool matchGeluTanhFill_CUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (output.dtype != DType::FLOAT32)
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runGeluTanhFill_CUDA(const KernelContext &ctx)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    uint64_t n = countElements(ctx.outViews[0].getShape());
    if (n == 0)
        return;

    int blockSize = 256;
    if ((reinterpret_cast<uintptr_t>(in) % 16 == 0) && (reinterpret_cast<uintptr_t>(out) % 16 == 0) && (n % 4 == 0))
    {
        uint64_t n4 = n / 4;
        int numBlocks = (n4 + blockSize - 1) / blockSize;
        if (numBlocks > 65535)
            numBlocks = 65535;
        gelu_tanh_fill_f32_vec4_cuda_kernel<<<numBlocks, blockSize, 0, stream>>>(
            reinterpret_cast<const float4 *>(in), reinterpret_cast<float4 *>(out), n4);
    }
    else
    {
        int numBlocks = (n + blockSize - 1) / blockSize;
        if (numBlocks > 65535)
            numBlocks = 65535;
        gelu_tanh_fill_f32_cuda_kernel<<<numBlocks, blockSize, 0, stream>>>(in, out, n);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        Error::throw_err("CUDA kernel launch failed in Gelu_Tanh_Fill_CUDA: " + std::string(cudaGetErrorString(err)));
    }
}

inline LogicalId refFactoryGeluTanhFill_CUDA(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    auto shape = g.getNode(x).getShape();

    LogicalId x_sq = g.mul(x, x);
    LogicalId x_cube = g.mul(x_sq, x);
    LogicalId c1 = g.fill(0.044715f, shape);
    LogicalId term1 = g.mul(x_cube, c1);
    LogicalId term2 = g.add(x, term1);
    LogicalId c2 = g.fill(0.79788456f, shape);
    LogicalId inner = g.mul(term2, c2);

    LogicalId neg_two = g.fill(-2.0f, shape);
    LogicalId neg_2u = g.mul(inner, neg_two);
    LogicalId exp_neg_2u = g.pow(g.fill(TGConstants::E, shape), neg_2u);
    LogicalId one = g.fill(1.0f, shape);
    LogicalId two = g.fill(2.0f, shape);
    LogicalId den = g.add(one, exp_neg_2u);
    LogicalId tanh_val = g.add(g.div(two, den), g.neg(one));

    LogicalId one_plus_tanh = g.add(one, tanh_val);
    LogicalId half_x = g.mul(x, g.fill(0.5f, shape));
    return g.mul(half_x, one_plus_tanh);
}

REGISTER_KERNEL("Gelu_Tanh_Fill_CUDA", 1, 1, matchGeluTanhFill_CUDA, runGeluTanhFill_CUDA, refFactoryGeluTanhFill_CUDA,
                {0}, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)}, {DType::FLOAT32},
                {{1, 128, 2048}}, {true}, {{MemSpace(2, HandleType::CUDA)}});

#endif // TG_USE_CUDA
