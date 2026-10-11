#include "kernels/fused/softmax/ref.hpp"

#ifdef TG_USE_CUDA
#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
#include <cuda_runtime.h>

__global__ void softmax_f32_3d_cuda_kernel(const float *input, float *output)
{
    constexpr unsigned int row_size = 2340;
    constexpr unsigned int values_per_thread = 10;
    constexpr unsigned int block_size = 256;

    const unsigned int lane = threadIdx.x;
    const unsigned int row = blockIdx.x;
    const unsigned int base = row * row_size;
    float values[values_per_thread] = {};
    __shared__ float reduction[block_size];

    float row_max = -3.402823466e+38f;
    #pragma unroll
    for (unsigned int i = 0; i < values_per_thread; ++i)
    {
        const unsigned int column = lane + i * blockDim.x;
        if (column < row_size)
        {
            const float value = input[base + column];
            values[i] = value;
            row_max = fmaxf(row_max, value);
        }
    }

    reduction[lane] = row_max;
    __syncthreads();
    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
        __syncthreads();
    }
    row_max = reduction[0];

    float row_sum = 0.0f;
    #pragma unroll
    for (unsigned int i = 0; i < values_per_thread; ++i)
    {
        const unsigned int column = lane + i * blockDim.x;
        if (column < row_size)
        {
            values[i] = __expf(values[i] - row_max);
            row_sum += values[i];
        }
    }

    reduction[lane] = row_sum;
    __syncthreads();
    for (unsigned int stride = block_size / 2; stride > 0; stride >>= 1)
    {
        if (lane < stride)
            reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    row_sum = reduction[0];

    const float inverse_sum = 1.0f / row_sum;
    #pragma unroll
    for (unsigned int i = 0; i < values_per_thread; ++i)
    {
        const unsigned int column = lane + i * blockDim.x;
        if (column < row_size)
            output[base + column] = values[i] * inverse_sum;
    }
}

inline bool matchSoftmaxF32_3D_CUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const std::vector<uint32_t> shape = {12, 2340, 2340};
    return inputs[0].dtype == DType::FLOAT32 && output.dtype == DType::FLOAT32 &&
           inputs[0].getShape() == shape && output.getShape() == shape;
}

inline void runSoftmaxF32_3D_CUDA(const KernelContext &ctx)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    constexpr unsigned int rows = 12 * 2340;
    constexpr unsigned int block_size = 256;
    softmax_f32_3d_cuda_kernel<<<rows, block_size, 0, stream>>>(
        static_cast<const float *>(ctx.inputs[0]), static_cast<float *>(ctx.outputs[0]));

    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
        Error::throw_err("CUDA kernel launch failed in Softmax_F32_3D_CUDA: " +
                         std::string(cudaGetErrorString(error)));
}

REGISTER_KERNEL("Softmax_F32_3D_CUDA", 1, 1, matchSoftmaxF32_3D_CUDA, runSoftmaxF32_3D_CUDA,
                refFactorySoftmax, {0}, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)},
                {DType::FLOAT32}, {{12, 2340, 2340}}, {true}, {{MemSpace(2, HandleType::CUDA)}});

#endif
