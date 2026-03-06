#ifdef USE_CUDA
#include "tensor_graphs_cpp/kernels/cuda/general/dot/F32_3D.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void dot_f32_3d_kernel(const float* A, const float* B, float* Out, 
                                  uint32_t B_count, uint32_t M, uint32_t K, uint32_t N) 
{
    uint32_t b = blockIdx.z;
    uint32_t m = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < B_count && m < M && n < N) {
        float sum = 0.0f;
        for (uint32_t k = 0; k < K; ++k) {
            sum += A[b * M * K + m * K + k] * B[b * K * N + k * N + n];
        }
        Out[b * M * N + m * N + n] = sum;
    }
}

bool matchDotF32_3D_CUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2) return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    const auto &s0 = inputs[0].shape;
    const auto &s1 = inputs[1].shape;
    const auto &so = output.shape;

    if (s0.size() != 3 || s1.size() != 3 || so.size() != 3) return false;
    
    // Batch and K-dim checks
    if (s0[0] != s1[0] || s0[2] != s1[1]) return false;
    // Output shape checks
    if (so[0] != s0[0] || so[1] != s0[1] || so[2] != s1[2]) return false;

    if (!inputs[0].view.isContiguous() || !inputs[1].view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

void runDotF32_3D_CUDA(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                       const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *A = static_cast<const float *>(inputs[0]);
    const float *B = static_cast<const float *>(inputs[1]);
    float *Out = static_cast<float *>(outputs[0]);

    uint32_t B_count = inViews[0].shape[0];
    uint32_t M = inViews[0].shape[1];
    uint32_t K = inViews[0].shape[2];
    uint32_t N = inViews[1].shape[2];

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, 
                (M + threads.y - 1) / threads.y, 
                B_count);

    dot_f32_3d_kernel<<<blocks, threads>>>(A, B, Out, B_count, M, K, N);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in runDotF32_3D_CUDA: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA kernel failed");
    }
}
#endif