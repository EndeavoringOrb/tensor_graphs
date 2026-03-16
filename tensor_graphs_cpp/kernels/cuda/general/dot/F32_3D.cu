#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <iostream>
#include <vector>

/**
 * CUDA KERNEL: F32 3D Dot Product (Batched MatMul)
 */
__global__ void dot_f32_3d_kernel(const float *A, const float *B, float *Out,
                                  uint32_t B_count, uint32_t M, uint32_t K, uint32_t N)
{
    uint32_t b = blockIdx.z;
    uint32_t m = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < B_count && m < M && n < N)
    {
        float sum = 0.0f;
        for (uint32_t k = 0; k < K; ++k)
        {
            sum += A[b * M * K + m * K + k] * B[b * K * N + k * N + n];
        }
        Out[b * M * N + m * N + n] = sum;
    }
}

/**
 * Match function: Validates if the tensors are compatible with this specific 3D CUDA implementation.
 */
inline bool matchDotF32_3D_CUDA(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2)
        return false;

    // Check Dtypes
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    const auto &s0 = inputs[0].shape;
    const auto &s1 = inputs[1].shape;
    const auto &so = output.shape;

    // Rank Check (Must be 3D)
    if (s0.size() != 3 || s1.size() != 3 || so.size() != 3)
        return false;

    // MatMul dimension checks: [B, M, K] x [B, K, N] -> [B, M, N]
    if (s0[0] != s1[0] || s0[2] != s1[1])
        return false;
    if (so[0] != s0[0] || so[1] != s0[1] || so[2] != s1[2])
        return false;

    // Memory layout check
    if (!inputs[0].view.isContiguous() || !inputs[1].view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

/**
 * Run function: Sets up the grid/block dimensions and launches the CUDA kernel.
 */
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

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Launch Error in runDotF32_3D_CUDA: " << cudaGetErrorString(err) << std::endl;
        Error:throw_error("CUDA kernel launch failed");
    }
}

/**
 * Reference Factory: Defines how this kernel relates to high-level graph operations.
 * Since this is a standard Dot implementation, it simply maps to the dot operation.
 */
inline uint32_t refFactoryDotF32_3D_CUDA(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error:throw_error("Dot 3D requires 2 inputs");

    return graph.dot(inputs[0], inputs[1]);
}

/**
 * Registration: Registers this named kernel with the engine.
 * Name: "Dot_F32_3D_CUDA"
 * Inputs: 2
 * Backend: CUDA
 */
REGISTER_KERNEL(
    "Dot_F32_3D_CUDA",
    2,
    Backend::CUDA,
    matchDotF32_3D_CUDA,
    runDotF32_3D_CUDA,
    refFactoryDotF32_3D_CUDA,
    {DType::FLOAT32, DType::FLOAT32},
    {{2, 8, 16}, {2, 16, 8}}, {true, true});

#endif