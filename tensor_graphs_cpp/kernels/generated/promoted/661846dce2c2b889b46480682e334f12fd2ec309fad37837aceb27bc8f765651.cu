#if defined(TG_USE_CUDA)
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/dot/ref.hpp"

// Memory-bound skinny batched GEMM for the goal's projection calls:
// C[16, N] = A[16, 640] * B[640, N] with batch 1 and N a large power-of-two
// multiple of 2048. cuBLAS reaches only ~45% of DRAM bandwidth here; this
// kernel streams B exactly once with coalesced 128-bit loads behind a
// 4-stage register prefetch while A lives in shared memory and is broadcast
// into 16 independent FMA chains per thread.

inline bool matchDotSkinnyF32CUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.dtype != DType::FLOAT32)
        return false;
    const std::vector<uint32_t> &shapeA = inputs[0].getShape();
    const std::vector<uint32_t> &shapeB = inputs[1].getShape();
    const std::vector<uint32_t> &shapeC = output.getShape();
    if (shapeA.size() != 3 || shapeB.size() != 3 || shapeC.size() != 3)
        return false;
    if (shapeA[0] != 1 || shapeB[0] != 1 || shapeC[0] != 1)
        return false;
    if (shapeA[1] != 16 || shapeA[2] != 640)
        return false;
    if (shapeB[1] != 640 || shapeC[1] != 16)
        return false;
    const uint32_t nCols = shapeB[2];
    if (shapeC[2] != nCols)
        return false;
    switch (nCols)
    {
    case 16384u:
    case 32768u:
    case 65536u:
    case 131072u:
    case 262144u:
        break;
    default:
        return false;
    }
    if (!isContiguous(output))
        return false;
    return true;
}

namespace dotSkinnyF32CUDADetail
{

constexpr int kRows = 16;
constexpr int kDepth = 640;
constexpr int kThreads = 512;
constexpr int kColsPerThread = 4;
constexpr int kColsPerBlock = kThreads * kColsPerThread; // 2048
constexpr int kPrefetch = 4;

__global__ void __launch_bounds__(kThreads, 1)
dotSkinnyKernel(const float *__restrict__ inputA, const float *__restrict__ inputB, float *__restrict__ outputC,
                int nCols)
{
    __shared__ float tileA[kRows][kDepth];

    const int tid = threadIdx.x;
    // Cooperative load of A (16 x 640 floats = 2560 float4).
    const float4 *vectorA = reinterpret_cast<const float4 *>(inputA);
    float4 *vectorTile = reinterpret_cast<float4 *>(tileA);
#pragma unroll
    for (int i = 0; i < 5; ++i)
    {
        const int slot = tid + i * kThreads;
        if (slot < (kRows * kDepth) / 4)
            vectorTile[slot] = vectorA[slot];
    }
    __syncthreads();

    const int col0 = blockIdx.x * kColsPerBlock + tid * kColsPerThread;
    const long long nVec = nCols >> 2;
    const float4 *columnB = reinterpret_cast<const float4 *>(inputB) + (col0 >> 2);

    float accumulator[kRows][kColsPerThread];
#pragma unroll
    for (int m = 0; m < kRows; ++m)
#pragma unroll
        for (int c = 0; c < kColsPerThread; ++c)
            accumulator[m][c] = 0.0f;

    float4 stage[kPrefetch];
#pragma unroll
    for (int s = 0; s < kPrefetch; ++s)
        stage[s] = columnB[static_cast<long long>(s) * nVec];

#pragma unroll 4
    for (int k = 0; k < kDepth; ++k)
    {
        const int slot = k & (kPrefetch - 1);
        const float4 valueB = stage[slot];
        const int nextK = k + kPrefetch;
        if (nextK < kDepth)
            stage[slot] = columnB[static_cast<long long>(nextK) * nVec];
#pragma unroll
        for (int m = 0; m < kRows; ++m)
        {
            const float valueA = tileA[m][k];
            accumulator[m][0] = fmaf(valueA, valueB.x, accumulator[m][0]);
            accumulator[m][1] = fmaf(valueA, valueB.y, accumulator[m][1]);
            accumulator[m][2] = fmaf(valueA, valueB.z, accumulator[m][2]);
            accumulator[m][3] = fmaf(valueA, valueB.w, accumulator[m][3]);
        }
    }

#pragma unroll
    for (int m = 0; m < kRows; ++m)
    {
        float4 result;
        result.x = accumulator[m][0];
        result.y = accumulator[m][1];
        result.z = accumulator[m][2];
        result.w = accumulator[m][3];
        reinterpret_cast<float4 *>(outputC + static_cast<long long>(m) * nCols + col0)[0] = result;
    }
}

} // namespace dotSkinnyF32CUDADetail

inline void runDotSkinnyF32CUDA(const KernelContext &ctx)
{
    const TensorView &viewC = ctx.outViews[0];
    const std::vector<uint32_t> &shapeC = viewC.getShape();
    if (shapeC.size() != 3 || shapeC[0] != 1 || shapeC[1] != dotSkinnyF32CUDADetail::kRows)
        Error::throw_err("Dot_Skinny_F32_CUDA: unexpected output shape");
    const int nCols = static_cast<int>(shapeC[2]);
    if (nCols % dotSkinnyF32CUDADetail::kColsPerBlock != 0)
        Error::throw_err("Dot_Skinny_F32_CUDA: columns are not a multiple of the block tile");

    const float *inputA = static_cast<const float *>(ctx.inputs[0]);
    const float *inputB = static_cast<const float *>(ctx.inputs[1]);
    float *outputC = static_cast<float *>(ctx.outputs[0]);

    const int grid = nCols / dotSkinnyF32CUDADetail::kColsPerBlock;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    dotSkinnyF32CUDADetail::dotSkinnyKernel<<<grid, dotSkinnyF32CUDADetail::kThreads, 0, stream>>>(inputA, inputB,
                                                                                                     outputC, nCols);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        Error::throw_err("CUDA kernel launch failed in Dot_Skinny_F32_CUDA: " + std::string(cudaGetErrorString(err)));
}

REGISTER_KERNEL("Dot_Skinny_F32_CUDA", 2, 2, matchDotSkinnyF32CUDA, runDotSkinnyF32CUDA, refFactoryDotF32, {},
                MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)},
                {DType::FLOAT32, DType::FLOAT32}, {{1, 16, 64}, {1, 64, 32}}, {true, true},
                {MemSpace(2, HandleType::CUDA), MemSpace(2, HandleType::CUDA)});
#endif
