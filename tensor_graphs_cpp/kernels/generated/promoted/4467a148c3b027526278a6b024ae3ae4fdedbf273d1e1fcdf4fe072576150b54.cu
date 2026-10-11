#if defined(TG_USE_CUDA)
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

// Last-dimension row sum for small F32 tensors (RMSNorm/softmax partial sums
// in the decode goal). One warp per output row: vectorized float4 loads plus
// a warp-shuffle reduce. The generic Sum_F32_ND_CUDA kernel spends ~75 us on a
// 40 KB reduction; this version is launch-bound at a few microseconds.

inline LogicalId refFactorySumLastDimF32CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.sum(inputs[0], inputs[1]);
}

namespace sumLastDimF32CUDADetail
{

// Returns the single reduced dimension index, or -1 when the (input, output)
// pair is not a single-axis reduction over the last dimension of a fully
// contiguous F32 input.
inline int32_t classify(const std::vector<uint32_t> &inShape, const std::vector<uint64_t> &inStrides,
                        const std::vector<uint32_t> &outShape)
{
    if (inShape.size() != outShape.size() || inShape.size() < 2)
        return -1;
    int32_t reducedDim = -1;
    for (size_t d = 0; d < inShape.size(); ++d)
    {
        if (inShape[d] == outShape[d])
            continue;
        if (outShape[d] == 1 && reducedDim == -1)
        {
            reducedDim = static_cast<int32_t>(d);
            continue;
        }
        return -1;
    }
    if (reducedDim != static_cast<int32_t>(inShape.size()) - 1)
        return -1;

    // Require a fully contiguous row-major input so rows are dense.
    uint64_t expected = 1;
    for (int64_t d = static_cast<int64_t>(inShape.size()) - 1; d >= 0; --d)
    {
        if (inStrides[d] != expected)
            return -1;
        expected *= inShape[d];
    }
    if (inShape.back() < 16 || inShape.back() % 4 != 0)
        return -1;
    return reducedDim;
}

__global__ void rowSumKernel(const float *__restrict__ in, float *__restrict__ out, uint32_t cols,
                             uint32_t rows)
{
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t row = static_cast<uint64_t>(blockIdx.x) * (blockDim.x >> 5) + warp;
    if (row >= rows)
        return;

    const float *rp = in + row * cols;
    float acc = 0.0f;
    uint32_t i = lane * 4;
    for (; i + 3 < cols; i += 128)
    {
        float4 v = *reinterpret_cast<const float4 *>(rp + i);
        acc += v.x + v.y + v.z + v.w;
    }
    for (i = cols - (cols & 3u) + lane; i < cols; i += 32)
        acc += rp[i];

    acc += __shfl_down_sync(0xffffffffu, acc, 16);
    acc += __shfl_down_sync(0xffffffffu, acc, 8);
    acc += __shfl_down_sync(0xffffffffu, acc, 4);
    acc += __shfl_down_sync(0xffffffffu, acc, 2);
    acc += __shfl_down_sync(0xffffffffu, acc, 1);
    if (lane == 0)
        out[row] = acc;
}

} // namespace sumLastDimF32CUDADetail

inline bool matchSumLastDimF32CUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.dtype != DType::FLOAT32)
        return false;
    return sumLastDimF32CUDADetail::classify(inputs[0].getShape(), inputs[0].strides, output.getShape()) >= 0;
}

inline void runSumLastDimF32CUDA(const KernelContext &ctx)
{
    const TensorView &inView = ctx.inViews[0];
    const TensorView &outView = ctx.outViews[0];
    if (sumLastDimF32CUDADetail::classify(inView.getShape(), inView.strides, outView.getShape()) < 0)
        Error::throw_err("Sum_LastDim_F32_CUDA: unsupported reduction layout");

    const auto &shape = inView.getShape();
    const uint32_t cols = shape.back();
    uint64_t rows = 1;
    for (size_t d = 0; d + 1 < shape.size(); ++d)
        rows *= shape[d];
    if (countElements(outView.getShape()) != rows)
        Error::throw_err("Sum_LastDim_F32_CUDA: output size mismatch");

    const int blockSize = 256;
    const uint32_t warpsPerBlock = blockSize / 32;
    const uint32_t grid = static_cast<uint32_t>((rows + warpsPerBlock - 1) / warpsPerBlock);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    sumLastDimF32CUDADetail::rowSumKernel<<<grid, blockSize, 0, stream>>>(
        static_cast<const float *>(ctx.inputs[0]), static_cast<float *>(ctx.outputs[0]), cols,
        static_cast<uint32_t>(rows));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        Error::throw_err("CUDA kernel launch failed in Sum_LastDim_F32_CUDA: " +
                         std::string(cudaGetErrorString(err)));
}

REGISTER_KERNEL("Sum_LastDim_F32_CUDA", 2, 2, matchSumLastDimF32CUDA, runSumLastDimF32CUDA,
                refFactorySumLastDimF32CUDA, {}, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)},
                {DType::FLOAT32, DType::INT32}, {{1, 16, 640}, {1}}, {true, true},
                {MemSpace(2, HandleType::CUDA), MemSpace(1, HandleType::CPP)});
#endif
