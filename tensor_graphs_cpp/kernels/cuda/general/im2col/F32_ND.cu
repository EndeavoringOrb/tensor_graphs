#ifdef USE_CUDA
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

__global__ void im2col_f32_nd_kernel(const float* img, float* col,
                                     int N, int C, int H, int W,
                                     int kernel_size, int stride, int padding,
                                     int H_out, int W_out) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total_threads = (uint64_t)N * C * H_out * W_out;
    if (idx >= total_threads) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    uint64_t col_batch_offset = n * (C * kernel_size * kernel_size * H_out * W_out);
    uint64_t col_c_offset = c * (kernel_size * kernel_size * H_out * W_out);
    uint64_t col_base = col_batch_offset + col_c_offset;

    uint64_t img_batch_offset = n * (C * H * W);
    uint64_t img_c_offset = c * (H * W);
    uint64_t img_base = img_batch_offset + img_c_offset;

    for (int ky = 0; ky < kernel_size; ++ky) {
        for (int kx = 0; kx < kernel_size; ++kx) {
            int in_y = h_out * stride - padding + ky;
            int in_x = w_out * stride - padding + kx;

            float val = 0.0f;
            if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                val = img[img_base + in_y * W + in_x];
            }

            int k_idx = ky * kernel_size + kx;
            uint64_t out_idx = col_base + k_idx * (H_out * W_out) + (h_out * W_out + w_out);
            col[out_idx] = val;
        }
    }
}

inline bool matchIm2ColF32_CUDA_ND(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    if (inputs.size() != 4) return false;
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32) return false;
    if (inputs[1].dtype != DType::INT32 || inputs[2].dtype != DType::INT32 || inputs[3].dtype != DType::INT32) return false;
    if (inputs[0].getShape().size() != 4 || output.getShape().size() != 3) return false;
    if (!isContiguous(output)) return false;
    return true;
}

inline void runIm2ColF32_CUDA_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                 const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews) {
    const float *img = static_cast<const float *>(inputs[0]);
    int32_t kernel_size = *static_cast<const int32_t *>(inputs[1]);
    int32_t stride = *static_cast<const int32_t *>(inputs[2]);
    int32_t padding = *static_cast<const int32_t *>(inputs[3]);
    float *col = static_cast<float *>(outputs[0]);

    auto inShape = inViews[0].getShape();
    int N = inShape[0];
    int C = inShape[1];
    int H = inShape[2];
    int W = inShape[3];

    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    uint64_t total_threads = (uint64_t)N * C * H_out * W_out;
    if (total_threads == 0) return;

    int blockSize = 256;
    int numBlocks = (total_threads + blockSize - 1) / blockSize;
    im2col_f32_nd_kernel<<<numBlocks, blockSize>>>(img, col, N, C, H, W, kernel_size, stride, padding, H_out, W_out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        Error::throw_err("CUDA kernel launch failed in Im2Col_CUDA_ND: " + std::string(cudaGetErrorString(err)));
    }
}

/**
 * Reference Factory
 */
inline uint32_t refFactoryIm2ColF32_ND_CUDA(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 4)
        Error::throw_err("Im2Col ND requires 4 inputs");

    return graph.im2col(inputs[0], inputs[1], inputs[2], inputs[3]);
}

REGISTER_KERNEL("Im2Col_F32_ND_CUDA", 4, matchIm2ColF32_CUDA_ND, runIm2ColF32_CUDA_ND, refFactoryIm2ColF32_ND_CUDA, {Backend::CUDA}, {DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32}, {{1, 3, 64, 64}, {1}, {1}, {1}}, {true, false, false, false}, {{Backend::CUDA}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});

#endif
