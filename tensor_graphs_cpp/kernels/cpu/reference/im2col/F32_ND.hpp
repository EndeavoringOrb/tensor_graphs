#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>
#include <algorithm>

inline bool matchIm2ColF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 4)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[1].dtype != DType::INT32 || inputs[2].dtype != DType::INT32 || inputs[3].dtype != DType::INT32)
        return false;
    if (inputs[0].getShape().size() != 4)
        return false;
    if (output.getShape().size() != 3)
        return false;

    // IM2COL is expected to output to a contiguous memory block
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runIm2ColF32_ND(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                            const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    int32_t kernel_size = *static_cast<const int32_t *>(inputs[1]);
    int32_t stride = *static_cast<const int32_t *>(inputs[2]);
    int32_t padding = *static_cast<const int32_t *>(inputs[3]);
    float *out = static_cast<float *>(outputs[0]);

    const auto &inShape = inViews[0].getShape();
    uint32_t N = inShape[0];
    uint32_t C = inShape[1];
    uint32_t H = inShape[2];
    uint32_t W = inShape[3];

    uint32_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    uint32_t W_out = (W + 2 * padding - kernel_size) / stride + 1;

    const auto &inStrides = inViews[0].strides;
    uint64_t out_idx = 0;

    for (uint32_t n = 0; n < N; ++n)
    {
        uint64_t n_offset = n * inStrides[0];
        for (uint32_t c = 0; c < C; ++c)
        {
            uint64_t nc_offset = n_offset + c * inStrides[1];
            for (uint32_t ky = 0; ky < (uint32_t)kernel_size; ++ky)
            {
                for (uint32_t kx = 0; kx < (uint32_t)kernel_size; ++kx)
                {
                    for (uint32_t hy = 0; hy < H_out; ++hy)
                    {
                        int32_t in_y = (int32_t)(hy * stride) - padding + ky;
                        if (in_y >= 0 && in_y < (int32_t)H)
                        {
                            uint64_t ncy_offset = nc_offset + in_y * inStrides[2];
                            for (uint32_t wx = 0; wx < W_out; ++wx)
                            {
                                int32_t in_x = (int32_t)(wx * stride) - padding + kx;
                                if (in_x >= 0 && in_x < (int32_t)W)
                                {
                                    out[out_idx++] = in[ncy_offset + in_x * inStrides[3]];
                                }
                                else
                                {
                                    out[out_idx++] = 0.0f; // Padding
                                }
                            }
                        }
                        else
                        {
                            for (uint32_t wx = 0; wx < W_out; ++wx)
                            {
                                out[out_idx++] = 0.0f; // Padding
                            }
                        }
                    }
                }
            }
        }
    }
}

REGISTER_REF_KERNEL(OpType::IM2COL, 4, matchIm2ColF32_ND, runIm2ColF32_ND, {Backend::CPU}, {DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32}, {{1, 1, 8, 8}, {1}, {1}, {1}}, {false, false, false, false}, {{Backend::CPU}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});