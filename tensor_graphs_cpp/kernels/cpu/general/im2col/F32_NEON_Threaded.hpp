#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

inline bool matchIm2ColF32_NEON_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 4 || output.getShape().size() != 3)
        return false;

    return isContiguous(output);
}

inline void runIm2ColF32_NEON_Threaded(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    int32_t kernel_size = *static_cast<const int32_t *>(ctx.inputs[1]);
    int32_t stride = *static_cast<const int32_t *>(ctx.inputs[2]);
    int32_t padding = *static_cast<const int32_t *>(ctx.inputs[3]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &inShape = ctx.inViews[0].getShape();
    uint32_t N = inShape[0];
    uint32_t C = inShape[1];
    uint32_t H = inShape[2];
    uint32_t W = inShape[3];

    if (kernel_size <= 0 || stride <= 0 || padding < 0)
        return;
    if (H + 2 * padding < static_cast<uint32_t>(kernel_size) || W + 2 * padding < static_cast<uint32_t>(kernel_size))
        return;

    uint32_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    uint32_t W_out = (W + 2 * padding - kernel_size) / stride + 1;
    uint32_t HW_out = H_out * W_out;

    const auto &inStrides = ctx.inViews[0].strides;
    uint64_t in_stride_N = inStrides[0];
    uint64_t in_stride_C = inStrides[1];
    uint64_t in_stride_H = inStrides[2];
    uint64_t in_stride_W = inStrides[3];

    uint32_t KK = static_cast<uint32_t>(kernel_size * kernel_size);
    uint32_t total_rows = N * C * KK;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    num_threads = std::min(num_threads, total_rows);

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t chunk = (total_rows + num_threads - 1) / num_threads;
        uint32_t start_row = t * chunk;
        uint32_t end_row = std::min(start_row + chunk, total_rows);

        for (uint32_t row_idx = start_row; row_idx < end_row; ++row_idx)
        {
            uint32_t n = row_idx / (C * KK);
            uint32_t rem = row_idx % (C * KK);
            uint32_t c = rem / KK;
            uint32_t k_idx = rem % KK;
            uint32_t ky = k_idx / kernel_size;
            uint32_t kx = k_idx % kernel_size;

            const float *in_nc = in + static_cast<uint64_t>(n) * in_stride_N + static_cast<uint64_t>(c) * in_stride_C;
            float *out_row = out + static_cast<uint64_t>(row_idx) * HW_out;

            // Determine valid horizontal span bounds [wx_valid_start, wx_valid_end)
            int32_t pad_kx = padding - static_cast<int32_t>(kx);
            uint32_t wx_valid_start = 0;
            if (pad_kx > 0)
            {
                wx_valid_start = static_cast<uint32_t>((pad_kx + stride - 1) / stride);
            }
            wx_valid_start = std::min(wx_valid_start, W_out);

            int32_t w_pad_kx = static_cast<int32_t>(W) + pad_kx;
            uint32_t wx_valid_end = W_out;
            if (w_pad_kx > 0)
            {
                wx_valid_end = static_cast<uint32_t>((w_pad_kx + stride - 1) / stride);
                wx_valid_end = std::min(wx_valid_end, W_out);
            }
            else
            {
                wx_valid_end = 0;
            }
            if (wx_valid_end < wx_valid_start)
                wx_valid_end = wx_valid_start;

            uint32_t copy_count = wx_valid_end - wx_valid_start;
            uint32_t tail_count = W_out - wx_valid_end;
            bool is_fast_contiguous = (stride == 1 && in_stride_W == 1);

            for (uint32_t hy = 0; hy < H_out; ++hy)
            {
                int32_t in_y = static_cast<int32_t>(hy * stride) - padding + static_cast<int32_t>(ky);
                float *out_hy = out_row + static_cast<uint64_t>(hy) * W_out;

                if (in_y < 0 || in_y >= static_cast<int32_t>(H))
                {
                    std::memset(out_hy, 0, W_out * sizeof(float));
                    continue;
                }

                const float *in_row = in_nc + static_cast<uint64_t>(in_y) * in_stride_H;

                if (wx_valid_start > 0)
                {
                    std::memset(out_hy, 0, wx_valid_start * sizeof(float));
                }

                if (copy_count > 0)
                {
                    if (is_fast_contiguous)
                    {
                        int32_t in_x_start = static_cast<int32_t>(wx_valid_start) - padding + static_cast<int32_t>(kx);
                        std::memcpy(out_hy + wx_valid_start, in_row + in_x_start, copy_count * sizeof(float));
                    }
                    else
                    {
                        for (uint32_t wx = wx_valid_start; wx < wx_valid_end; ++wx)
                        {
                            int32_t in_x = static_cast<int32_t>(wx * stride) - padding + static_cast<int32_t>(kx);
                            out_hy[wx] = in_row[static_cast<uint64_t>(in_x) * in_stride_W];
                        }
                    }
                }

                if (tail_count > 0)
                {
                    std::memset(out_hy + wx_valid_end, 0, tail_count * sizeof(float));
                }
            }
        }
    });
}

inline LogicalId refFactoryIm2ColF32_NEON_Threaded(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.im2col(inputs[0], inputs[1], inputs[2], inputs[3]);
}

REGISTER_KERNEL("Im2Col_F32_NEON_Threaded", 4, 4, matchIm2ColF32_NEON_Threaded, runIm2ColF32_NEON_Threaded,
                refFactoryIm2ColF32_NEON_Threaded, {}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32},
                {{1, 96, 64, 64}, {1}, {1}, {1}}, {false, false, false, false},
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}});