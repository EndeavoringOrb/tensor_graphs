#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchConv2d3x3Dot(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return isContiguous(output);
}

inline void runConv2d3x3Dot(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    const float *w = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &sX = ctx.inViews[0].getShape();
    const auto &sW = ctx.inViews[1].getShape();
    const auto &sO = ctx.outViews[0].getShape();

    uint32_t out_c = sO[1];
    uint32_t H_out = sO[2];
    uint32_t W_out = sO[3];

    uint32_t in_c = sX[1];
    uint32_t H_in = sX[2];
    uint32_t W_in = sX[3];

    uint32_t k_sq = sW[2] / in_c;
    uint32_t k = static_cast<uint32_t>(std::round(std::sqrt(static_cast<float>(k_sq))));

    uint32_t stride = 1;
    uint32_t pad = 1;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t chunk = (out_c + num_threads - 1) / num_threads;
        uint32_t start_c = t * chunk;
        uint32_t end_c = std::min(start_c + chunk, out_c);

        for (uint32_t oc = start_c; oc < end_c; ++oc)
        {
            const float *w_row = w + static_cast<uint64_t>(oc) * (in_c * k * k);
            for (uint32_t oh = 0; oh < H_out; ++oh)
            {
                for (uint32_t ow = 0; ow < W_out; ++ow)
                {
                    float sum = 0.0f;
                    uint32_t w_idx = 0;
                    for (uint32_t ic = 0; ic < in_c; ++ic)
                    {
                        for (uint32_t ky = 0; ky < k; ++ky)
                        {
                            int32_t in_y = static_cast<int32_t>(oh * stride) - static_cast<int32_t>(pad) + ky;
                            for (uint32_t kx = 0; kx < k; ++kx)
                            {
                                int32_t in_x = static_cast<int32_t>(ow * stride) - static_cast<int32_t>(pad) + kx;
                                if (in_y >= 0 && in_y < static_cast<int32_t>(H_in) && in_x >= 0 &&
                                    in_x < static_cast<int32_t>(W_in))
                                {
                                    sum += x[ic * H_in * W_in + in_y * W_in + in_x] * w_row[w_idx];
                                }
                                w_idx++;
                            }
                        }
                    }
                    out[oc * H_out * W_out + oh * W_out + ow] = sum;
                }
            }
        }
    });
}

inline LogicalId refFactoryConv2d3x3Dot(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    LogicalId w = inputs[1]; // [1, out_c, in_c * k * k]

    int32_t k = 3;
    int32_t stride = 1;
    int32_t pad = 1;

    LogicalId k_node = g.constant({1}, &k, DType::INT32);
    LogicalId s_node = g.constant({1}, &stride, DType::INT32);
    LogicalId p_node = g.constant({1}, &pad, DType::INT32);

    LogicalId col = g.im2col(x, k_node, s_node, p_node);
    LogicalId out_flat = g.dot(w, col);

    auto sX = g.getNode(x).getShape();
    auto sW = g.getNode(w).getShape();
    uint32_t out_c = sW[1];
    uint32_t H = sX[2];
    uint32_t W = sX[3];

    int32_t sh4[] = {1, static_cast<int32_t>(out_c), static_cast<int32_t>(H), static_cast<int32_t>(W)};
    return g.reshape(out_flat, g.constant({4}, sh4, DType::INT32));
}

REGISTER_KERNEL("Conv2d_3x3_Im2Col_Dot", 2, 2, matchConv2d3x3Dot, runConv2d3x3Dot, refFactoryConv2d3x3Dot, {},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 384, 128, 128}, {1, 384, 3456}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});