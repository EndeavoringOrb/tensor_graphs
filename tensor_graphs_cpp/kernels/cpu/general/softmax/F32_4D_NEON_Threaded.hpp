#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <cmath>
#include <thread>
#include <algorithm>

inline bool matchSoftmaxF32_4D_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape().size() == 4 && isContiguous(output);
}

inline void runSoftmaxF32_4D_Threaded(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    const auto &shape = ctx.inViews[0].getShape();

    uint32_t outer_size = shape[0] * shape[1] * shape[2];
    uint32_t dim_size = shape[3];

    uint32_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    uint32_t chunk = (outer_size + num_threads - 1) / num_threads;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t start = t * chunk;
            uint32_t end = std::min(start + chunk, outer_size);
            for (uint32_t i = start; i < end; ++i) {
                const float *r_in = in + i * dim_size;
                float *r_out = out + i * dim_size;

                float32x4_t v_max = vdupq_n_f32(-1e30f);
                uint32_t d = 0;
                for (; d + 4 <= dim_size; d += 4) v_max = vmaxq_f32(v_max, vld1q_f32(r_in + d));
                float max_val = vmaxvq_f32(v_max);
                for (; d < dim_size; ++d) max_val = std::max(max_val, r_in[d]);

                float sum_val = 0.0f;
                for (d = 0; d < dim_size; ++d) {
                    float e = std::exp(r_in[d] - max_val);
                    r_out[d] = e;
                    sum_val += e;
                }

                float inv_sum = 1.0f / sum_val;
                float32x4_t v_inv_sum = vdupq_n_f32(inv_sum);
                for (d = 0; d + 4 <= dim_size; d += 4) vst1q_f32(r_out + d, vmulq_f32(vld1q_f32(r_out + d), v_inv_sum));
                for (; d < dim_size; ++d) r_out[d] *= inv_sum;
            } });
    }
    for (auto &w : workers)
        w.join();
}

inline LogicalId refFactorySoftmax4D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    auto s = g.getNode(x).getShape();
    int32_t ax = -1;
    LogicalId axis_node = g.constant({1}, &ax, DType::INT32);
    LogicalId m_rep = g.constant({1}, (int32_t *)&s[3], DType::INT32);
    LogicalId ax_rep = g.constant({1}, (int32_t *)&ax, DType::INT32);

    LogicalId max_s = g.repeat(g.max(x, axis_node), m_rep, ax_rep);
    LogicalId shifted = g.add(x, g.neg(max_s));

    float e_v = 2.7182818f;
    LogicalId e_n = g.constant({1}, &e_v, DType::FLOAT32);
    int32_t sh4[] = {1, 1, 1, 1};
    LogicalId e_b = g.reshape(e_n, g.constant({4}, sh4, DType::INT32));
    for (int i = 0; i < 4; ++i)
    {
        int32_t r = (int32_t)s[i];
        if (r <= 1) continue;
        int32_t a = i;
        e_b = g.repeat(e_b, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }

    LogicalId exps = g.pow(e_b, shifted);
    LogicalId sums = g.repeat(g.sum(exps, axis_node), m_rep, ax_rep);
    return g.div(exps, sums);
}

REGISTER_KERNEL("Softmax_4D_Threaded", 1, 1, matchSoftmaxF32_4D_Threaded, runSoftmaxF32_4D_Threaded, refFactorySoftmax4D, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 24, 1536, 1536}}, {true}, {{MemSpace(1, HandleType::CPP)}});
#endif