#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchRMSNormF32_ND_Threaded_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[1].getShape().size() != 1)
        return false;
    if (inputs[0].getShape().back() != inputs[1].getShape()[0])
        return false;
    return isContiguous(output);
}

inline void runRMSNormF32_ND_Threaded_NEON(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                           const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *x = static_cast<const float *>(inputs[0]);
    const float *w = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    const auto &shape = inViews[0].getShape();
    uint32_t D = shape.back();
    uint64_t outer_size = countElements(shape) / D;
    float eps = 1e-6f;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    uint64_t chunk = (outer_size + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint64_t start = t * chunk;
            uint64_t end = std::min(start + chunk, outer_size);

            for (uint64_t i = start; i < end; ++i) {
                const float *x_row = x + i * D;
                float *out_row = out + i * D;

                float32x4_t v_sum = vdupq_n_f32(0.0f);
                uint32_t d = 0;
                for (; d + 3 < D; d += 4) {
                    float32x4_t vx = vld1q_f32(x_row + d);
                    v_sum = vfmaq_f32(v_sum, vx, vx);
                }
                float sum_sq = vaddvq_f32(v_sum); // NEON native horizontal reduction
                for (; d < D; ++d) sum_sq += x_row[d] * x_row[d];

                float mean_sq = sum_sq / (float)D;
                float inv_std = 1.0f / std::sqrt(mean_sq + eps);
                float32x4_t v_inv_std = vdupq_n_f32(inv_std);

                for (d = 0; d + 3 < D; d += 4) {
                    float32x4_t vx = vld1q_f32(x_row + d);
                    float32x4_t vw = vld1q_f32(w + d);
                    float32x4_t vw_plus_one = vaddq_f32(vw, vdupq_n_f32(1.0f));
                    vst1q_f32(out_row + d, vmulq_f32(vmulq_f32(vx, v_inv_std), vw_plus_one));
                }
                for (; d < D; ++d) {
                    out_row[d] = x_row[d] * inv_std * (w[d] + 1.0f);
                }
            } });
    }
    for (auto &w : workers)
        w.join();
}

inline uint32_t refFactoryRMSNorm_Threaded_NEON(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t x_id = inputs[0];
    uint32_t w_id = inputs[1];

    auto shapeX = graph.getNode(x_id).getShape();
    uint32_t D = shapeX.back();

    uint32_t x_sq = graph.mul(x_id, x_id);
    int32_t ax_val = -1;
    uint32_t sum_sq = graph.sum(x_sq, graph.constant({1}, &ax_val, DType::INT32));

    float d_float = (float)D;
    auto create_bcast = [&](float val)
    {
        std::vector<int32_t> ones(shapeX.size() - 1, 1);
        uint32_t node = graph.reshape(graph.constant({1}, &val, DType::FLOAT32), graph.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));
        for (size_t i = 0; i < shapeX.size() - 1; ++i)
        {
            if (shapeX[i] > 1)
            {
                int32_t rep = shapeX[i];
                int32_t ax = i;
                node = graph.repeat(node, graph.constant({1}, &rep, DType::INT32), graph.constant({1}, &ax, DType::INT32));
            }
        }
        return node;
    };

    uint32_t mean_sq = graph.div(sum_sq, create_bcast(d_float));
    uint32_t var = graph.add(mean_sq, create_bcast(1e-6f));
    uint32_t std = graph.pow(var, create_bcast(0.5f));
    uint32_t inv_std = graph.div(create_bcast(1.0f), std);

    int32_t d_rep = (int32_t)D;
    int32_t d_ax = shapeX.size() - 1;
    uint32_t inv_std_exp = graph.repeat(inv_std, graph.constant({1}, &d_rep, DType::INT32), graph.constant({1}, &d_ax, DType::INT32));
    uint32_t x_norm = graph.mul(x_id, inv_std_exp);

    std::vector<int32_t> sh_w(shapeX.size(), 1);
    sh_w.back() = D;
    uint32_t w_reshaped = graph.reshape(w_id, graph.constant({(uint32_t)sh_w.size()}, sh_w.data(), DType::INT32));

    uint32_t w_exp = w_reshaped;
    for (size_t i = 0; i < shapeX.size() - 1; ++i)
    {
        if (shapeX[i] > 1)
        {
            int32_t rep = shapeX[i], ax = i;
            w_exp = graph.repeat(w_exp, graph.constant({1}, &rep, DType::INT32), graph.constant({1}, &ax, DType::INT32));
        }
    }

    std::vector<int32_t> ones_full(shapeX.size(), 1);
    uint32_t one_full = graph.reshape(graph.constant({1}, &d_float, DType::FLOAT32), graph.constant({(uint32_t)ones_full.size()}, ones_full.data(), DType::INT32)); // Dummy, let compiler fold
    // Actually simpler: Add broadcast 1.0f directly to weights
    return graph.mul(x_norm, graph.add(w_exp, graph.constant({1}, &d_float, DType::FLOAT32))); // Using generic broadcast logic will match E-graph
}

REGISTER_KERNEL("RMSNorm_F32_ND_Threaded_NEON", 2, matchRMSNormF32_ND_Threaded_NEON, runRMSNormF32_ND_Threaded_NEON, refFactoryRMSNorm_Threaded_NEON, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 32, 512, 128}, {128}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});
#endif