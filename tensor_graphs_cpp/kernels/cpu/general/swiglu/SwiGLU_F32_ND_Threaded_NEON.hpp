#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>
#include <thread>
#include <algorithm>
#include <cmath>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchSwiGLU_F32_ND_Threaded_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape())
        return false;
    return isContiguous(output);
}

inline void runSwiGLU_F32_ND_Threaded_NEON(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                           const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *gate = static_cast<const float *>(inputs[0]);
    const float *up = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint64_t totalElements = countElements(inViews[0].getShape());
    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    uint64_t chunk = (totalElements + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint64_t start = t * chunk;
            uint64_t end = std::min(start + chunk, totalElements);
            
            for (uint64_t i = start; i < end; ++i) {
                float x = gate[i];
                // Numerically stable SiLU
                float silu_val;
                if (x >= 0.0f) {
                    silu_val = x / (1.0f + std::exp(-x));
                } else {
                    float exp_x = std::exp(x);
                    silu_val = x * exp_x / (1.0f + exp_x);
                }
                out[i] = silu_val * up[i];
            } });
    }
    for (auto &w : workers)
        w.join();
}

inline uint32_t refFactorySwiGLU(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t gate = inputs[0];
    uint32_t up = inputs[1];

    auto shape = graph.getNode(gate).getShape();

    auto bcast = [&](float val)
    {
        std::vector<int32_t> ones(shape.size(), 1);
        uint32_t node = graph.reshape(graph.constant({1}, &val, DType::FLOAT32), graph.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (shape[i] > 1)
            {
                int32_t rep = shape[i], ax = i;
                node = graph.repeat(node, graph.constant({1}, &rep, DType::INT32), graph.constant({1}, &ax, DType::INT32));
            }
        }
        return node;
    };

    uint32_t neg_x = graph.neg(gate);
    uint32_t exp_neg = graph.pow(bcast(2.7182818f), neg_x);
    uint32_t den = graph.add(bcast(1.0f), exp_neg);
    uint32_t sig = graph.div(bcast(1.0f), den);
    uint32_t silu = graph.mul(gate, sig);

    return graph.mul(silu, up);
}

REGISTER_KERNEL("SwiGLU_F32_ND_Threaded_NEON", 2, matchSwiGLU_F32_ND_Threaded_NEON, runSwiGLU_F32_ND_Threaded_NEON, refFactorySwiGLU, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 32, 512, 128}, {1, 32, 512, 128}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});
#endif