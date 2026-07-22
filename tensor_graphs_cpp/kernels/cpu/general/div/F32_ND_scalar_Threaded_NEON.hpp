#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>
#include <thread>
#include <algorithm>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchDivF32_ND_Scalar_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != 1)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    return isContiguous(output);
}

inline void runDivF32_ND_Scalar_Threaded(const KernelContext &ctx)
{
    const float *dataND = static_cast<const float *>(ctx.inputs[0]);
    float scalarValue = *static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    // Optimize division into multiplication by reciprocal
    float invScalar = 1.0f / scalarValue;

    uint64_t totalElements = countElements(ctx.inViews[0].getShape());
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
            uint64_t i = start;
            
            float32x4_t v_inv = vdupq_n_f32(invScalar);
            
            for (; i + 4 <= end; i += 4) {
                float32x4_t v_data = vld1q_f32(dataND + i);
                vst1q_f32(out + i, vmulq_f32(v_data, v_inv)); // NEON FMA/Mul is faster than Div
            }
            for (; i < end; ++i) {
                out[i] = dataND[i] * invScalar;
            } });
    }
    for (auto &w : workers)
        w.join();
}

inline uint32_t refFactoryDivND_Scalar_Threaded(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t idND = inputs[0];
    uint32_t idScalar = inputs[1];
    auto shapeND = graph.getNode(idND).getShape();

    std::vector<int32_t> ones(shapeND.size(), 1);
    uint32_t reshaped = graph.reshape(idScalar, graph.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));

    uint32_t out = reshaped;
    for (uint64_t i = 0; i < shapeND.size(); ++i)
    {
        if (shapeND[i] > 1)
        {
            int32_t rep = shapeND[i];
            int32_t ax = i;
            out = graph.repeat(out, graph.constant({1}, &rep, DType::INT32), graph.constant({1}, &ax, DType::INT32));
        }
    }
    return graph.div(idND, out);
}

REGISTER_KERNEL("Div_ND_Scalar_Threaded_NEON", 2, matchDivF32_ND_Scalar_Threaded, runDivF32_ND_Scalar_Threaded, refFactoryDivND_Scalar_Threaded, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 32, 512, 128}, {1}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});
#endif