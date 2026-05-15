// File: tensor_graphs_cpp/kernels/cpu/general/swiglu/3D_NEON_F32.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline float32x4_t fast_exp_neon(float32x4_t x)
{
    // clamp x to [-88.0f, 88.0f] to prevent overflow/underflow
    x = vmaxq_f32(vdupq_n_f32(-88.0f), vminq_f32(x, vdupq_n_f32(88.0f)));

    // exp(x) = 2^(x * log2(e))
    float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);
    float32x4_t y = vmulq_f32(x, log2e);

    // Split y into integer (n) and fractional (f) parts
    float32x4_t magic = vdupq_n_f32(12582912.0f);
    float32x4_t shifted = vaddq_f32(y, magic);
    float32x4_t n_float = vsubq_f32(shifted, magic);

    float32x4_t f = vsubq_f32(y, n_float);

    float32x4_t c1 = vdupq_n_f32(0.6931471805599453f);
    float32x4_t c2 = vdupq_n_f32(0.24022650695910139f);
    float32x4_t c3 = vdupq_n_f32(0.055504108664821579f);
    float32x4_t c4 = vdupq_n_f32(0.009618129107628477f);

    float32x4_t poly = vfmaq_f32(c3, f, c4);
    poly = vfmaq_f32(c2, f, poly);
    poly = vfmaq_f32(c1, f, poly);
    poly = vfmaq_f32(vdupq_n_f32(1.0f), f, poly);

    int32x4_t n_int = vcvtq_s32_f32(n_float);
    int32x4_t bias = vdupq_n_s32(127);
    n_int = vaddq_s32(n_int, bias);
    n_int = vshlq_n_s32(n_int, 23);
    float32x4_t twon = vreinterpretq_f32_s32(n_int);

    return vmulq_f32(twon, poly);
}

inline bool matchSwiGLU_3D_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape())
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return inputs[0].getShape().size() == 3;
}

inline void runSwiGLU_3D_NEON(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *gate = static_cast<const float *>(inputs[0]);
    const float *up = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint64_t n = countElements(inViews[0].getShape());

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    uint64_t chunk = (n + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint64_t start = t * chunk;
            uint64_t end = std::min(start + chunk, n);
            uint64_t i = start;
            
            for (; i + 4 <= end; i += 4)
            {
                float32x4_t v_gate = vld1q_f32(gate + i);
                float32x4_t v_up = vld1q_f32(up + i);
                
                float32x4_t neg_gate = vnegq_f32(v_gate);
                float32x4_t exp_neg_gate = fast_exp_neon(neg_gate);
                float32x4_t den = vaddq_f32(vdupq_n_f32(1.0f), exp_neg_gate);
                
                float32x4_t silu_gate = vdivq_f32(v_gate, den);
                float32x4_t result = vmulq_f32(silu_gate, v_up);
                
                vst1q_f32(out + i, result);
            }

            for (; i < end; ++i)
            {
                float x = gate[i];
                float y = up[i];
                if (x >= 0.0f)
                {
                    out[i] = (x / (1.0f + std::exp(-x))) * y;
                }
                else
                {
                    float exp_x = std::exp(x);
                    out[i] = (x * exp_x / (1.0f + exp_x)) * y;
                }
            } });
    }

    for (auto &w : workers)
        w.join();
}

inline void runSwiGLU_3D_NEON_Inplace(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                      const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    float *gate_out = static_cast<float *>(outputs[0]);
    const float *up = static_cast<const float *>(inputs[1]);

    uint64_t n = countElements(inViews[0].getShape());

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    uint64_t chunk = (n + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint64_t start = t * chunk;
            uint64_t end = std::min(start + chunk, n);
            uint64_t i = start;
            
            for (; i + 4 <= end; i += 4)
            {
                float32x4_t v_gate = vld1q_f32(gate_out + i);
                float32x4_t v_up = vld1q_f32(up + i);
                
                float32x4_t neg_gate = vnegq_f32(v_gate);
                float32x4_t exp_neg_gate = fast_exp_neon(neg_gate);
                float32x4_t den = vaddq_f32(vdupq_n_f32(1.0f), exp_neg_gate);
                
                float32x4_t silu_gate = vdivq_f32(v_gate, den);
                float32x4_t result = vmulq_f32(silu_gate, v_up);
                
                vst1q_f32(gate_out + i, result);
            }

            for (; i < end; ++i)
            {
                float x = gate_out[i];
                float y = up[i];
                if (x >= 0.0f)
                {
                    gate_out[i] = (x / (1.0f + std::exp(-x))) * y;
                }
                else
                {
                    float exp_x = std::exp(x);
                    gate_out[i] = (x * exp_x / (1.0f + exp_x)) * y;
                }
            } });
    }

    for (auto &w : workers)
        w.join();
}

inline uint32_t ref_swiglu_broadcast_scalar(Graph &g, uint32_t scalar_id, const std::vector<uint32_t> &target_shape)
{
    std::vector<int32_t> ones(target_shape.size(), 1);
    uint32_t out = g.reshape(scalar_id, g.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));
    for (size_t i = 0; i < target_shape.size(); ++i)
    {
        if (target_shape[i] > 1)
        {
            int32_t rep = (int32_t)target_shape[i];
            int32_t axis = (int32_t)i;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &axis, DType::INT32));
        }
    }
    return out;
}

inline uint32_t refFactorySwiGLU_3D_NEON(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t gate = inputs[0];
    uint32_t up = inputs[1];
    const auto &target_shape = graph.getNode(gate).getShape();

    // 1. neg_x = -x
    uint32_t neg_x = graph.neg(gate);

    // 2. exp_neg = pow(e, -x)
    float e_val = 2.7182818f;
    uint32_t e_node = ref_swiglu_broadcast_scalar(graph, graph.constant({1}, &e_val, DType::FLOAT32), target_shape);
    uint32_t exp_neg = graph.pow(e_node, neg_x);

    // 3. den = 1 + exp(-x)
    float one_val = 1.0f;
    uint32_t one_node = ref_swiglu_broadcast_scalar(graph, graph.constant({1}, &one_val, DType::FLOAT32), target_shape);
    uint32_t den = graph.add(one_node, exp_neg);

    // 4. sig = 1 / den
    uint32_t sig = graph.div(one_node, den);

    // 5. silu_gate = gate * sig
    uint32_t silu_gate = graph.mul(gate, sig);

    // 6. result = silu_gate * up
    return graph.mul(silu_gate, up);
}

REGISTER_KERNEL("SwiGLU_3D_NEON_F32", 2, matchSwiGLU_3D_NEON, runSwiGLU_3D_NEON, refFactorySwiGLU_3D_NEON, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 1536, 9216}, {1, 1536, 9216}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});

#endif // TG_HAS_NEON