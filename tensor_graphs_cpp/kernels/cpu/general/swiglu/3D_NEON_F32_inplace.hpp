// File: tensor_graphs_cpp/kernels/cpu/general/swiglu/3D_NEON_F32_inplace.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cmath>
#include <thread>
#include <vector>
#include <algorithm>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchSwiGLU_3D_NEON_Inplace(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape())
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return inputs[0].getShape().size() == 3;
}

inline void runSwiGLU_3D_NEON_Inplace(const KernelContext &ctx)
{
    float *gate_out = static_cast<float *>(ctx.outputs[0]);
    const float *up = static_cast<const float *>(ctx.inputs[1]);

    uint64_t n = countElements(ctx.inViews[0].getShape());

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
                
                float32x4_t v_abs_gate = vabsq_f32(v_gate);
                float32x4_t v_neg_abs = vnegq_f32(v_abs_gate);

                float e0 = std::exp(vgetq_lane_f32(v_neg_abs, 0));
                float e1 = std::exp(vgetq_lane_f32(v_neg_abs, 1));
                float e2 = std::exp(vgetq_lane_f32(v_neg_abs, 2));
                float e3 = std::exp(vgetq_lane_f32(v_neg_abs, 3));
                
                float e_arr[4] = {e0, e1, e2, e3};
                float32x4_t v_e = vld1q_f32(e_arr);
                
                uint32x4_t v_mask = vcgeq_f32(v_gate, vdupq_n_f32(0.0f));
                float32x4_t v_gate_times_e = vmulq_f32(v_gate, v_e);
                
                float32x4_t v_num = vbslq_f32(v_mask, v_gate, v_gate_times_e);
                float32x4_t v_den = vaddq_f32(vdupq_n_f32(1.0f), v_e);
                
                float32x4_t v_silu = vdivq_f32(v_num, v_den);
                float32x4_t v_res = vmulq_f32(v_silu, v_up);
                
                vst1q_f32(gate_out + i, v_res);
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

inline uint32_t ref_swiglu_broadcast_scalar_Inplace(Graph &g, uint32_t scalar_id, const std::vector<uint32_t> &target_shape)
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

inline uint32_t refFactorySwiGLU_3D_NEON_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    uint32_t gate = inputs[0];
    uint32_t up = inputs[1];
    const auto &target_shape = graph.getNode(gate).getShape();

    // 1. neg_x = -x
    uint32_t neg_x = graph.neg(gate);

    // 2. exp_neg = pow(e, -x)
    float e_val = 2.7182818f;
    uint32_t e_node = ref_swiglu_broadcast_scalar_Inplace(graph, graph.constant({1}, &e_val, DType::FLOAT32), target_shape);
    uint32_t exp_neg = graph.pow(e_node, neg_x);

    // 3. den = 1 + exp(-x)
    float one_val = 1.0f;
    uint32_t one_node = ref_swiglu_broadcast_scalar_Inplace(graph, graph.constant({1}, &one_val, DType::FLOAT32), target_shape);
    uint32_t den = graph.add(one_node, exp_neg);

    // 4. sig = 1 / den
    uint32_t sig = graph.div(one_node, den);

    // 5. silu_gate = gate * sig
    uint32_t silu_gate = graph.mul(gate, sig);

    // 6. result = silu_gate * up
    return graph.mul(silu_gate, up);
}

REGISTER_KERNEL_INPLACE("SwiGLU_3D_NEON_F32_Inplace", 2, matchSwiGLU_3D_NEON_Inplace, runSwiGLU_3D_NEON_Inplace, refFactorySwiGLU_3D_NEON_Inplace, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 1536, 9216}, {1, 1536, 9216}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});

#endif // TG_HAS_NEON