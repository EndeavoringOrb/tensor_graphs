#pragma once
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/swiglu_3d/ref.hpp"
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
        workers.emplace_back([=]() {
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
            }
        });
    }

    for (auto &w : workers)
        w.join();
}




REGISTER_KERNEL("SwiGLU_3D_NEON_F32_Inplace", 2, 2, matchSwiGLU_3D_NEON_Inplace, runSwiGLU_3D_NEON_Inplace,
                refFactorySwiGLU_3D_NEON, {0, 1}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32}, {{1, 1536, 9216}, {1, 1536, 9216}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON