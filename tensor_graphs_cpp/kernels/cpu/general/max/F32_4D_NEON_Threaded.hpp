#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <thread>
#include <algorithm>
#include <vector>

inline bool matchMaxF32_4D_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (!isContiguous(output))
        return false;
    return inputs[0].getShape().size() == 4;
}

inline void runMaxF32_4D_Threaded(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    const auto &shape = ctx.inViews[0].getShape();

    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[1]);
    if (axis < 0)
        axis += 4;

    uint64_t outer = 1, mid = shape[axis], inner = 1;
    for (int i = 0; i < axis; ++i)
        outer *= shape[i];
    for (int i = axis + 1; i < 4; ++i)
        inner *= shape[i];

    uint32_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    uint32_t chunk = (outer + num_threads - 1) / num_threads;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t o_start = t * chunk;
            uint32_t o_end = std::min(o_start + chunk, (uint32_t)outer);
            
            for (uint32_t o = o_start; o < o_end; ++o) {
                if (inner == 1) {
                    const float* row_in = in + (o * mid);
                    float32x4_t v_max = vdupq_n_f32(-1e30f);
                    uint32_t m = 0;
                    for (; m + 4 <= mid; m += 4) {
                        v_max = vmaxq_f32(v_max, vld1q_f32(row_in + m));
                    }
                    float row_max = vmaxvq_f32(v_max);
                    for (; m < mid; ++m) row_max = std::max(row_max, row_in[m]);
                    out[o] = row_max;
                } else {
                    for (uint32_t i = 0; i < inner; ++i) {
                        float max_val = -1e30f;
                        for (uint32_t m = 0; m < mid; ++m) {
                            max_val = std::max(max_val, in[(o * mid + m) * inner + i]);
                        }
                        out[o * inner + i] = max_val;
                    }
                }
            } });
    }
    for (auto &w : workers)
        w.join();
}

inline uint32_t refFactoryMax4D(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.max(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Max_F32_4D_Threaded", 2, matchMaxF32_4D_Threaded, runMaxF32_4D_Threaded, refFactoryMax4D, {Backend::CPU}, {DType::FLOAT32, DType::INT32}, {{1, 24, 1536, 1536}, {1}}, {true, false}, {{Backend::CPU}, {Backend::CPU}});
#endif