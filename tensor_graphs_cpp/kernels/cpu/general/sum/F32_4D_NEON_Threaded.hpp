#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <thread>
#include <algorithm>
#include <vector>

inline bool matchSumF32_4D_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (!isContiguous(output))
        return false;
    const auto &shape = inputs[0].getShape();
    if (shape.size() != 4)
        return false;

    return true;
}

inline void runSumF32_4D_Threaded(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                  const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    float *out = static_cast<float *>(outputs[0]);
    const auto &shape = inViews[0].getShape();

    int32_t axis = *static_cast<const int32_t *>(inputs[1]);
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
                // Optimized Path: Reduction on last dimension (inner == 1)
                if (inner == 1) {
                    const float* row_in = in + (o * mid);
                    float32x4_t v_acc = vdupq_n_f32(0.0f);
                    uint32_t m = 0;
                    for (; m + 4 <= mid; m += 4) {
                        v_acc = vaddq_f32(v_acc, vld1q_f32(row_in + m));
                    }
                    float row_sum = vaddvq_f32(v_acc);
                    for (; m < mid; ++m) row_sum += row_in[m];
                    out[o] = row_sum;
                } else {
                    // General Path: Reduction on middle dimension
                    for (uint32_t i = 0; i < inner; ++i) {
                        float sum = 0.0f;
                        for (uint32_t m = 0; m < mid; ++m) {
                            sum += in[(o * mid + m) * inner + i];
                        }
                        out[o * inner + i] = sum;
                    }
                }
            } });
    }
    for (auto &w : workers)
        w.join();
}

inline uint32_t refFactorySum4D(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.sum(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Sum_F32_4D_Threaded", 2, matchSumF32_4D_Threaded, runSumF32_4D_Threaded, refFactorySum4D, {Backend::CPU}, {DType::FLOAT32, DType::INT32}, {{1, 24, 1536, 128}, {1}}, {true, false}, {{Backend::CPU}, {Backend::CPU}});
#endif