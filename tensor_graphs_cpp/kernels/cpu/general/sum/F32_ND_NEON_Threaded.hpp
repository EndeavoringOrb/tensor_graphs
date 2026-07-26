#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

inline bool matchSumF32_ND_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (!isContiguous(output))
        return false;

    const auto &shape = inputs[0].getShape();
    // Support any dimensionality >= 1
    if (shape.empty())
        return false;

    return true;
}

inline void runSumF32_ND_Threaded(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    const auto &shape = ctx.inViews[0].getShape();
    int32_t ndim = static_cast<int32_t>(shape.size());

    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[1]);
    if (axis < 0)
        axis += ndim;

    // Calculate ND strides generalized into Outer, Mid (axis), and Inner
    uint64_t outer = 1, mid = shape[axis], inner = 1;
    for (int i = 0; i < axis; ++i)
        outer *= shape[i];
    for (int i = axis + 1; i < ndim; ++i)
        inner *= shape[i];

    uint32_t num_threads = std::thread::hardware_concurrency();
    // Prevent over-threading on small tensors
    num_threads = std::min((uint32_t)outer, num_threads);
    if (num_threads == 0)
        num_threads = 1;

    std::vector<std::thread> workers;
    uint32_t chunk = (outer + num_threads - 1) / num_threads;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]() {
            uint32_t o_start = t * chunk;
            uint32_t o_end = std::min(o_start + chunk, (uint32_t)outer);

            for (uint32_t o = o_start; o < o_end; ++o)
            {
                // Path A: Reduction on the very last dimension (Contiguous horizontal
                // sum)
                if (inner == 1)
                {
                    const float *row_in = in + (o * mid);
                    float32x4_t v_acc = vdupq_n_f32(0.0f);
                    uint32_t m = 0;
                    for (; m + 4 <= mid; m += 4)
                    {
                        v_acc = vaddq_f32(v_acc, vld1q_f32(row_in + m));
                    }
                    float row_sum = vaddvq_f32(v_acc); // Horizontal sum (ARMv8)
                    for (; m < mid; ++m)
                        row_sum += row_in[m];
                    out[o] = row_sum;
                }
                // Path B: Reduction on a middle or first dimension (Vector addition)
                else
                {
                    float *row_out = out + (o * inner);
                    // Initialize the output segment for this 'outer' index to 0
                    std::memset(row_out, 0, inner * sizeof(float));

                    for (uint32_t m = 0; m < mid; ++m)
                    {
                        const float *row_in = in + (o * mid + m) * inner;
                        uint32_t i = 0;
                        // Use SIMD to add the entire 'inner' row
                        for (; i + 4 <= inner; i += 4)
                        {
                            float32x4_t v_in = vld1q_f32(row_in + i);
                            float32x4_t v_out = vld1q_f32(row_out + i);
                            vst1q_f32(row_out + i, vaddq_f32(v_out, v_in));
                        }
                        for (; i < inner; ++i)
                        {
                            row_out[i] += row_in[i];
                        }
                    }
                }
            }
        });
    }

    for (auto &w : workers)
        w.join();
}

inline LogicalId refFactorySumND(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.sum(inputs[0], inputs[1]);
}

// Updated Registration: Works for any ND shape, assuming float32 data and int32
// axis
REGISTER_KERNEL("Sum_F32_ND_Threaded", 2, 2, matchSumF32_ND_Threaded, runSumF32_ND_Threaded, refFactorySumND,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::INT32},
                {{1, 4, 256, 128}, {1}}, {true, false},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif