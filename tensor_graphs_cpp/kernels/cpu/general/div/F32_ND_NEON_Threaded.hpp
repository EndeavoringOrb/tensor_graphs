#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <vector>

#include "core/common/thread_pool.hpp"

inline bool matchDivF32_ND_Fast(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape() == inputs[1].getShape() && isContiguous(output);
}

inline void runDivF32_ND_Fast(const KernelContext &ctx)
{
    const float *a = static_cast<const float *>(ctx.inputs[0]);
    const float *b = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint64_t chunk = (n + num_threads - 1) / num_threads;
        uint64_t start = t * chunk;
        uint64_t end = std::min(start + chunk, n);
        uint64_t i = start;
        for (; i + 4 <= end; i += 4)
        {
            vst1q_f32(out + i, vdivq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
        }
        for (; i < end; ++i)
            out[i] = a[i] / b[i];
    });
}

inline LogicalId refFactoryDivND_Fast(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Div ND Fast requires exactly 2 inputs");

    return graph.div(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Div_ND_NEON_Threaded", 2, 2, matchDivF32_ND_Fast, runDivF32_ND_Fast, refFactoryDivND_Fast, {0, 1},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 24, 1536, 1536}, {1, 24, 1536, 1536}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif