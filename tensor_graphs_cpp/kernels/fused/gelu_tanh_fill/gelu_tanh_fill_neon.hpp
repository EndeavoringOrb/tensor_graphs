#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/constants.hpp"
#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/gelu_tanh_fill/ref.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

inline bool matchGeluTanhFill_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    return isContiguous(output);
}

inline void runGeluTanhFill_NEON(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint64_t chunk = (n + num_threads - 1) / num_threads;
        uint64_t start = t * chunk;
        uint64_t end = std::min(start + chunk, n);

        for (uint64_t i = start; i < end; ++i)
        {
            float x = in[i];
            float x3 = x * x * x;
            float inner = 0.79788456f * (x + 0.044715f * x3);
            float tanh_val = std::tanh(inner);
            out[i] = 0.5f * x * (1.0f + tanh_val);
        }
    });
}



REGISTER_KERNEL("Gelu_Tanh_Fill_NEON", 1, 1, matchGeluTanhFill_NEON, runGeluTanhFill_NEON, refFactoryGeluTanhFill, {0},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 128, 6144}}, {true},
                {{MemSpace(1, HandleType::CPP)}});