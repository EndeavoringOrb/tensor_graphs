#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/gelu_3d_threaded/ref.hpp"
inline bool matchGeluF32_3D_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape().size() == 3 && isContiguous(output);
}

inline void runGeluF32_3D_Threaded(const KernelContext &ctx)
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
            float t_val = std::tanh(inner);
            out[i] = 0.5f * x * (1.0f + t_val);
        }
    });
}

REGISTER_KERNEL("Gelu_3D_Threaded", 1, 1, matchGeluF32_3D_Threaded, runGeluF32_3D_Threaded, refFactoryGelu_3D_Threaded,
                {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 1, 2048}},
                {true}, {{MemSpace(1, HandleType::CPP)}});
