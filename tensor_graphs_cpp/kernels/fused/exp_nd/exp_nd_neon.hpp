#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/exp_nd/ref.hpp"
inline bool matchExpND_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    return isContiguous(output);
}

inline void runExpND_NEON(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    auto compute = [&](uint64_t start, uint64_t end) {
        for (uint64_t i = start; i < end; ++i)
            out[i] = std::exp(in[i]);
    };

    uint32_t nt = std::thread::hardware_concurrency();
    if (nt == 0)
        nt = 1;

    ThreadPool::get().parallel_for(nt, [=](uint32_t t) {
        uint64_t chunk = (n + nt - 1) / nt;
        compute(t * chunk, std::min((t + 1) * chunk, n));
    });
}



REGISTER_KERNEL("Exp_ND_NEON", 1, 1, matchExpND_NEON, runExpND_NEON, refFactoryExpND, {0}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 256, 128}}, {true},
                {{MemSpace(1, HandleType::CPP)}});
