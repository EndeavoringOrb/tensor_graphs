#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"
#include "core/common/thread_pool.hpp"

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

    if (n < 131072)
    {
        compute(0, n);
        return;
    }

    uint32_t nt = std::thread::hardware_concurrency();
    if (nt == 0)
        nt = 1;

    ThreadPool::get().parallel_for(nt, [=](uint32_t t) {
        uint64_t chunk = (n + nt - 1) / nt;
        compute(t * chunk, std::min((t + 1) * chunk, n));
    });
}

inline LogicalId refFactoryExpND(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    auto shape = g.getNode(x).getShape();
    float e_val = 2.7182818f;
    LogicalId e_node = g.constant({1}, &e_val, DType::FLOAT32);

    std::vector<int32_t> ones(shape.size(), 1);
    LogicalId current_e = g.reshape(e_node, g.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));

    for (uint64_t ax = 0; ax < shape.size(); ++ax)
    {
        if (shape[ax] > 1)
        {
            int32_t r = (int32_t)shape[ax];
            int32_t a = (int32_t)ax;
            current_e = g.repeat(current_e, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
    }
    return g.pow(current_e, x);
}

REGISTER_KERNEL("Exp_ND_NEON", 1, 1, matchExpND_NEON, runExpND_NEON, refFactoryExpND, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 256, 128}}, {true},
                {{MemSpace(1, HandleType::CPP)}});