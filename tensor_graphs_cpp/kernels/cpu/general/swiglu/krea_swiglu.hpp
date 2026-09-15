#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/constants.hpp"
#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

inline bool matchKreaSwiglu(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 3)
        return false;
    if (inputs[0].getShape() != inputs[1].getShape())
        return false;
    return isContiguous(output);
}

inline void runKreaSwiglu(const KernelContext &ctx)
{
    const float *gate = static_cast<const float *>(ctx.inputs[0]);
    const float *up = static_cast<const float *>(ctx.inputs[1]);
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
            float x = gate[i];
            float sig = (x >= 0.0f) ? (1.0f / (1.0f + std::exp(-x))) : (std::exp(x) / (1.0f + std::exp(x)));
            out[i] = x * sig * up[i];
        }
    });
}

inline LogicalId refFactoryKreaSwiglu(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    LogicalId up = inputs[1];
    auto shape = g.getNode(x).getShape();

    LogicalId neg_x = g.mul(x, g.fill(-1.0f, shape));
    LogicalId exp_neg_x = g.pow(g.fill(TGConstants::E, shape), neg_x);
    LogicalId one = g.fill(1.0f, shape);
    LogicalId sig = g.div(one, g.add(one, exp_neg_x));
    LogicalId gate_silu = g.mul(x, sig);
    return g.mul(gate_silu, up);
}

REGISTER_KERNEL("Krea_SwiGLU", 2, 2, matchKreaSwiglu, runKreaSwiglu, refFactoryKreaSwiglu, {0, 1},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 4224, 6144}, {1, 4224, 6144}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});