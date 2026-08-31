#pragma once
#include "core/common/constants.hpp"
#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

inline bool matchExp3DFill_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != output.getShape())
        return false;
    return isContiguous(output);
}

inline void runExp3DFill_NEON(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    uint32_t nt = std::thread::hardware_concurrency();
    if (nt == 0)
        nt = 1;

    ThreadPool::get().parallel_for(nt, [=](uint32_t t) {
        uint64_t chunk = (n + nt - 1) / nt;
        uint64_t start = t * chunk;
        uint64_t end = std::min(start + chunk, n);
        for (uint64_t i = start; i < end; ++i)
            out[i] = std::exp(in[i]);
    });
}

inline LogicalId refFactoryExp3DFill(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    auto shape = g.getNode(x).getShape();
    LogicalId e_node = g.fill(TGConstants::E, shape);
    return g.pow(e_node, x);
}

REGISTER_KERNEL("Exp_3D_Fill_NEON", 1, 1, matchExp3DFill_NEON, runExp3DFill_NEON, refFactoryExp3DFill, {0},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 256, 128}}, {true},
                {{MemSpace(1, HandleType::CPP)}});