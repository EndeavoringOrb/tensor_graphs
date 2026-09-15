#pragma once
#include "core/common/constants.hpp"
#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

inline bool matchSigmoidF32_3D_Neg_Neon(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    return isContiguous(output);
}

inline void runSigmoidF32_3D_Neg_Neon(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    uint64_t totalElements = countElements(ctx.inViews[0].getShape());

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint64_t chunk = (totalElements + num_threads - 1) / num_threads;
        uint64_t start = t * chunk;
        uint64_t end = std::min(start + chunk, totalElements);

        for (uint64_t i = start; i < end; ++i)
        {
            float x = in[i];
            if (x >= 0.0f)
            {
                out[i] = 1.0f / (1.0f + std::exp(-x));
            }
            else
            {
                float ex = std::exp(x);
                out[i] = ex / (1.0f + ex);
            }
        }
    });
}

inline LogicalId refFactorySigmoid_3D_Neg(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x = inputs[0];
    auto shape = graph.getNode(x).getShape();
    LogicalId neg_x = graph.neg(x);
    LogicalId e_node = graph.fill(TGConstants::E, shape);
    LogicalId exp_neg_x = graph.pow(e_node, neg_x);
    LogicalId one = graph.fill(1.0f, shape);
    LogicalId den = graph.add(one, exp_neg_x);
    return graph.div(one, den);
}

REGISTER_KERNEL("Sigmoid_3D_Neg_NEON", 1, 1, matchSigmoidF32_3D_Neg_Neon, runSigmoidF32_3D_Neg_Neon,
                refFactorySigmoid_3D_Neg, {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32}, {{1, 8, 2048}}, {true}, {{MemSpace(1, HandleType::CPP)}});