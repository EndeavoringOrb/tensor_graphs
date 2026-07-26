#pragma once
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchPowF32_ND_Scalar_Threaded(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != 1)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    return isContiguous(output);
}

inline void runPowF32_ND_Scalar_Threaded(const KernelContext &ctx)
{
    const float *dataND = static_cast<const float *>(ctx.inputs[0]);
    float scalarValue = *static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    uint64_t totalElements = countElements(ctx.inViews[0].getShape());
    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    uint64_t chunk = (totalElements + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]() {
            uint64_t start = t * chunk;
            uint64_t end = std::min(start + chunk, totalElements);

            // Fast paths for common powers
            if (scalarValue == 0.5f)
            {
                for (uint64_t i = start; i < end; ++i)
                    out[i] = std::sqrt(dataND[i]);
            }
            else if (scalarValue == 2.0f)
            {
                for (uint64_t i = start; i < end; ++i)
                    out[i] = dataND[i] * dataND[i];
            }
            else
            {
                for (uint64_t i = start; i < end; ++i)
                    out[i] = std::pow(dataND[i], scalarValue);
            }
        });
    }
    for (auto &w : workers)
        w.join();
}

inline LogicalId refFactoryPowND_Scalar_Threaded(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId idND = inputs[0];
    LogicalId idScalar = inputs[1];
    auto shapeND = graph.getNode(idND).getShape();

    std::vector<int32_t> ones(shapeND.size(), 1);
    LogicalId reshaped = graph.reshape(idScalar, graph.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));

    LogicalId out = reshaped;
    for (uint64_t i = 0; i < shapeND.size(); ++i)
    {
        if (shapeND[i] > 1)
        {
            int32_t rep = shapeND[i];
            int32_t ax = i;
            out = graph.repeat(out, graph.constant({1}, &rep, DType::INT32), graph.constant({1}, &ax, DType::INT32));
        }
    }
    return graph.pow(idND, out);
}

REGISTER_KERNEL("Pow_ND_Scalar_Threaded", 2, 2, matchPowF32_ND_Scalar_Threaded, runPowF32_ND_Scalar_Threaded,
                refFactoryPowND_Scalar_Threaded, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32}, {{2, 128}, {1}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});