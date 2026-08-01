#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <cstring>
#include <vector>

#include "core/common/thread_pool.hpp"

inline bool matchConcatF32_Fast(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runConcatF32_Fast(const KernelContext &ctx)
{
    float *out_ptr = static_cast<float *>(ctx.outputs[0]);
    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[0]);
    const auto &out_shape = ctx.outViews[0].getShape();
    if (axis < 0)
        axis += out_shape.size();

    uint64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i)
        outer *= out_shape[i];
    for (int i = axis + 1; i < (int)out_shape.size(); ++i)
        inner *= out_shape[i];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t chunk = (outer + num_threads - 1) / num_threads;
        uint32_t o_start = t * chunk;
        uint32_t o_end = std::min(o_start + chunk, (uint32_t)outer);
        for (uint32_t o = o_start; o < o_end; ++o)
        {
            uint64_t out_axis_offset = 0;
            for (uint64_t n = 1; n < ctx.inputs.size(); ++n)
            {
                uint32_t axis_dim = ctx.inViews[n].getShape()[axis];
                const float *src = static_cast<const float *>(ctx.inputs[n]) + (o * axis_dim * inner);
                float *dst = out_ptr + (o * out_shape[axis] * inner) + (out_axis_offset * inner);
                std::memcpy(dst, src, axis_dim * inner * sizeof(float));
                out_axis_offset += axis_dim;
            }
        }
    });
}

inline LogicalId refFactoryConcatF32_Fast(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() < 2)
        Error::throw_err("Concat Fast requires at least 2 inputs");

    std::vector<LogicalId> tensors(inputs.begin() + 1, inputs.end());
    LogicalId axis = inputs[0];
    return graph.concat(tensors, axis);
}

REGISTER_KERNEL("Concat_F32_Fast", 2, UINT32_MAX, matchConcatF32_Fast, runConcatF32_Fast, refFactoryConcatF32_Fast,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::INT32, DType::FLOAT32},
                {{1}, {1, 24, 1536, 128}}, {false, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif