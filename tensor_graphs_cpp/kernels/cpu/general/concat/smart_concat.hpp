#pragma once
#include <algorithm>
#include <cstring>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchSmartConcat(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return isContiguous(output);
}

inline void runSmartConcat(const KernelContext &ctx)
{
    float *out_ptr = static_cast<float *>(ctx.outputs[0]);
    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[0]);
    const auto &out_shape = ctx.outViews[0].getShape();
    if (axis < 0)
        axis += out_shape.size();

    if (axis < 0 || axis >= static_cast<int32_t>(out_shape.size()))
    {
        Error::throw_err("[Smart_Concat_F32] Axis " + std::to_string(axis) + " is outside output rank (" +
                         std::to_string(out_shape.size()) + ").");
    }

    uint64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i)
        outer *= out_shape[i];
    for (int i = axis + 1; i < (int)out_shape.size(); ++i)
        inner *= out_shape[i];

    uint64_t total_elements = outer * inner * out_shape[axis];

    auto compute = [&](uint64_t o_start, uint64_t o_end) {
        for (uint64_t o = o_start; o < o_end; ++o)
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
    };

    if (total_elements < 262144)
    {
        compute(0, outer);
        return;
    }

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t chunk = (outer + num_threads - 1) / num_threads;
        compute(t * chunk, std::min((uint64_t)(t * chunk + chunk), outer));
    });
}

inline LogicalId refSmartConcat(const std::vector<LogicalId> &inputs, Graph &graph)
{
    std::vector<LogicalId> tensors(inputs.begin() + 1, inputs.end());
    LogicalId axis = inputs[0];
    return graph.concat(tensors, axis);
}

REGISTER_KERNEL("Smart_Concat_F32", 2, UINT32_MAX, matchSmartConcat, runSmartConcat, refSmartConcat, {},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::INT32, DType::FLOAT32},
                {{1}, {1, 32, 1, 128}}, {false, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});