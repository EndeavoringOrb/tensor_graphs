#pragma once
#include <algorithm>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchAddNC_F32_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 3)
        return false;
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    return true;
}

inline void runAddNC_F32_3D(const KernelContext &ctx)
{
    const float *a = static_cast<const float *>(ctx.inputs[0]);
    const float *b = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    uint32_t B = ctx.inViews[0].getShape()[0];
    uint32_t M = ctx.inViews[0].getShape()[1];
    uint32_t N = ctx.inViews[0].getShape()[2];

    uint64_t a_str0 = ctx.inViews[0].strides[0];
    uint64_t a_str1 = ctx.inViews[0].strides[1];
    uint64_t a_str2 = ctx.inViews[0].strides[2];

    uint64_t b_str0 = ctx.inViews[1].strides[0];
    uint64_t b_str1 = ctx.inViews[1].strides[1];
    uint64_t b_str2 = ctx.inViews[1].strides[2];

    uint64_t out_str0 = ctx.outViews[0].strides[0];
    uint64_t out_str1 = ctx.outViews[0].strides[1];
    uint64_t out_str2 = ctx.outViews[0].strides[2];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t b_per_thread = (B + num_threads - 1) / num_threads;
        if (b_per_thread == 0)
            b_per_thread = 1;
        uint32_t start_b = t * b_per_thread;
        uint32_t end_b = std::min(start_b + b_per_thread, B);

        for (uint32_t i = start_b; i < end_b; ++i)
        {
            for (uint32_t j = 0; j < M; ++j)
            {
                const float *a_row = a + i * a_str0 + j * a_str1;
                const float *b_row = b + i * b_str0 + j * b_str1;
                float *out_row = out + i * out_str0 + j * out_str1;

                if (a_str2 == 1 && b_str2 == 1 && out_str2 == 1)
                {
                    for (uint32_t k = 0; k < N; ++k)
                    {
                        out_row[k] = a_row[k] + b_row[k];
                    }
                }
                else
                {
                    for (uint32_t k = 0; k < N; ++k)
                    {
                        out_row[k * out_str2] = a_row[k * a_str2] + b_row[k * b_str2];
                    }
                }
            }
        }
    });
}

inline LogicalId refFactoryAddNC_F32_3D(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.add(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Add_NC_F32_3D", 2, 2, matchAddNC_F32_3D, runAddNC_F32_3D, refFactoryAddNC_F32_3D,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 1, 1}, {1, 1, 1}}, {false, false},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});