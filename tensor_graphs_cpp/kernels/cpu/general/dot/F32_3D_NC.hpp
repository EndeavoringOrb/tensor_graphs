#pragma once
#include <algorithm>
#include <cstring>
#include <thread>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/cpu/utils.hpp"

inline bool matchDotF32_3D_NC(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &sA = inputs[0].getShape();
    const auto &sB = inputs[1].getShape();
    const auto &sC = output.getShape();

    if (sA.size() != 3 || sB.size() != 3 || sC.size() != 3)
        return false;
    // A: [B, M, K], B: [B, K, N], C: [B, M, N]
    if (sA[0] != sB[0] || sA[2] != sB[1])
        return false;
    if (sC[0] != sA[0] || sC[1] != sA[1] || sC[2] != sB[2])
        return false;

    if (output.dtype != DType::FLOAT32)
        return false;

    // Verify that the inner 2 dimensions of A, B, and output are contiguous
    if (!isLast2DimsContiguous(inputs[0]) || !isLast2DimsContiguous(inputs[1]) || !isLast2DimsContiguous(output))
        return false;

    return true;
}

inline void runDotF32_3D_NC(const KernelContext &ctx)
{
    const float *A = static_cast<const float *>(ctx.inputs[0]);
    const float *B = static_cast<const float *>(ctx.inputs[1]);
    float *C = static_cast<float *>(ctx.outputs[0]);

    const auto &viewA = ctx.inViews[0];
    const auto &viewB = ctx.inViews[1];
    const auto &viewC = ctx.outViews[0];

    uint32_t B_count = viewA.getShape()[0];
    uint32_t M = viewA.getShape()[1];
    uint32_t K = viewA.getShape()[2];
    uint32_t N = viewB.getShape()[2];

    uint64_t strideA_B = viewA.strides[0];
    uint64_t strideB_B = viewB.strides[0];
    uint64_t strideC_B = viewC.strides[0];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    uint32_t total_rows = B_count * M;
    if (num_threads > total_rows)
        num_threads = total_rows;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t)
                                   {
        uint32_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
        uint32_t start_row = t * rows_per_thread;
        uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);

        for (uint32_t row_idx = start_row; row_idx < end_row; ++row_idx)
        {
            uint32_t b = row_idx / M;
            uint32_t m = row_idx % M;

            const float *a_row = A + b * strideA_B + m * K;
            const float *b_mat = B + b * strideB_B;
            float *c_row = C + b * strideC_B + m * N;

            std::memset(c_row, 0, N * sizeof(float));

            for (uint32_t k = 0; k < K; ++k)
            {
                float a_val = a_row[k];
                const float *b_row = b_mat + k * N;

                for (uint32_t n = 0; n < N; ++n)
                {
                    c_row[n] += a_val * b_row[n];
                }
            }
        } });
}

inline LogicalId refDotF32_3D_NC(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.dot(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Dot_F32_3D_NC", 2, 2, matchDotF32_3D_NC, runDotF32_3D_NC, refDotF32_3D_NC,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32}, {{1, 8, 8}, {1, 8, 8}},
                {false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});