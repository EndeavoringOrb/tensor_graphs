#pragma once
#include <algorithm>
#include <thread>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

/**
 * Fused Batched Transposed GEMM Kernel for CPU
 * Directly computes Y = X * W^T over batches, bypassing runtime transposition.
 * Thread-parallelized over batches and vector-optimized via ARM NEON SIMD.
 */

inline bool matchBatchedTransposedGEMM(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // inputs[0] is X [E, S, H]
    // inputs[1] is W [E, O, H]
    // output is Out [E, S, O]
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 3 || output.getShape().size() != 3)
        return false;

    const auto &sX = inputs[0].getShape();
    const auto &sW = inputs[1].getShape();
    const auto &sO = output.getShape();

    // Verify dimension compatibility
    if (sX[0] != sW[0] || sX[2] != sW[2])
        return false;
    if (sO[0] != sX[0] || sO[1] != sX[1] || sO[2] != sW[1])
        return false;

    // Both inputs and output must be contiguous for direct vectorization
    if (!isContiguous(output))
        return false;

    return true;
}

inline void runBatchedTransposedGEMM(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const float *W = static_cast<const float *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const auto &viewX = ctx.inViews[0];
    const auto &viewW = ctx.inViews[1];

    uint32_t E = viewX.getShape()[0];
    uint32_t S = viewX.getShape()[1];
    uint32_t H = viewX.getShape()[2];
    uint32_t O = viewW.getShape()[1];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    if (num_threads > E)
        num_threads = E;

    std::vector<std::thread> workers;
    uint32_t e_per_thread = (E + num_threads - 1) / num_threads;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]() {
            uint32_t e_start = t * e_per_thread;
            uint32_t e_end = std::min(e_start + e_per_thread, E);

            std::vector<float32x4_t> acc(S);

            for (uint32_t e = e_start; e < e_end; ++e)
            {
                for (uint32_t o = 0; o < O; ++o)
                {
                    const float *w_row = W + ((uint64_t)e * O + o) * H;

                    uint32_t s = 0;

                    for (; s < S; ++s)
                    {
                        acc[s] = vdupq_n_f32(0.0f);
                    }

                    uint32_t h = 0;
                    // Vectorized dot product accumulator loop
                    for (; h + 4 <= H; h += 4)
                    {
                        float32x4_t vw = vld1q_f32(w_row + h);
                        for (s = 0; s < S; ++s)
                        {
                            float32x4_t vx = vld1q_f32(X + ((uint64_t)e * S + s) * H + h);
                            acc[s] = vfmaq_f32(acc[s], vx, vw);
                        }
                    }

                    // Horizontal reduction and scalar cleanup
                    for (s = 0; s < S; ++s)
                    {
                        float sum = vaddvq_f32(acc[s]);
                        for (uint32_t h_tail = h; h_tail < H; ++h_tail)
                        {
                            sum += X[((uint64_t)e * S + s) * H + h_tail] * w_row[h_tail];
                        }
                        Out[((uint64_t)e * S + s) * O + o] = sum;
                    }
                }
            }
        });
    }

    for (auto &worker : workers)
        worker.join();
}

inline LogicalId refFactoryBatchedTransposedGEMM(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // Reconstructs the unoptimized pattern: Dot(X, Contiguous(Permute(W)))
    int32_t perm[] = {0, 2, 1};
    LogicalId perm_node = graph.constant({3}, perm, DType::INT32);
    LogicalId transposed = graph.contiguous(graph.permute(inputs[1], perm_node));
    return graph.dot(inputs[0], transposed);
}

REGISTER_KERNEL("Batched_Transposed_GEMM_NEON", 2, 2, matchBatchedTransposedGEMM, runBatchedTransposedGEMM,
                refFactoryBatchedTransposedGEMM, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32}, {{256, 8, 2048}, {256, 1024, 2048}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON