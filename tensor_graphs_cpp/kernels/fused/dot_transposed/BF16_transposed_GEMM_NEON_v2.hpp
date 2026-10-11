#pragma once
#include <algorithm>
#include <thread>
#include <vector>

#include "core/kernels.hpp"
#include "kernels/fused/dot_transposed/ref.hpp"
#include "core/types.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchBF16TransposedGEMM_v2(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    auto sX = inputs[0].getShape(); // [B, S, K]
    auto sW = inputs[1].getShape(); // [N, K]
    auto sO = output.getShape();    // [B, S, N]
    if (sX.size() != 3 || sW.size() != 2 || sO.size() != 3)
        return false;
    if (sX[2] != sW[1] || sO[2] != sW[0])
        return false;
    // Enforce contiguous output to guarantee cache-friendly write patterns
    return isContiguous(output);
}

inline void runBF16TransposedGEMM_v2(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const uint16_t *W = static_cast<const uint16_t *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);
    uint32_t B = ctx.inViews[0].getShape()[0];
    uint32_t S = ctx.inViews[0].getShape()[1];
    uint32_t K = ctx.inViews[0].getShape()[2];
    uint32_t N = ctx.inViews[1].getShape()[0];

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    if (num_threads > N)
        num_threads = N; // Avoid oversubscribing tiny N

    std::vector<std::thread> workers;
    uint32_t n_per_thread = (N + num_threads - 1) / num_threads;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]() {
            uint32_t start_n = t * n_per_thread;
            uint32_t end_n = std::min(start_n + n_per_thread, N);

            // Loop order changed to ensure contiguous writes to Out
            for (uint32_t b = 0; b < B; ++b)
            {
                for (uint32_t s = 0; s < S; ++s)
                {
                    const float *x_row = X + (b * S * K) + (s * K);
                    // Point to the start of this thread's chunk for the current (b,s)
                    float *out_row = Out + (b * S * N) + (s * N) + start_n;

                    for (uint32_t n = start_n; n < end_n; ++n)
                    {
                        const uint16_t *w_row = W + n * K;
                        float32x4_t acc = vdupq_n_f32(0.0f);
                        uint32_t k = 0;

                        // NEON SIMD: Process 4 K-elements at a time
                        for (; k + 4 <= K; k += 4)
                        {
                            uint16x4_t vbf16 = vld1_u16(w_row + k);
                            float32x4_t vw = vreinterpretq_f32_u32(vshll_n_u16(vbf16, 16));
                            float32x4_t vx = vld1q_f32(x_row + k);
                            acc = vfmaq_f32(acc, vx, vw);
                        }
                        float sum = vaddvq_f32(acc);

                        // Tail loop
                        for (; k < K; ++k)
                        {
                            uint32_t bits = (uint32_t)w_row[k] << 16;
                            float wf;
                            std::memcpy(&wf, &bits, 4);
                            sum += x_row[k] * wf;
                        }
                        // Contiguous write!
                        *out_row++ = sum;
                    }
                }
            }
        });
    }
    for (auto &worker : workers)
        worker.join();
}



REGISTER_KERNEL("BF16_Transposed_GEMM_NEON_v2", 2, 2, matchBF16TransposedGEMM_v2, runBF16TransposedGEMM_v2,
                refFactoryDotTransposedBF16, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::BF16}, {{1, 8, 64}, {1024, 64}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif
