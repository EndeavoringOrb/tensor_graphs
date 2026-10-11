#pragma once
#include <algorithm>
#include <thread>
#include <vector>

#include "core/kernels.hpp"
#include "kernels/fused/dot_transposed/ref.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchBF16TransposedGEMM(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    auto sX = inputs[0].getShape(); // [B, S, K]
    auto sW = inputs[1].getShape(); // [N, K]
    auto sO = output.getShape();    // [B, S, N]
    return sX.size() == 3 && sW.size() == 2 && sO.size() == 3 && sX[2] == sW[1] && sO[2] == sW[0];
}

inline void runBF16TransposedGEMM(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const uint16_t *W = static_cast<const uint16_t *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    uint32_t B = ctx.inViews[0].getShape()[0];
    uint32_t S = ctx.inViews[0].getShape()[1];
    uint32_t K = ctx.inViews[0].getShape()[2];
    uint32_t N = ctx.inViews[1].getShape()[0];

    uint32_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]() {
            uint32_t start_n = (N * t) / num_threads;
            uint32_t end_n = (N * (t + 1)) / num_threads;

            for (uint32_t n = start_n; n < end_n; ++n)
            {
                const uint16_t *w_row = W + n * K;
                for (uint32_t b = 0; b < B; ++b)
                {
                    for (uint32_t s = 0; s < S; ++s)
                    {
                        const float *x_row = X + (b * S * K) + (s * K);
                        float32x4_t acc = vdupq_n_f32(0.0f);
                        uint32_t k = 0;
                        for (; k + 4 <= K; k += 4)
                        {
                            uint16x4_t vbf16 = vld1_u16(w_row + k);
                            float32x4_t vw = vreinterpretq_f32_u32(vshll_n_u16(vbf16, 16));
                            float32x4_t vx = vld1q_f32(x_row + k);
                            acc = vfmaq_f32(acc, vx, vw);
                        }
                        float sum = vaddvq_f32(acc);
                        for (; k < K; ++k)
                        {
                            uint32_t bits = (uint32_t)w_row[k] << 16;
                            float wf;
                            std::memcpy(&wf, &bits, 4);
                            sum += x_row[k] * wf;
                        }
                        Out[(b * S * N) + (s * N) + n] = sum;
                    }
                }
            }
        });
    }
    for (auto &worker : workers)
        worker.join();
}



REGISTER_KERNEL("BF16_Transposed_GEMM_NEON", 2, 2, matchBF16TransposedGEMM, runBF16TransposedGEMM,
                refFactoryDotTransposedBF16, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::BF16}, {{1, 8, 64}, {1024, 64}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif
