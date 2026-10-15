// BF16 transposed GEMM, W streamed once (NEON, thread pool)
//
// Out[b, s, n] = sum_k X[b, s, k] * W[n, k], X fp32, W bf16 (converted
// exactly to fp32 in registers), fp32 accumulation.
//
// For few rows of X (decode / small prefill blocks, e.g. the LM head over a
// 262144 x 640 vocabulary) the cost is reading W. The v5 kernel walks X tiles
// outermost, so every 4-row X tile streams the thread's whole share of W from
// memory again. Here a block of 4 W rows (4 x K bf16, 5 KB at K = 640) is the
// outer unit: every X row is multiplied against it while it sits in L1, so W
// is read from memory once regardless of the number of X rows.
#pragma once
#include "core/kernels.hpp"
#include "kernels/fused/dot_transposed/ref.hpp"
#include "core/types.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>

#include <algorithm>
#include <cstring>

#include "core/common/thread_pool.hpp"

inline bool matchBF16TransposedGemmWStream(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    auto sX = inputs[0].getShape();
    auto sW = inputs[1].getShape();
    auto sO = output.getShape();
    if (sX.size() != 3 || sW.size() != 2 || sO.size() != 3)
        return false;
    if (sX[2] != sW[1] || sO[2] != sW[0] || sO[0] != sX[0] || sO[1] != sX[1])
        return false;
    return isContiguous(output);
}

namespace bf16t_wstream
{
inline float32x4_t loadW4(const uint16_t *p)
{
    return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16));
}

inline float bf16ToF32(uint16_t v)
{
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// 4 X rows x 4 W rows.
inline void block4x4(const float *x, uint32_t ldx, const uint16_t *w, uint32_t K, float *out, uint32_t ldo)
{
    float32x4_t a[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            a[i][j] = vdupq_n_f32(0.0f);
    const uint32_t k4 = K & ~3u;
    for (uint32_t k = 0; k < k4; k += 4)
    {
        const float32x4_t w0 = loadW4(w + 0 * K + k), w1 = loadW4(w + 1 * K + k), w2 = loadW4(w + 2 * K + k),
                          w3 = loadW4(w + 3 * K + k);
        for (int i = 0; i < 4; ++i)
        {
            const float32x4_t xv = vld1q_f32(x + i * ldx + k);
            a[i][0] = vfmaq_f32(a[i][0], xv, w0);
            a[i][1] = vfmaq_f32(a[i][1], xv, w1);
            a[i][2] = vfmaq_f32(a[i][2], xv, w2);
            a[i][3] = vfmaq_f32(a[i][3], xv, w3);
        }
    }
    for (int i = 0; i < 4; ++i)
    {
        float r[4] = {vaddvq_f32(a[i][0]), vaddvq_f32(a[i][1]), vaddvq_f32(a[i][2]), vaddvq_f32(a[i][3])};
        for (uint32_t k = k4; k < K; ++k)
            for (int j = 0; j < 4; ++j)
                r[j] += x[i * ldx + k] * bf16ToF32(w[j * K + k]);
        vst1q_f32(out + i * ldo, vld1q_f32(r));
    }
}

// 1 X row x 4 W rows.
inline void block1x4(const float *x, const uint16_t *w, uint32_t K, float *out)
{
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f), a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    const uint32_t k4 = K & ~3u;
    for (uint32_t k = 0; k < k4; k += 4)
    {
        const float32x4_t xv = vld1q_f32(x + k);
        a0 = vfmaq_f32(a0, xv, loadW4(w + 0 * K + k));
        a1 = vfmaq_f32(a1, xv, loadW4(w + 1 * K + k));
        a2 = vfmaq_f32(a2, xv, loadW4(w + 2 * K + k));
        a3 = vfmaq_f32(a3, xv, loadW4(w + 3 * K + k));
    }
    float r[4] = {vaddvq_f32(a0), vaddvq_f32(a1), vaddvq_f32(a2), vaddvq_f32(a3)};
    for (uint32_t k = k4; k < K; ++k)
        for (int j = 0; j < 4; ++j)
            r[j] += x[k] * bf16ToF32(w[j * K + k]);
    vst1q_f32(out, vld1q_f32(r));
}

// 1 X row x 1 W row (W rows past the last full block of 4).
inline float dot1(const float *x, const uint16_t *w, uint32_t K)
{
    float32x4_t a = vdupq_n_f32(0.0f);
    const uint32_t k4 = K & ~3u;
    for (uint32_t k = 0; k < k4; k += 4)
        a = vfmaq_f32(a, vld1q_f32(x + k), loadW4(w + k));
    float r = vaddvq_f32(a);
    for (uint32_t k = k4; k < K; ++k)
        r += x[k] * bf16ToF32(w[k]);
    return r;
}
} // namespace bf16t_wstream

inline void runBF16TransposedGemmWStream(const KernelContext &ctx)
{
    using namespace bf16t_wstream;
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const uint16_t *W = static_cast<const uint16_t *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t B = ctx.inViews[0].getShape()[0];
    const uint32_t S = ctx.inViews[0].getShape()[1];
    const uint32_t K = ctx.inViews[0].getShape()[2];
    const uint32_t N = ctx.inViews[1].getShape()[0];

    const uint32_t blocks = (N + 3) / 4;
    ThreadPool &pool = ThreadPool::get();
    // Several tasks per thread so uneven progress balances out.
    const uint32_t tasks = std::max(1u, std::min(blocks, pool.get_num_threads() * 4));
    const uint32_t per_task = (blocks + tasks - 1) / tasks;
    pool.parallel_for(tasks, [&](uint32_t t) {
        const uint32_t first = t * per_task;
        const uint32_t last = std::min(blocks, first + per_task);
        for (uint32_t blk = first; blk < last; ++blk)
        {
            const uint32_t n = blk * 4;
            const uint16_t *w = W + static_cast<size_t>(n) * K;
            for (uint32_t b = 0; b < B; ++b)
            {
                const float *xb = X + static_cast<size_t>(b) * S * K;
                float *ob = Out + static_cast<size_t>(b) * S * N;
                if (n + 4 <= N)
                {
                    uint32_t s = 0;
                    for (; s + 4 <= S; s += 4)
                        block4x4(xb + static_cast<size_t>(s) * K, K, w, K, ob + static_cast<size_t>(s) * N + n, N);
                    for (; s < S; ++s)
                        block1x4(xb + static_cast<size_t>(s) * K, w, K, ob + static_cast<size_t>(s) * N + n);
                }
                else
                {
                    for (uint32_t nn = n; nn < N; ++nn)
                        for (uint32_t s = 0; s < S; ++s)
                            ob[static_cast<size_t>(s) * N + nn] =
                                dot1(xb + static_cast<size_t>(s) * K, W + static_cast<size_t>(nn) * K, K);
                }
            }
        }
    });
}

REGISTER_KERNEL("BF16_Transposed_GEMM_NEON_WStream_Threaded", 2, 2, matchBF16TransposedGemmWStream,
                runBF16TransposedGemmWStream, refFactoryDotTransposedBF16, {}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::BF16}, {{1, 8, 64}, {1024, 64}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif
