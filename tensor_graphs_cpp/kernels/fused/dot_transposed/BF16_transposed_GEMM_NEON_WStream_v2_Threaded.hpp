// BF16 transposed GEMM, W streamed once, W block converted once (NEON,
// thread pool)
//
// Out[b, s, n] = sum_k X[b, s, k] * W[n, k], X fp32, W bf16 (converted
// exactly to fp32), fp32 accumulation.
//
// Like BF16_Transposed_GEMM_NEON_WStream_Threaded (W rows outermost so W is
// read from memory once), but each block of 4 W rows is converted to fp32 once
// into a per-task buffer and every X tile reads that buffer: the WStream
// kernel converted the block again for each 4-row X tile, and those
// conversions compete with the FMAs for the vector pipes.
#pragma once
#include "core/kernels.hpp"
#include "kernels/fused/dot_transposed/ref.hpp"
#include "core/types.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "core/common/thread_pool.hpp"

inline bool matchBF16TransposedGemmWStreamV2(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

namespace bf16t_wstream_v2
{
inline float bf16ToF32(uint16_t v)
{
    const uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// rows W rows (bf16, row stride K) -> fp32, row stride K.
inline void convertRows(const uint16_t *w, uint32_t rows, uint32_t K, float *out)
{
    const uint32_t k8 = K & ~7u;
    for (uint32_t r = 0; r < rows; ++r)
    {
        const uint16_t *src = w + static_cast<size_t>(r) * K;
        float *dst = out + static_cast<size_t>(r) * K;
        for (uint32_t k = 0; k < k8; k += 8)
        {
            const uint16x8_t v = vld1q_u16(src + k);
            vst1q_f32(dst + k, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16)));
            vst1q_f32(dst + k + 4, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(v), 16)));
        }
        for (uint32_t k = k8; k < K; ++k)
            dst[k] = bf16ToF32(src[k]);
    }
}

// 4 X rows x 4 W rows (fp32 W, row stride K).
inline void block4x4(const float *x, uint32_t ldx, const float *w, uint32_t K, float *out, uint32_t ldo)
{
    float32x4_t a[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            a[i][j] = vdupq_n_f32(0.0f);
    const uint32_t k4 = K & ~3u;
    for (uint32_t k = 0; k < k4; k += 4)
    {
        const float32x4_t w0 = vld1q_f32(w + 0 * K + k), w1 = vld1q_f32(w + 1 * K + k), w2 = vld1q_f32(w + 2 * K + k),
                          w3 = vld1q_f32(w + 3 * K + k);
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
                r[j] += x[i * ldx + k] * w[j * K + k];
        vst1q_f32(out + i * ldo, vld1q_f32(r));
    }
}

// 1 X row x 4 W rows.
inline void block1x4(const float *x, const float *w, uint32_t K, float *out)
{
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f), a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
    const uint32_t k4 = K & ~3u;
    for (uint32_t k = 0; k < k4; k += 4)
    {
        const float32x4_t xv = vld1q_f32(x + k);
        a0 = vfmaq_f32(a0, xv, vld1q_f32(w + 0 * K + k));
        a1 = vfmaq_f32(a1, xv, vld1q_f32(w + 1 * K + k));
        a2 = vfmaq_f32(a2, xv, vld1q_f32(w + 2 * K + k));
        a3 = vfmaq_f32(a3, xv, vld1q_f32(w + 3 * K + k));
    }
    float r[4] = {vaddvq_f32(a0), vaddvq_f32(a1), vaddvq_f32(a2), vaddvq_f32(a3)};
    for (uint32_t k = k4; k < K; ++k)
        for (int j = 0; j < 4; ++j)
            r[j] += x[k] * w[j * K + k];
    vst1q_f32(out, vld1q_f32(r));
}

// 1 X row x 1 W row.
inline float dot1(const float *x, const float *w, uint32_t K)
{
    float32x4_t a = vdupq_n_f32(0.0f);
    const uint32_t k4 = K & ~3u;
    for (uint32_t k = 0; k < k4; k += 4)
        a = vfmaq_f32(a, vld1q_f32(x + k), vld1q_f32(w + k));
    float r = vaddvq_f32(a);
    for (uint32_t k = k4; k < K; ++k)
        r += x[k] * w[k];
    return r;
}
} // namespace bf16t_wstream_v2

inline void runBF16TransposedGemmWStreamV2(const KernelContext &ctx)
{
    using namespace bf16t_wstream_v2;
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const uint16_t *W = static_cast<const uint16_t *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t B = ctx.inViews[0].getShape()[0];
    const uint32_t S = ctx.inViews[0].getShape()[1];
    const uint32_t K = ctx.inViews[0].getShape()[2];
    const uint32_t N = ctx.inViews[1].getShape()[0];

    const uint32_t blocks = (N + 3) / 4;
    ThreadPool &pool = ThreadPool::get();
    const uint32_t tasks = std::max(1u, std::min(blocks, pool.get_num_threads() * 4));
    const uint32_t per_task = (blocks + tasks - 1) / tasks;
    pool.parallel_for(tasks, [&](uint32_t t) {
        thread_local std::vector<float> wf;
        if (wf.size() < static_cast<size_t>(4) * K)
            wf.resize(static_cast<size_t>(4) * K);
        const uint32_t first = t * per_task;
        const uint32_t last = std::min(blocks, first + per_task);
        for (uint32_t blk = first; blk < last; ++blk)
        {
            const uint32_t n = blk * 4;
            const uint32_t rows = std::min(4u, N - n);
            convertRows(W + static_cast<size_t>(n) * K, rows, K, wf.data());
            for (uint32_t b = 0; b < B; ++b)
            {
                const float *xb = X + static_cast<size_t>(b) * S * K;
                float *ob = Out + static_cast<size_t>(b) * S * N;
                if (rows == 4)
                {
                    uint32_t s = 0;
                    for (; s + 4 <= S; s += 4)
                        block4x4(xb + static_cast<size_t>(s) * K, K, wf.data(), K, ob + static_cast<size_t>(s) * N + n, N);
                    for (; s < S; ++s)
                        block1x4(xb + static_cast<size_t>(s) * K, wf.data(), K, ob + static_cast<size_t>(s) * N + n);
                }
                else
                {
                    for (uint32_t j = 0; j < rows; ++j)
                        for (uint32_t s = 0; s < S; ++s)
                            ob[static_cast<size_t>(s) * N + n + j] =
                                dot1(xb + static_cast<size_t>(s) * K, wf.data() + static_cast<size_t>(j) * K, K);
                }
            }
        }
    });
}

REGISTER_KERNEL("BF16_Transposed_GEMM_NEON_WStream_v2_Threaded", 2, 2, matchBF16TransposedGemmWStreamV2,
                runBF16TransposedGemmWStreamV2, refFactoryDotTransposedBF16, {}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::BF16}, {{1, 8, 64}, {1024, 64}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
#endif
