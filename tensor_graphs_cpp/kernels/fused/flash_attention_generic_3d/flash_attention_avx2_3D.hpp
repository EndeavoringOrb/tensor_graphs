#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/flash_attention_generic_3d/ref.hpp"

#if defined(TG_HAS_AVX2)
#include <immintrin.h>

inline bool matchFlashAttentionAVX2_3D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.dtype != DType::FLOAT32)
        return false;

    const auto &shape = inputs[0].getShape();
    if (shape != std::vector<uint32_t>{12, 2340, 64})
        return false;
    if (inputs[1].getShape() != shape || inputs[2].getShape() != shape || output.getShape() != shape)
        return false;

    return true;
}

__attribute__((target("avx2,fma")))
inline __m256 flashExpAVX2(__m256 x)
{
    const __m256 lower = _mm256_set1_ps(-80.0f);
    x = _mm256_max_ps(x, lower);

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 n_float = _mm256_mul_ps(x, log2e);
    const __m256i n = _mm256_cvtps_epi32(n_float);
    const __m256 r = _mm256_fnmadd_ps(_mm256_cvtepi32_ps(n), ln2, x);

    // Degree-six exp polynomial on the reduced interval [-ln(2)/2, ln(2)/2].
    __m256 p = _mm256_set1_ps(1.0f / 720.0f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));

    const __m256i exponent = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(exponent));
}

__attribute__((target("avx2,fma")))
inline void runFlashAttentionAVX2Range(const float *q_base, const float *k_base, const float *v_base,
                                       float *out_base, uint32_t batch, uint32_t start_row, uint32_t end_row)
{
    constexpr uint32_t sequence = 2340;
    constexpr uint32_t width = 64;
    constexpr uint64_t batch_stride = static_cast<uint64_t>(sequence) * width;
    alignas(32) float scores[sequence];
    const float *q_batch = q_base + batch * batch_stride;
    const float *k_batch = k_base + batch * batch_stride;
    const float *v_batch = v_base + batch * batch_stride;

    for (uint32_t row = start_row; row < end_row; ++row)
    {
        const float *q_row = q_batch + row * width;

        __m256 q0 = _mm256_loadu_ps(q_row + 0);
        __m256 q1 = _mm256_loadu_ps(q_row + 8);
        __m256 q2 = _mm256_loadu_ps(q_row + 16);
        __m256 q3 = _mm256_loadu_ps(q_row + 24);
        __m256 q4 = _mm256_loadu_ps(q_row + 32);
        __m256 q5 = _mm256_loadu_ps(q_row + 40);
        __m256 q6 = _mm256_loadu_ps(q_row + 48);
        __m256 q7 = _mm256_loadu_ps(q_row + 56);

        float max_score = -1.0e30f;
        for (uint32_t j = 0; j < sequence; ++j)
        {
            const float *k_row = k_batch + static_cast<uint64_t>(j) * width;
            __m256 acc0 = _mm256_mul_ps(q0, _mm256_loadu_ps(k_row + 0));
            __m256 acc1 = _mm256_mul_ps(q1, _mm256_loadu_ps(k_row + 8));
            __m256 acc2 = _mm256_mul_ps(q2, _mm256_loadu_ps(k_row + 16));
            __m256 acc3 = _mm256_mul_ps(q3, _mm256_loadu_ps(k_row + 24));
            __m256 acc4 = _mm256_mul_ps(q4, _mm256_loadu_ps(k_row + 32));
            __m256 acc5 = _mm256_mul_ps(q5, _mm256_loadu_ps(k_row + 40));
            __m256 acc6 = _mm256_mul_ps(q6, _mm256_loadu_ps(k_row + 48));
            __m256 acc7 = _mm256_mul_ps(q7, _mm256_loadu_ps(k_row + 56));
            acc0 = _mm256_add_ps(acc0, acc1);
            acc2 = _mm256_add_ps(acc2, acc3);
            acc4 = _mm256_add_ps(acc4, acc5);
            acc6 = _mm256_add_ps(acc6, acc7);
            acc0 = _mm256_add_ps(acc0, acc2);
            acc4 = _mm256_add_ps(acc4, acc6);
            acc0 = _mm256_add_ps(acc0, acc4);
            __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc0), _mm256_extractf128_ps(acc0, 1));
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            const float score = _mm_cvtss_f32(sum);
            scores[j] = score;
            max_score = std::max(max_score, score);
        }

        uint32_t j = 0;
        for (; j + 8 <= sequence; j += 8)
        {
            const __m256 shifted = _mm256_sub_ps(_mm256_loadu_ps(scores + j), _mm256_set1_ps(max_score));
            _mm256_storeu_ps(scores + j, flashExpAVX2(shifted));
        }
        for (; j < sequence; ++j)
            scores[j] = std::exp(scores[j] - max_score);

        float sum = 0.0f;
        for (j = 0; j < sequence; ++j)
            sum += scores[j];
        const float inv_sum = 1.0f / sum;

        __m256 o0 = _mm256_setzero_ps();
        __m256 o1 = _mm256_setzero_ps();
        __m256 o2 = _mm256_setzero_ps();
        __m256 o3 = _mm256_setzero_ps();
        __m256 o4 = _mm256_setzero_ps();
        __m256 o5 = _mm256_setzero_ps();
        __m256 o6 = _mm256_setzero_ps();
        __m256 o7 = _mm256_setzero_ps();
        for (j = 0; j < sequence; ++j)
        {
            const __m256 p = _mm256_set1_ps(scores[j] * inv_sum);
            const float *v_row = v_batch + static_cast<uint64_t>(j) * width;
            o0 = _mm256_fmadd_ps(p, _mm256_loadu_ps(v_row + 0), o0);
            o1 = _mm256_fmadd_ps(p, _mm256_loadu_ps(v_row + 8), o1);
            o2 = _mm256_fmadd_ps(p, _mm256_loadu_ps(v_row + 16), o2);
            o3 = _mm256_fmadd_ps(p, _mm256_loadu_ps(v_row + 24), o3);
            o4 = _mm256_fmadd_ps(p, _mm256_loadu_ps(v_row + 32), o4);
            o5 = _mm256_fmadd_ps(p, _mm256_loadu_ps(v_row + 40), o5);
            o6 = _mm256_fmadd_ps(p, _mm256_loadu_ps(v_row + 48), o6);
            o7 = _mm256_fmadd_ps(p, _mm256_loadu_ps(v_row + 56), o7);
        }

        float *out_row = out_base + batch * batch_stride + row * width;
        _mm256_storeu_ps(out_row + 0, o0);
        _mm256_storeu_ps(out_row + 8, o1);
        _mm256_storeu_ps(out_row + 16, o2);
        _mm256_storeu_ps(out_row + 24, o3);
        _mm256_storeu_ps(out_row + 32, o4);
        _mm256_storeu_ps(out_row + 40, o5);
        _mm256_storeu_ps(out_row + 48, o6);
        _mm256_storeu_ps(out_row + 56, o7);
    }
}

inline void runFlashAttentionAVX2_3D(const KernelContext &ctx)
{
    const float *q = static_cast<const float *>(ctx.inputs[0]);
    const float *k = static_cast<const float *>(ctx.inputs[1]);
    const float *v = static_cast<const float *>(ctx.inputs[2]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    constexpr uint32_t sequence = 2340;
    constexpr uint32_t batches = 12;
    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;
    num_threads = std::min(num_threads, sequence);

    for (uint32_t batch = 0; batch < batches; ++batch)
    {
        ThreadPool::get().parallel_for(num_threads, [=](uint32_t thread_idx) {
            const uint32_t rows_per_thread = (sequence + num_threads - 1) / num_threads;
            const uint32_t start = thread_idx * rows_per_thread;
            const uint32_t end = std::min(start + rows_per_thread, sequence);
            if (start < end)
                runFlashAttentionAVX2Range(q, k, v, out, batch, start, end);
        });
    }
}

REGISTER_KERNEL("Flash_Attention_AVX2_3D", 3, 3, matchFlashAttentionAVX2_3D, runFlashAttentionAVX2_3D,
                refFactoryFlashAttentionGeneric3D, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32}, {{12, 2340, 64}, {12, 2340, 64}, {12, 2340, 64}},
                {true, true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif
