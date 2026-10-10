#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/cast_bf16_f32/ref.hpp"
#if defined(TG_HAS_AVX2)
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#include <immintrin.h>

/**
 * KERNEL: Cast_BF16_F32_AVX2_Stream_32
 *
 * Highly optimized BF16 -> FP32 conversion using AVX2 SIMD and non-temporal streaming stores (_mm256_stream_ps).
 * Multi-threaded across all available CPU threads via ThreadPool without input-size branching.
 * Writes bypass the CPU cache hierarchy directly to write-combining buffers, avoiding write-allocate (RFO) overhead.
 */

inline bool matchCastBF16_F32_AVX2_Stream_32(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    // CPP arenas are 4096-byte aligned and the planner assigns CPP buffers on
    // 64-byte boundaries. Require full 32-element blocks so worker starts and
    // the final store are aligned without peeling or scalar/SIMD tails.
    if ((countElements(output.getShape()) & 31ULL) != 0)
        return false;
    return true;
}

__attribute__((always_inline, target("avx2,fma")))
inline void castBf16F32Avx2StreamRange_32(const uint16_t *src, float *dst, uint64_t start, uint64_t end)
{
    // The matcher guarantees a multiple of 32 elements. Worker chunks are
    // rounded up to 32 elements, and CPP output pointers are 64-byte aligned.
    uint64_t i = start;
    for (; i + 32 <= end; i += 32)
    {
        __m256i raw0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
        __m256i raw1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i + 16));

        __m128i lo0 = _mm256_castsi256_si128(raw0);
        __m128i hi0 = _mm256_extracti128_si256(raw0, 1);
        __m128i lo1 = _mm256_castsi256_si128(raw1);
        __m128i hi1 = _mm256_extracti128_si256(raw1, 1);

        __m256i f32_0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(lo0), 16);
        __m256i f32_1 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(hi0), 16);
        __m256i f32_2 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(lo1), 16);
        __m256i f32_3 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(hi1), 16);

        _mm256_stream_ps(dst + i, _mm256_castsi256_ps(f32_0));
        _mm256_stream_ps(dst + i + 8, _mm256_castsi256_ps(f32_1));
        _mm256_stream_ps(dst + i + 16, _mm256_castsi256_ps(f32_2));
        _mm256_stream_ps(dst + i + 24, _mm256_castsi256_ps(f32_3));
    }

    // Store fence to ensure streaming stores are visible to subsequent reads
    _mm_sfence();
}

inline void runCastBF16_F32_AVX2_Stream_32(const KernelContext &ctx)
{
    const uint16_t *src = static_cast<const uint16_t *>(ctx.inputs[0]);
    float *dst = static_cast<float *>(ctx.outputs[0]);

    uint64_t num_elements = countElements(ctx.outViews[0].getShape());
    if (num_elements == 0)
        return;

    uint32_t num_threads = ThreadPool::get().get_num_threads();
    if (num_threads == 0)
        num_threads = std::max(1U, std::thread::hardware_concurrency());

    uint64_t chunk = (num_elements + num_threads - 1) / num_threads;
    chunk = (chunk + 31) & ~31ULL;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) __attribute__((target("avx2,fma"))) {
        uint64_t start = t * chunk;
        if (start >= num_elements)
            return;
        uint64_t end = std::min(start + chunk, num_elements);
        castBf16F32Avx2StreamRange_32(src, dst, start, end);
    });
}



REGISTER_KERNEL("Cast_BF16_F32_AVX2_Stream_32", 1, 1, matchCastBF16_F32_AVX2_Stream_32, runCastBF16_F32_AVX2_Stream_32,
                refFactoryCastBF16_F32, {0}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::BF16}, {{2048, 640}}, {true},
                {{MemSpace(1, HandleType::CPP)}});

#pragma GCC pop_options
#endif
