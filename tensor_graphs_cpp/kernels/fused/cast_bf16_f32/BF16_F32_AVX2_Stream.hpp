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
// TODO: do in hardware detection
#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#include <immintrin.h>
#define TG_HAS_AVX2 1
#endif

/**
 * KERNEL: Cast_BF16_F32_AVX2_Stream
 *
 * Highly optimized BF16 -> FP32 conversion using AVX2 SIMD and non-temporal streaming stores (_mm256_stream_ps).
 * Multi-threaded across all available CPU threads via ThreadPool without input-size branching.
 * Writes bypass the CPU cache hierarchy directly to write-combining buffers, avoiding write-allocate (RFO) overhead.
 */

inline bool matchCastBF16_F32_AVX2_Stream(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

#if defined(TG_HAS_AVX2)
__attribute__((always_inline, target("avx2,fma")))
#endif
inline void castBf16F32Avx2StreamRange(const uint16_t *src, float *dst, uint64_t start, uint64_t end)
{
#if defined(TG_HAS_AVX2)
    uint64_t i = start;

    // Peel head elements until dst + i is 32-byte aligned for _mm256_stream_ps
    while (i < end && ((reinterpret_cast<uintptr_t>(dst + i) & 31) != 0))
    {
        uint32_t val32 = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &val32, sizeof(float));
        ++i;
    }

    // Process 32 elements at a time (64 bytes BF16 in, 128 bytes FP32 out = 2 cache lines out)
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

    // Process remaining 16 elements (32 bytes BF16 in, 64 bytes FP32 out = 1 cache line out)
    for (; i + 16 <= end; i += 16)
    {
        __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + i));
        __m128i lo = _mm256_castsi256_si128(raw);
        __m128i hi = _mm256_extracti128_si256(raw, 1);

        __m256i f32_0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(lo), 16);
        __m256i f32_1 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(hi), 16);

        _mm256_stream_ps(dst + i, _mm256_castsi256_ps(f32_0));
        _mm256_stream_ps(dst + i + 8, _mm256_castsi256_ps(f32_1));
    }

    // Process remaining 8 elements (16 bytes BF16 in, 32 bytes FP32 out)
    for (; i + 8 <= end; i += 8)
    {
        __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + i));
        __m256i f32_0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16);
        _mm256_stream_ps(dst + i, _mm256_castsi256_ps(f32_0));
    }

    // Scalar tail loop
    for (; i < end; ++i)
    {
        uint32_t val32 = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &val32, sizeof(float));
    }

    // Store fence to ensure streaming stores are visible to subsequent reads
    _mm_sfence();
#else
    for (uint64_t i = start; i < end; ++i)
    {
        uint32_t val32 = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &val32, sizeof(float));
    }
#endif
}

inline void runCastBF16_F32_AVX2_Stream(const KernelContext &ctx)
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

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint64_t start = t * chunk;
        if (start >= num_elements)
            return;
        uint64_t end = std::min(start + chunk, num_elements);
        castBf16F32Avx2StreamRange(src, dst, start, end);
    });
}



REGISTER_KERNEL("Cast_BF16_F32_AVX2_Stream", 1, 1, matchCastBF16_F32_AVX2_Stream, runCastBF16_F32_AVX2_Stream,
                refFactoryCastBF16_F32, {0}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::BF16}, {{2048, 640}}, {true},
                {{MemSpace(1, HandleType::CPP)}});

#if defined(TG_HAS_AVX2)
#pragma GCC pop_options
#endif
