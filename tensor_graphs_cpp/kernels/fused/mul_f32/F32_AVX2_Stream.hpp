#pragma once

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/mul_f32/ref.hpp"
#if defined(TG_HAS_AVX2)
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#include <immintrin.h>

/**
 * KERNEL: Mul_F32_AVX2_Stream
 *
 * Element-wise F32 multiplication for CPU using AVX2 SIMD and streaming non-temporal stores.
 * Bypasses CPU cache hierarchy (L1/L2/L3) via _mm256_stream_ps to prevent cache pollution
 * and eliminate write-allocate (RFO) bus traffic, maximizing memory bandwidth.
 */

inline bool matchMulF32_AVX2_Stream(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

__attribute__((always_inline, target("avx2,fma")))
inline void mulF32Avx2StreamRange(const float *a, const float *b, float *out, uint64_t start, uint64_t end)
{
    uint64_t i = start;

    // Peel head elements until out + i is 32-byte aligned for _mm256_stream_ps
    while (i < end && ((reinterpret_cast<uintptr_t>(out + i) & 31) != 0))
    {
        out[i] = a[i] * b[i];
        ++i;
    }

    // Main vector loop: unroll by 16 floats (64 bytes = 1 full CPU cache line)
    // Writes directly to write-combining buffers, bypassing L1/L2/L3 cache
    for (; i + 16 <= end; i += 16)
    {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);

        __m256 vres0 = _mm256_mul_ps(va0, vb0);
        __m256 vres1 = _mm256_mul_ps(va1, vb1);

        _mm256_stream_ps(out + i, vres0);
        _mm256_stream_ps(out + i + 8, vres1);
    }

    // Process remaining 8-element vector chunk
    for (; i + 8 <= end; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vres = _mm256_mul_ps(va, vb);
        _mm256_stream_ps(out + i, vres);
    }

    // Scalar tail loop
    for (; i < end; ++i)
    {
        out[i] = a[i] * b[i];
    }

    // Store fence to flush write-combining buffers to memory
    _mm_sfence();
}

inline void runMulF32_AVX2_Stream(const KernelContext &ctx)
{
    const float *a = static_cast<const float *>(ctx.inputs[0]);
    const float *b = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &out_shape = ctx.outViews[0].getShape();
    uint64_t total_elements = countElements(out_shape);
    if (total_elements == 0)
        return;

    uint32_t num_threads = ThreadPool::get().get_num_threads();
    if (num_threads == 0)
        num_threads = std::max(1U, std::thread::hardware_concurrency());

    uint64_t chunk_size = (total_elements + num_threads - 1) / num_threads;
    chunk_size = (chunk_size + 15) & ~15ULL;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) __attribute__((target("avx2,fma"))) {
        uint64_t start = t * chunk_size;
        if (start >= total_elements)
            return;
        uint64_t end = std::min(start + chunk_size, total_elements);
        mulF32Avx2StreamRange(a, b, out, start, end);
    });
}



REGISTER_KERNEL("Mul_F32_AVX2_Stream", 2, 2, matchMulF32_AVX2_Stream, runMulF32_AVX2_Stream,
                refFactoryMulF32, {0, 1}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 16, 640}, {1, 16, 640}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#pragma GCC pop_options
#endif
