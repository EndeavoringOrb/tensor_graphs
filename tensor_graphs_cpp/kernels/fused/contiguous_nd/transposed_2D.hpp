#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/contiguous_nd/ref.hpp"
#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#include <immintrin.h>
#define TG_HAS_AVX2 1
#endif

/**
 * Highly optimized multi-threaded cache-blocked 2D Transposition / Contiguous kernel.
 * Replaces the slow, un-vectorized RecursiveContiguous_ND fallback for [M, N] transposed strides [1, M].
 */

inline bool matchContiguousTransposed2D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 2 || output.getShape().size() != 2)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (output.dtype != DType::FLOAT32)
        return false;
    if (!isContiguous(output))
        return false;

    const auto &shape = inputs[0].getShape();
    const auto &strides = inputs[0].strides;
    if (strides.size() != 2)
        return false;

    uint64_t m_dim = shape[0];
    uint64_t n_dim = shape[1];

    if (m_dim <= 1 || n_dim <= 1)
        return false;

    // Verify the input has transposed strides [1, M]
    if (strides[0] != 1 || strides[1] != m_dim)
        return false;

    return true;
}

#if defined(TG_HAS_AVX2)
__attribute__((always_inline, target("avx2,fma")))
inline void transpose8x8Avx2(const float *src, uint64_t src_stride, float *dst, uint64_t dst_stride)
{
    __m256 row0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 row1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 row2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 row3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 row4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 row5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 row6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 row7 = _mm256_loadu_ps(src + 7 * src_stride);

    __m256 t0 = _mm256_unpacklo_ps(row0, row1);
    __m256 t1 = _mm256_unpackhi_ps(row0, row1);
    __m256 t2 = _mm256_unpacklo_ps(row2, row3);
    __m256 t3 = _mm256_unpackhi_ps(row2, row3);
    __m256 t4 = _mm256_unpacklo_ps(row4, row5);
    __m256 t5 = _mm256_unpackhi_ps(row4, row5);
    __m256 t6 = _mm256_unpacklo_ps(row6, row7);
    __m256 t7 = _mm256_unpackhi_ps(row6, row7);

    __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));

    __m256 col0 = _mm256_permute2f128_ps(u0, u4, 0x20);
    __m256 col1 = _mm256_permute2f128_ps(u1, u5, 0x20);
    __m256 col2 = _mm256_permute2f128_ps(u2, u6, 0x20);
    __m256 col3 = _mm256_permute2f128_ps(u3, u7, 0x20);
    __m256 col4 = _mm256_permute2f128_ps(u0, u4, 0x31);
    __m256 col5 = _mm256_permute2f128_ps(u1, u5, 0x31);
    __m256 col6 = _mm256_permute2f128_ps(u2, u6, 0x31);
    __m256 col7 = _mm256_permute2f128_ps(u3, u7, 0x31);

    _mm256_storeu_ps(dst + 0 * dst_stride, col0);
    _mm256_storeu_ps(dst + 1 * dst_stride, col1);
    _mm256_storeu_ps(dst + 2 * dst_stride, col2);
    _mm256_storeu_ps(dst + 3 * dst_stride, col3);
    _mm256_storeu_ps(dst + 4 * dst_stride, col4);
    _mm256_storeu_ps(dst + 5 * dst_stride, col5);
    _mm256_storeu_ps(dst + 6 * dst_stride, col6);
    _mm256_storeu_ps(dst + 7 * dst_stride, col7);
}
#endif

#if defined(TG_HAS_AVX2)
__attribute__((target("avx2,fma")))
#endif
inline void transposeTile(const float *in, float *out, uint64_t m_dim, uint64_t n_dim,
                          uint64_t tm, uint64_t tn, uint64_t m_end, uint64_t n_end)
{
#if defined(TG_HAS_AVX2)
    uint64_t m_vec_end = tm + ((m_end - tm) / 8) * 8;
    uint64_t n_vec_end = tn + ((n_end - tn) / 8) * 8;

    for (uint64_t m = tm; m < m_vec_end; m += 8)
    {
        for (uint64_t n = tn; n < n_vec_end; n += 8)
        {
            transpose8x8Avx2(in + n * m_dim + m, m_dim, out + m * n_dim + n, n_dim);
        }
        for (uint64_t n = n_vec_end; n < n_end; ++n)
        {
            for (uint64_t k = 0; k < 8; ++k)
            {
                out[(m + k) * n_dim + n] = in[n * m_dim + (m + k)];
            }
        }
    }

    for (uint64_t m = m_vec_end; m < m_end; ++m)
    {
        for (uint64_t n = tn; n < n_end; ++n)
        {
            out[m * n_dim + n] = in[n * m_dim + m];
        }
    }
#else
    for (uint64_t m = tm; m < m_end; ++m)
    {
        for (uint64_t n = tn; n < n_end; ++n)
        {
            out[m * n_dim + n] = in[n * m_dim + m];
        }
    }
#endif
}

inline void runContiguousTransposed2D(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &view = ctx.inViews[0];
    const auto &shape = view.getShape();

    uint64_t m_dim = shape[0];
    uint64_t n_dim = shape[1];

    constexpr uint64_t tile_m = 64;
    constexpr uint64_t tile_n = 64;

    uint64_t num_m_tiles = (m_dim + tile_m - 1) / tile_m;
    uint64_t num_n_tiles = (n_dim + tile_n - 1) / tile_n;
    uint64_t total_tiles = num_m_tiles * num_n_tiles;

    uint32_t num_threads = ThreadPool::get().get_num_threads();
    if (num_threads == 0)
        num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    uint32_t num_tasks = std::min<uint32_t>(num_threads, static_cast<uint32_t>(total_tiles));

    ThreadPool::get().parallel_for(num_tasks, [=](uint32_t t) __attribute__((target("avx2,fma"))) {
        uint64_t tiles_per_task = (total_tiles + num_tasks - 1) / num_tasks;
        uint64_t t_start = t * tiles_per_task;
        uint64_t t_end = std::min(t_start + tiles_per_task, total_tiles);

        for (uint64_t tile_idx = t_start; tile_idx < t_end; ++tile_idx)
        {
            uint64_t m_tile_idx = tile_idx / num_n_tiles;
            uint64_t n_tile_idx = tile_idx % num_n_tiles;
            uint64_t tm = m_tile_idx * tile_m;
            uint64_t tn = n_tile_idx * tile_n;
            uint64_t m_end = std::min(tm + tile_m, m_dim);
            uint64_t n_end = std::min(tn + tile_n, n_dim);

            transposeTile(in, out, m_dim, n_dim, tm, tn, m_end, n_end);
        }
    });
}



REGISTER_KERNEL("Contiguous_Transposed_2D", 1, 1, matchContiguousTransposed2D, runContiguousTransposed2D,
                refFactoryContiguous, {}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32}, {{640, 2048}}, {false}, {{MemSpace(1, HandleType::CPP)}});

#if defined(__x86_64__) || defined(_M_X64)
#pragma GCC pop_options
#endif
