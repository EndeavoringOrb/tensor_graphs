// Fused 2D Projection GEMM with Streaming Storage Loading for FP8 (F8_E4M3) Weights
// ----------------------------------------------------------------------------------
// Fuses COPY_TO (STORAGE -> CPU F8_E4M3) + CAST (F8_E4M3 -> FP32) + PERMUTE + CONTIGUOUS + DOT
// directly into a streaming matmul kernel that reads row chunks from disk and converts
// FP8 weights in-register/via L1 LUT during the GEMM inner loop.

#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

#ifdef TG_OS_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// Match function
// ---------------------------------------------------------------------------
inline bool matchFusedProjStreamingStorage_F8_E4M3(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // X: [1, S, K], W: [N, K], Out: [1, S, N]
    if (inputs[0].getShape().size() != 3)
        return false;
    if (inputs[1].getShape().size() != 2)
        return false;
    if (output.getShape().size() != 3)
        return false;

    const auto &sX = inputs[0].getShape(); // [1, S, K]
    const auto &sW = inputs[1].getShape(); // [N, K]
    const auto &sO = output.getShape();    // [1, S, N]

    if (sX[0] != 1)
        return false;
    if (sX[2] != sW[1])
        return false; // K matches
    if (sO[0] != 1)
        return false;
    if (sO[1] != sX[1])
        return false; // S matches
    if (sO[2] != sW[0])
        return false; // N matches

    if (!isContiguous(output))
        return false;

    return true;
}

// ---------------------------------------------------------------------------
// Fast Portable FP8 E4M3FN -> FP32 Conversion
// ---------------------------------------------------------------------------
static inline float fp8e4m3fn_to_fp32_val(uint8_t input)
{
    if (input == 0x7F || input == 0xFF)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    float sign = (input & 0x80) ? -1.0f : 1.0f;
    uint32_t exp = (input & 0x78) >> 3;
    uint32_t mant = input & 0x07;
    if (exp == 0)
    {
        if (mant == 0)
        {
            return sign * 0.0f;
        }
        else
        {
            return sign * std::ldexp(static_cast<float>(mant), -9);
        }
    }
    else
    {
        return sign * std::ldexp(1.0f + static_cast<float>(mant) * 0.125f, static_cast<int>(exp) - 7);
    }
}

// 1 KB static L1 lookup table for O(1) FP8 conversion
static inline const float *get_fp8_e4m3_lut()
{
    static const auto lut = []() {
        std::array<float, 256> table{};
        for (int i = 0; i < 256; ++i)
        {
            table[i] = fp8e4m3fn_to_fp32_val(static_cast<uint8_t>(i));
        }
        return table;
    }();
    return lut.data();
}

// ---------------------------------------------------------------------------
// Thread-safe positional disk reader
// ---------------------------------------------------------------------------
static inline bool fusedProjF8_readFromFileAtOffset(int fd, uint64_t offset, void *buf, uint64_t bytes)
{
    if (bytes == 0)
        return true;
    uint8_t *p = static_cast<uint8_t *>(buf);
    uint64_t remaining = bytes;
    uint64_t cur = offset;

#ifdef TG_OS_WINDOWS
    HANDLE hFile = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (hFile == INVALID_HANDLE_VALUE)
        return false;
    while (remaining > 0)
    {
        OVERLAPPED ov = {};
        ov.Offset = static_cast<DWORD>(cur & 0xFFFFFFFFull);
        ov.OffsetHigh = static_cast<DWORD>((cur >> 32) & 0xFFFFFFFFull);
        DWORD toRead = static_cast<DWORD>(std::min<uint64_t>(remaining, 0x40000000ull));
        DWORD bytesRead = 0;
        if (!ReadFile(hFile, p, toRead, &bytesRead, &ov))
            return false;
        if (bytesRead == 0)
            return false;
        p += bytesRead;
        cur += bytesRead;
        remaining -= bytesRead;
    }
    return true;
#else
    while (remaining > 0)
    {
        int64_t n = pread(fd, p, remaining, cur);
        if (n <= 0)
            return false;
        p += n;
        cur += n;
        remaining -= static_cast<uint64_t>(n);
    }
    return true;
#endif
}

// ---------------------------------------------------------------------------
// Micro-GEMM tile computation with NEON FMA acceleration
// ---------------------------------------------------------------------------
static inline void fusedProjF8_computeTile(const float *X, const uint8_t *W, float *Out, uint32_t S, uint32_t K,
                                           uint32_t N, uint32_t n_start, uint32_t n_end)
{
    const float *lut = get_fp8_e4m3_lut();
    const uint32_t K4 = K & ~3u;
    const uint32_t this_rows = n_end - n_start;
    const uint32_t N4 = this_rows & ~3u;

    for (uint32_t n_off = 0; n_off < N4; n_off += 4)
    {
        uint32_t n = n_start + n_off;
        const uint8_t *w0 = W + (n_off + 0) * K;
        const uint8_t *w1 = W + (n_off + 1) * K;
        const uint8_t *w2 = W + (n_off + 2) * K;
        const uint8_t *w3 = W + (n_off + 3) * K;

        for (uint32_t s = 0; s < S; ++s)
        {
            const float *x_row = X + static_cast<uint64_t>(s) * K;
            float *out_row = Out + static_cast<uint64_t>(s) * N + n;

#if defined(TG_HAS_NEON)
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);

            for (uint32_t k = 0; k < K4; k += 4)
            {
                float32x4_t xv = vld1q_f32(x_row + k);

                float32x4_t w0v = {lut[w0[k]], lut[w0[k + 1]], lut[w0[k + 2]], lut[w0[k + 3]]};
                float32x4_t w1v = {lut[w1[k]], lut[w1[k + 1]], lut[w1[k + 2]], lut[w1[k + 3]]};
                float32x4_t w2v = {lut[w2[k]], lut[w2[k + 1]], lut[w2[k + 2]], lut[w2[k + 3]]};
                float32x4_t w3v = {lut[w3[k]], lut[w3[k + 1]], lut[w3[k + 2]], lut[w3[k + 3]]};

                acc0 = vfmaq_f32(acc0, xv, w0v);
                acc1 = vfmaq_f32(acc1, xv, w1v);
                acc2 = vfmaq_f32(acc2, xv, w2v);
                acc3 = vfmaq_f32(acc3, xv, w3v);
            }

            float sum0 = vaddvq_f32(acc0);
            float sum1 = vaddvq_f32(acc1);
            float sum2 = vaddvq_f32(acc2);
            float sum3 = vaddvq_f32(acc3);
#else
            float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
            for (uint32_t k = 0; k < K4; k += 4)
            {
                float x0 = x_row[k], x1 = x_row[k + 1], x2 = x_row[k + 2], x3 = x_row[k + 3];
                sum0 += x0 * lut[w0[k]] + x1 * lut[w0[k + 1]] + x2 * lut[w0[k + 2]] + x3 * lut[w0[k + 3]];
                sum1 += x0 * lut[w1[k]] + x1 * lut[w1[k + 1]] + x2 * lut[w1[k + 2]] + x3 * lut[w1[k + 3]];
                sum2 += x0 * lut[w2[k]] + x1 * lut[w2[k + 1]] + x2 * lut[w2[k + 2]] + x3 * lut[w2[k + 3]];
                sum3 += x0 * lut[w3[k]] + x1 * lut[w3[k + 1]] + x2 * lut[w3[k + 2]] + x3 * lut[w3[k + 3]];
            }
#endif

            for (uint32_t k = K4; k < K; ++k)
            {
                float xv = x_row[k];
                sum0 += xv * lut[w0[k]];
                sum1 += xv * lut[w1[k]];
                sum2 += xv * lut[w2[k]];
                sum3 += xv * lut[w3[k]];
            }

            out_row[0] = sum0;
            out_row[1] = sum1;
            out_row[2] = sum2;
            out_row[3] = sum3;
        }
    }

    // N remainder
    for (uint32_t n_off = N4; n_off < this_rows; ++n_off)
    {
        uint32_t n = n_start + n_off;
        const uint8_t *w = W + n_off * K;
        for (uint32_t s = 0; s < S; ++s)
        {
            const float *x_row = X + static_cast<uint64_t>(s) * K;
            float *out_row = Out + static_cast<uint64_t>(s) * N + n;

            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
            {
                sum += x_row[k] * lut[w[k]];
            }
            *out_row = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel execution entry point
// ---------------------------------------------------------------------------
inline void runFusedProjStreamingStorage_F8_E4M3(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const auto &viewX = ctx.inViews[0];
    const auto &viewW = ctx.inViews[1]; // STORAGE F8_E4M3 [N, K]

    const uint32_t S = viewX.getShape()[1];
    const uint32_t K = viewX.getShape()[2];
    const uint32_t N = viewW.getShape()[0];

    const int fd = ctx.fd[1];
    if (fd < 0)
    {
        Error::throw_err("Fused_Proj_StreamingStorage_F8_E4M3: expected STORAGE input for W (fd[1] >= 0).");
    }

    const uint64_t fileOffset = viewW.offset;
    const uint64_t W_total_bytes = static_cast<uint64_t>(N) * K * sizeof(uint8_t);

    constexpr uint64_t SMALL_W_THRESHOLD = 64ull * 1024 * 1024; // 64 MB
    constexpr uint64_t STREAM_CHUNK_BYTES = 256ull * 1024;      // 256 KB

    uint32_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0)
        hw_threads = 1;
    uint32_t num_threads = std::min(hw_threads, std::max(1u, N / 4u));
    if (num_threads == 0)
        num_threads = 1;

    auto worker = [&](uint32_t tid, uint32_t n_start, uint32_t n_end) {
        const uint32_t n_range = n_end - n_start;
        if (n_range == 0)
            return;

        const uint64_t my_W_bytes = static_cast<uint64_t>(n_range) * K * sizeof(uint8_t);
        const uint64_t my_W_offset = fileOffset + static_cast<uint64_t>(n_start) * K * sizeof(uint8_t);

        if (W_total_bytes <= SMALL_W_THRESHOLD)
        {
            std::vector<uint8_t> w_buf(static_cast<uint64_t>(n_range) * K);
            if (!fusedProjF8_readFromFileAtOffset(fd, my_W_offset, w_buf.data(), my_W_bytes))
            {
                std::memset(w_buf.data(), 0, my_W_bytes);
            }
            fusedProjF8_computeTile(X, w_buf.data(), Out, S, K, N, n_start, n_end);
        }
        else
        {
            const uint32_t chunk_rows =
                std::max(1u, static_cast<uint32_t>(STREAM_CHUNK_BYTES / (static_cast<uint64_t>(K) * sizeof(uint8_t))));

            std::vector<uint8_t> w_buf(static_cast<uint64_t>(chunk_rows) * K);

            for (uint32_t chunk_start = n_start; chunk_start < n_end; chunk_start += chunk_rows)
            {
                uint32_t chunk_end = std::min(chunk_start + chunk_rows, n_end);
                uint32_t this_rows = chunk_end - chunk_start;
                uint64_t chunk_off = fileOffset + static_cast<uint64_t>(chunk_start) * K * sizeof(uint8_t);
                uint64_t chunk_bytes = static_cast<uint64_t>(this_rows) * K * sizeof(uint8_t);

                if (!fusedProjF8_readFromFileAtOffset(fd, chunk_off, w_buf.data(), chunk_bytes))
                {
                    std::memset(w_buf.data(), 0, chunk_bytes);
                }

                fusedProjF8_computeTile(X, w_buf.data(), Out, S, K, N, chunk_start, chunk_end);
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    uint32_t per_thread = (N + num_threads - 1) / num_threads;
    per_thread = (per_thread + 3u) & ~3u;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        uint32_t start = t * per_thread;
        uint32_t end = std::min(start + per_thread, N);
        if (start >= end)
            break;
        workers.emplace_back(worker, t, start, end);
    }
    for (auto &w : workers)
        w.join();
}

// ---------------------------------------------------------------------------
// Reference Factory for E-Graph pattern matching
// ---------------------------------------------------------------------------
inline LogicalId refFactoryFusedProjStreamingStorage_F8_E4M3(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // inputs[0]: X [1, S, K] fp32 CPU
    // inputs[1]: W [N, K]    F8_E4M3 STORAGE

    // 1. COPY_TO from STORAGE to CPU
    LogicalId w_copy = graph._copyto(inputs[1]);

    // 2. CAST to FP32
    LogicalId w_cast = graph.cast(w_copy, DType::FLOAT32);

    // 3. PERMUTE: [N, K] -> [K, N]
    int32_t perm[] = {1, 0};
    LogicalId w_t = graph.permute(w_cast, graph.constant({2}, perm, DType::INT32));

    // 4. CONTIGUOUS
    LogicalId w_t_contig = graph.contiguous(w_t);

    // 5. RESHAPE: [K, N] -> [1, K, N]
    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, static_cast<int32_t>(w_shape[1]), static_cast<int32_t>(w_shape[0])};
    LogicalId w_3d = graph.reshape(w_t_contig, graph.constant({3}, s3, DType::INT32));

    // 6. DOT
    return graph.dot(inputs[0], w_3d);
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
REGISTER_KERNEL("Fused_Proj_StreamingStorage_F8_E4M3", 2, 2, matchFusedProjStreamingStorage_F8_E4M3,
                runFusedProjStreamingStorage_F8_E4M3, refFactoryFusedProjStreamingStorage_F8_E4M3, {},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::F8_E4M3},
                {{1, 8, 2048}, {8192, 2048}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(0, HandleType::STORAGE)}});