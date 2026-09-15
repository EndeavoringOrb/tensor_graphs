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

inline bool matchFusedProjBiasStreamingStorage_F8_E4M3(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 2 || inputs[2].getShape().size() != 3 ||
        output.getShape().size() != 3)
        return false;

    const auto &sX = inputs[0].getShape(); // [1, S, K]
    const auto &sW = inputs[1].getShape(); // [N, K]
    const auto &sB = inputs[2].getShape(); // [1, S, N]
    const auto &sO = output.getShape();    // [1, S, N]

    if (sX[0] != 1 || sB[0] != 1 || sO[0] != 1)
        return false;
    if (sX[2] != sW[1])
        return false;
    if (sO[1] != sX[1] || sB[1] != sX[1])
        return false;
    if (sO[2] != sW[0] || sB[2] != sW[0])
        return false;

    if (!isContiguous(output))
        return false;

    return true;
}

static inline float projBias_fp8e4m3fn_to_fp32(uint8_t input)
{
    if (input == 0x7F || input == 0xFF)
        return std::numeric_limits<float>::quiet_NaN();
    float sign = (input & 0x80) ? -1.0f : 1.0f;
    uint32_t exp = (input & 0x78) >> 3;
    uint32_t mant = input & 0x07;
    if (exp == 0)
    {
        if (mant == 0)
            return sign * 0.0f;
        return sign * std::ldexp(static_cast<float>(mant), -9);
    }
    return sign * std::ldexp(1.0f + static_cast<float>(mant) * 0.125f, static_cast<int>(exp) - 7);
}

static inline const float *get_proj_bias_fp8_lut()
{
    static const auto lut = []() {
        std::array<float, 256> table{};
        for (int i = 0; i < 256; ++i)
            table[i] = projBias_fp8e4m3fn_to_fp32(static_cast<uint8_t>(i));
        return table;
    }();
    return lut.data();
}

static inline bool projBias_readFromFileAtOffset(int fd, uint64_t offset, void *buf, uint64_t bytes)
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

inline void runFusedProjBiasStreamingStorage_F8_E4M3(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const float *B = static_cast<const float *>(ctx.inputs[2]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const auto &viewX = ctx.inViews[0];
    const auto &viewW = ctx.inViews[1];

    const uint32_t S = viewX.getShape()[1];
    const uint32_t K = viewX.getShape()[2];
    const uint32_t N = viewW.getShape()[0];

    const int fd = ctx.fd[1];
    if (fd < 0)
    {
        Error::throw_err("Fused_Proj_Bias_StreamingStorage_F8_E4M3: expected STORAGE input for W.");
    }

    const uint64_t fileOffset = viewW.offset;
    const float *lut = get_proj_bias_fp8_lut();

    uint32_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0)
        hw_threads = 1;
    uint32_t num_threads = std::min(hw_threads, std::max(1u, N / 4u));
    if (num_threads == 0)
        num_threads = 1;

    uint32_t per_thread = ((N + num_threads - 1) / num_threads + 3u) & ~3u;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t n_start = t * per_thread;
        uint32_t n_end = std::min(n_start + per_thread, N);
        if (n_start >= n_end)
            return;

        constexpr uint32_t STREAM_CHUNK_BYTES = 256ull * 1024;
        const uint32_t chunk_rows =
            std::max(1u, static_cast<uint32_t>(STREAM_CHUNK_BYTES / (static_cast<uint64_t>(K) * sizeof(uint8_t))));

        std::vector<uint8_t> w_buf(static_cast<uint64_t>(chunk_rows) * K);

        for (uint32_t chunk_start = n_start; chunk_start < n_end; chunk_start += chunk_rows)
        {
            uint32_t chunk_end = std::min(chunk_start + chunk_rows, n_end);
            uint32_t this_rows = chunk_end - chunk_start;
            uint64_t chunk_off = fileOffset + static_cast<uint64_t>(chunk_start) * K * sizeof(uint8_t);
            uint64_t chunk_bytes = static_cast<uint64_t>(this_rows) * K * sizeof(uint8_t);

            if (!projBias_readFromFileAtOffset(fd, chunk_off, w_buf.data(), chunk_bytes))
            {
                std::memset(w_buf.data(), 0, static_cast<uint64_t>(chunk_bytes));
            }

            const uint32_t K4 = K & ~3u;
            const uint32_t N4 = this_rows & ~3u;

            for (uint32_t n_off = 0; n_off < N4; n_off += 4)
            {
                uint32_t n = chunk_start + n_off;
                const uint8_t *w0 = w_buf.data() + (n_off + 0) * K;
                const uint8_t *w1 = w_buf.data() + (n_off + 1) * K;
                const uint8_t *w2 = w_buf.data() + (n_off + 2) * K;
                const uint8_t *w3 = w_buf.data() + (n_off + 3) * K;

                for (uint32_t s = 0; s < S; ++s)
                {
                    const float *x_row = X + static_cast<uint64_t>(s) * K;
                    const float *b_row = B + static_cast<uint64_t>(s) * N + n;
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

                    float sum0 = vaddvq_f32(acc0) + b_row[0];
                    float sum1 = vaddvq_f32(acc1) + b_row[1];
                    float sum2 = vaddvq_f32(acc2) + b_row[2];
                    float sum3 = vaddvq_f32(acc3) + b_row[3];
#else
                    float sum0 = b_row[0], sum1 = b_row[1], sum2 = b_row[2], sum3 = b_row[3];
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

            for (uint32_t n_off = N4; n_off < this_rows; ++n_off)
            {
                uint32_t n = chunk_start + n_off;
                const uint8_t *w = w_buf.data() + n_off * K;
                for (uint32_t s = 0; s < S; ++s)
                {
                    const float *x_row = X + static_cast<uint64_t>(s) * K;
                    float sum = B[static_cast<uint64_t>(s) * N + n];
                    for (uint32_t k = 0; k < K; ++k)
                    {
                        sum += x_row[k] * lut[w[k]];
                    }
                    Out[static_cast<uint64_t>(s) * N + n] = sum;
                }
            }
        }
    });
}

inline LogicalId refFactoryFusedProjBiasStreamingStorage_F8_E4M3(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId w_copy = graph._copyto(inputs[1]);
    LogicalId w_cast = graph.cast(w_copy, DType::FLOAT32);
    int32_t perm[] = {1, 0};
    LogicalId w_t = graph.permute(w_cast, graph.constant({2}, perm, DType::INT32));
    LogicalId w_t_contig = graph.contiguous(w_t);
    auto w_shape = graph.getNode(inputs[1]).getShape();
    int32_t s3[] = {1, static_cast<int32_t>(w_shape[1]), static_cast<int32_t>(w_shape[0])};
    LogicalId w_3d = graph.reshape(w_t_contig, graph.constant({3}, s3, DType::INT32));
    LogicalId dot = graph.dot(inputs[0], w_3d);
    return graph.add(dot, inputs[2]);
}

REGISTER_KERNEL("Fused_Proj_Bias_StreamingStorage_F8_E4M3", 3, 3, matchFusedProjBiasStreamingStorage_F8_E4M3,
                runFusedProjBiasStreamingStorage_F8_E4M3, refFactoryFusedProjBiasStreamingStorage_F8_E4M3, {2},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::F8_E4M3, DType::FLOAT32}, {{1, 8, 2048}, {4096, 2048}, {1, 8, 4096}},
                {true, true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(0, HandleType::STORAGE)}, {MemSpace(1, HandleType::CPP)}});