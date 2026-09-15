#pragma once
#include "core/common/constants.hpp"
#include "core/common/thread_pool.hpp"
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

inline bool matchKreaSwiGLU_MLP_StreamingStorage_F8_E4M3(const std::vector<TensorNode> &inputs,
                                                         const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 2 || inputs[2].getShape().size() != 2 ||
        inputs[3].getShape().size() != 2 || output.getShape().size() != 3)
        return false;

    const auto &sX = inputs[0].getShape();     // [1, S, K]
    const auto &sWgate = inputs[1].getShape(); // [I, K]
    const auto &sWup = inputs[2].getShape();   // [I, K]
    const auto &sWdown = inputs[3].getShape(); // [K, I]
    const auto &sO = output.getShape();        // [1, S, K]

    if (sX[0] != 1 || sO[0] != 1)
        return false;
    if (sX[1] != sO[1])
        return false;
    if (sX[2] != sWgate[1] || sX[2] != sWup[1] || sX[2] != sWdown[0] || sX[2] != sO[2])
        return false;
    if (sWgate[0] != sWup[0] || sWgate[0] != sWdown[1])
        return false;

    if (!isContiguous(output))
        return false;

    return true;
}

static inline float krea_fp8e4m3fn_to_fp32(uint8_t input)
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

static inline const float *get_krea_fp8_lut()
{
    static const auto lut = []() {
        std::array<float, 256> table{};
        for (int i = 0; i < 256; ++i)
            table[i] = krea_fp8e4m3fn_to_fp32(static_cast<uint8_t>(i));
        return table;
    }();
    return lut.data();
}

static inline bool kreaMlp_readFromFileAtOffset(int fd, uint64_t offset, void *buf, uint64_t bytes)
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

inline void runKreaSwiGLU_MLP_StreamingStorage_F8_E4M3(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t S = ctx.inViews[0].getShape()[1];
    const uint32_t K = ctx.inViews[0].getShape()[2];
    const uint32_t I = ctx.inViews[1].getShape()[0];

    const int fd_gate = ctx.fd[1];
    const int fd_up = ctx.fd[2];
    const int fd_down = ctx.fd[3];

    if (fd_gate < 0 || fd_up < 0 || fd_down < 0)
    {
        Error::throw_err("KreaSwiGLU_MLP_StreamingStorage_F8_E4M3: expected STORAGE inputs for weights.");
    }

    const uint64_t off_gate = ctx.inViews[1].offset;
    const uint64_t off_up = ctx.inViews[2].offset;
    const uint64_t off_down = ctx.inViews[3].offset;

    const float *lut = get_krea_fp8_lut();

    std::memset(Out, 0, static_cast<uint64_t>(S) * K * sizeof(float));

    uint32_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0)
        hw_threads = 1;
    uint32_t num_threads = std::min(hw_threads, std::max(1u, S));

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t s_per_thread = (S + num_threads - 1) / num_threads;
        uint32_t s_start = t * s_per_thread;
        uint32_t s_end = std::min(s_start + s_per_thread, S);
        if (s_start >= s_end)
            return;

        uint32_t cur_S = s_end - s_start;
        std::vector<float> swiglu(static_cast<uint64_t>(cur_S) * I);

        constexpr uint32_t I_CHUNK = 256;
        std::vector<uint8_t> w_gate_chunk(static_cast<uint64_t>(I_CHUNK) * K);
        std::vector<uint8_t> w_up_chunk(static_cast<uint64_t>(I_CHUNK) * K);

        for (uint32_t i_start = 0; i_start < I; i_start += I_CHUNK)
        {
            uint32_t i_end = std::min(i_start + I_CHUNK, I);
            uint32_t chunk_rows = i_end - i_start;
            uint64_t chunk_bytes = static_cast<uint64_t>(chunk_rows) * K;

            kreaMlp_readFromFileAtOffset(fd_gate, off_gate + static_cast<uint64_t>(i_start) * K, w_gate_chunk.data(),
                                         chunk_bytes);
            kreaMlp_readFromFileAtOffset(fd_up, off_up + static_cast<uint64_t>(i_start) * K, w_up_chunk.data(),
                                         chunk_bytes);

            const uint32_t K4 = K & ~3u;

            for (uint32_t s_local = 0; s_local < cur_S; ++s_local)
            {
                uint32_t s = s_start + s_local;
                const float *x_row = X + static_cast<uint64_t>(s) * K;
                float *swiglu_row = swiglu.data() + static_cast<uint64_t>(s_local) * I;

                for (uint32_t i_off = 0; i_off < chunk_rows; ++i_off)
                {
                    uint32_t i_idx = i_start + i_off;
                    const uint8_t *wg = w_gate_chunk.data() + static_cast<uint64_t>(i_off) * K;
                    const uint8_t *wu = w_up_chunk.data() + static_cast<uint64_t>(i_off) * K;

#if defined(TG_HAS_NEON)
                    float32x4_t acc_g = vdupq_n_f32(0.0f);
                    float32x4_t acc_u = vdupq_n_f32(0.0f);

                    for (uint32_t k = 0; k < K4; k += 4)
                    {
                        float32x4_t xv = vld1q_f32(x_row + k);
                        float32x4_t wgv = {lut[wg[k]], lut[wg[k + 1]], lut[wg[k + 2]], lut[wg[k + 3]]};
                        float32x4_t wuv = {lut[wu[k]], lut[wu[k + 1]], lut[wu[k + 2]], lut[wu[k + 3]]};
                        acc_g = vfmaq_f32(acc_g, xv, wgv);
                        acc_u = vfmaq_f32(acc_u, xv, wuv);
                    }

                    float gate_val = vaddvq_f32(acc_g);
                    float up_val = vaddvq_f32(acc_u);
#else
                    float gate_val = 0.0f, up_val = 0.0f;
                    for (uint32_t k = 0; k < K4; k += 4)
                    {
                        float x0 = x_row[k], x1 = x_row[k + 1], x2 = x_row[k + 2], x3 = x_row[k + 3];
                        gate_val += x0 * lut[wg[k]] + x1 * lut[wg[k + 1]] + x2 * lut[wg[k + 2]] + x3 * lut[wg[k + 3]];
                        up_val += x0 * lut[wu[k]] + x1 * lut[wu[k + 1]] + x2 * lut[wu[k + 2]] + x3 * lut[wu[k + 3]];
                    }
#endif
                    for (uint32_t k = K4; k < K; ++k)
                    {
                        gate_val += x_row[k] * lut[wg[k]];
                        up_val += x_row[k] * lut[wu[k]];
                    }

                    float sig = (gate_val >= 0.0f) ? (1.0f / (1.0f + std::exp(-gate_val)))
                                                   : (std::exp(gate_val) / (1.0f + std::exp(gate_val)));
                    swiglu_row[i_idx] = (gate_val * sig) * up_val;
                }
            }
        }

        constexpr uint32_t K_CHUNK = 256;
        std::vector<uint8_t> w_down_chunk(static_cast<uint64_t>(K_CHUNK) * I);

        for (uint32_t k_start = 0; k_start < K; k_start += K_CHUNK)
        {
            uint32_t k_end = std::min(k_start + K_CHUNK, K);
            uint32_t chunk_rows = k_end - k_start;
            uint64_t chunk_bytes = static_cast<uint64_t>(chunk_rows) * I;

            kreaMlp_readFromFileAtOffset(fd_down, off_down + static_cast<uint64_t>(k_start) * I, w_down_chunk.data(),
                                         chunk_bytes);

            const uint32_t I4 = I & ~3u;

            for (uint32_t s_local = 0; s_local < cur_S; ++s_local)
            {
                uint32_t s = s_start + s_local;
                const float *swiglu_row = swiglu.data() + static_cast<uint64_t>(s_local) * I;
                float *out_row = Out + static_cast<uint64_t>(s) * K;

                for (uint32_t k_off = 0; k_off < chunk_rows; ++k_off)
                {
                    uint32_t k_idx = k_start + k_off;
                    const uint8_t *wd = w_down_chunk.data() + static_cast<uint64_t>(k_off) * I;

#if defined(TG_HAS_NEON)
                    float32x4_t acc_d = vdupq_n_f32(0.0f);
                    for (uint32_t i = 0; i < I4; i += 4)
                    {
                        float32x4_t sv = vld1q_f32(swiglu_row + i);
                        float32x4_t wdv = {lut[wd[i]], lut[wd[i + 1]], lut[wd[i + 2]], lut[wd[i + 3]]};
                        acc_d = vfmaq_f32(acc_d, sv, wdv);
                    }
                    float sum_d = vaddvq_f32(acc_d);
#else
                    float sum_d = 0.0f;
                    for (uint32_t i = 0; i < I4; i += 4)
                    {
                        sum_d += swiglu_row[i] * lut[wd[i]] + swiglu_row[i + 1] * lut[wd[i + 1]] +
                                 swiglu_row[i + 2] * lut[wd[i + 2]] + swiglu_row[i + 3] * lut[wd[i + 3]];
                    }
#endif
                    for (uint32_t i = I4; i < I; ++i)
                    {
                        sum_d += swiglu_row[i] * lut[wd[i]];
                    }

                    out_row[k_idx] = sum_d;
                }
            }
        }
    });
}

inline LogicalId refFactoryKreaSwiGLU_MLP_StreamingStorage_F8_E4M3(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x = inputs[0];
    auto sX = graph.getNode(x).getShape();
    uint32_t S = sX[1];
    uint32_t K = sX[2];
    auto sWgate = graph.getNode(inputs[1]).getShape();
    uint32_t I = sWgate[0];

    int32_t perm[] = {1, 0};
    LogicalId perm_node = graph.constant({2}, perm, DType::INT32);

    LogicalId w_gate_copy = graph._copyto(inputs[1]);
    LogicalId w_gate_cast = graph.cast(w_gate_copy, DType::FLOAT32);
    LogicalId w_gate_t = graph.contiguous(graph.permute(w_gate_cast, perm_node));
    int32_t s3_gate[] = {1, (int32_t)K, (int32_t)I};
    LogicalId gate_mlp = graph.dot(x, graph.reshape(w_gate_t, graph.constant({3}, s3_gate, DType::INT32)));

    LogicalId w_up_copy = graph._copyto(inputs[2]);
    LogicalId w_up_cast = graph.cast(w_up_copy, DType::FLOAT32);
    LogicalId w_up_t = graph.contiguous(graph.permute(w_up_cast, perm_node));
    LogicalId up_mlp = graph.dot(x, graph.reshape(w_up_t, graph.constant({3}, s3_gate, DType::INT32)));

    LogicalId neg_one = graph.fill(-1.0f, {1, S, I});
    LogicalId neg_x = graph.mul(gate_mlp, neg_one);
    LogicalId exp_neg_x = graph.pow(graph.fill(TGConstants::E, {1, S, I}), neg_x);
    LogicalId one = graph.fill(1.0f, {1, S, I});
    LogicalId sig = graph.div(one, graph.add(one, exp_neg_x));
    LogicalId silu_gate = graph.mul(gate_mlp, sig);

    LogicalId swiglu = graph.mul(silu_gate, up_mlp);

    LogicalId w_down_copy = graph._copyto(inputs[3]);
    LogicalId w_down_cast = graph.cast(w_down_copy, DType::FLOAT32);
    LogicalId w_down_t = graph.contiguous(graph.permute(w_down_cast, perm_node));
    int32_t s3_down[] = {1, (int32_t)I, (int32_t)K};
    LogicalId mlp_out = graph.dot(swiglu, graph.reshape(w_down_t, graph.constant({3}, s3_down, DType::INT32)));

    return mlp_out;
}

REGISTER_KERNEL("Fused_Krea_SwiGLU_MLP_StreamingStorage_F8_E4M3", 4, 4, matchKreaSwiGLU_MLP_StreamingStorage_F8_E4M3,
                runKreaSwiGLU_MLP_StreamingStorage_F8_E4M3, refFactoryKreaSwiGLU_MLP_StreamingStorage_F8_E4M3, {},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::F8_E4M3, DType::F8_E4M3, DType::F8_E4M3},
                {{1, 8, 2048}, {4096, 2048}, {4096, 2048}, {2048, 4096}}, {true, true, true, true},
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(0, HandleType::STORAGE)},
                 {MemSpace(0, HandleType::STORAGE)},
                 {MemSpace(0, HandleType::STORAGE)}});