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

inline bool matchKreaGatedAttnProj_StreamingStorage_F8_E4M3(const std::vector<TensorNode> &inputs,
                                                            const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 3 || inputs[2].getShape().size() != 2 ||
        inputs[3].getShape().size() != 2 || output.getShape().size() != 3)
        return false;

    const auto &sH = inputs[0].getShape();     // [1, S, D]
    const auto &sCtx = inputs[1].getShape();   // [1, S, D]
    const auto &sWgate = inputs[2].getShape(); // [D, D]
    const auto &sWo = inputs[3].getShape();    // [D, D]
    const auto &sO = output.getShape();        // [1, S, D]

    if (sH[0] != 1 || sCtx[0] != 1 || sO[0] != 1)
        return false;
    if (sH[1] != sCtx[1] || sH[1] != sO[1])
        return false;
    if (sH[2] != sCtx[2] || sH[2] != sWgate[1] || sH[2] != sWgate[0] || sH[2] != sWo[1] || sH[2] != sWo[0] ||
        sH[2] != sO[2])
        return false;

    if (!isContiguous(output))
        return false;

    return true;
}

static inline float kreaGated_fp8e4m3fn_to_fp32(uint8_t input)
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

static inline const float *get_krea_gated_fp8_lut()
{
    static const auto lut = []() {
        std::array<float, 256> table{};
        for (int i = 0; i < 256; ++i)
            table[i] = kreaGated_fp8e4m3fn_to_fp32(static_cast<uint8_t>(i));
        return table;
    }();
    return lut.data();
}

static inline bool kreaGated_readFromFileAtOffset(int fd, uint64_t offset, void *buf, uint64_t bytes)
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

inline void runKreaGatedAttnProj_StreamingStorage_F8_E4M3(const KernelContext &ctx)
{
    const float *H = static_cast<const float *>(ctx.inputs[0]);
    const float *Ctx = static_cast<const float *>(ctx.inputs[1]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t S = ctx.inViews[0].getShape()[1];
    const uint32_t D = ctx.inViews[0].getShape()[2];

    const int fd_gate = ctx.fd[2];
    const int fd_o = ctx.fd[3];

    if (fd_gate < 0 || fd_o < 0)
    {
        Error::throw_err("KreaGatedAttnProj_StreamingStorage_F8_E4M3: expected STORAGE inputs for weights.");
    }

    const uint64_t off_gate = ctx.inViews[2].offset;
    const uint64_t off_o = ctx.inViews[3].offset;

    const float *lut = get_krea_gated_fp8_lut();

    std::memset(Out, 0, static_cast<uint64_t>(S) * D * sizeof(float));

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
        std::vector<float> gated_ctx(static_cast<uint64_t>(cur_S) * D);

        constexpr uint32_t CHUNK = 256;
        std::vector<uint8_t> w_chunk(static_cast<uint64_t>(CHUNK) * D);

        // Step 1: gate = H @ W_gate^T, gated_ctx = Ctx * sigmoid(gate)
        for (uint32_t d_start = 0; d_start < D; d_start += CHUNK)
        {
            uint32_t d_end = std::min(d_start + CHUNK, D);
            uint32_t chunk_rows = d_end - d_start;
            uint64_t chunk_bytes = static_cast<uint64_t>(chunk_rows) * D;

            kreaGated_readFromFileAtOffset(fd_gate, off_gate + static_cast<uint64_t>(d_start) * D, w_chunk.data(),
                                           chunk_bytes);

            const uint32_t D4 = D & ~3u;

            for (uint32_t s_local = 0; s_local < cur_S; ++s_local)
            {
                uint32_t s = s_start + s_local;
                const float *h_row = H + static_cast<uint64_t>(s) * D;
                const float *ctx_row = Ctx + static_cast<uint64_t>(s) * D;
                float *gated_ctx_row = gated_ctx.data() + static_cast<uint64_t>(s_local) * D;

                for (uint32_t d_off = 0; d_off < chunk_rows; ++d_off)
                {
                    uint32_t d_idx = d_start + d_off;
                    const uint8_t *wg = w_chunk.data() + static_cast<uint64_t>(d_off) * D;

#if defined(TG_HAS_NEON)
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (uint32_t k = 0; k < D4; k += 4)
                    {
                        float32x4_t hv = vld1q_f32(h_row + k);
                        float32x4_t wgv = {lut[wg[k]], lut[wg[k + 1]], lut[wg[k + 2]], lut[wg[k + 3]]};
                        acc = vfmaq_f32(acc, hv, wgv);
                    }
                    float gate_val = vaddvq_f32(acc);
#else
                    float gate_val = 0.0f;
                    for (uint32_t k = 0; k < D4; k += 4)
                    {
                        gate_val += h_row[k] * lut[wg[k]] + h_row[k + 1] * lut[wg[k + 1]] +
                                    h_row[k + 2] * lut[wg[k + 2]] + h_row[k + 3] * lut[wg[k + 3]];
                    }
#endif
                    for (uint32_t k = D4; k < D; ++k)
                    {
                        gate_val += h_row[k] * lut[wg[k]];
                    }

                    float sig = (gate_val >= 0.0f) ? (1.0f / (1.0f + std::exp(-gate_val)))
                                                   : (std::exp(gate_val) / (1.0f + std::exp(gate_val)));
                    gated_ctx_row[d_idx] = ctx_row[d_idx] * sig;
                }
            }
        }

        // Step 2: Out = gated_ctx @ W_o^T
        for (uint32_t d_start = 0; d_start < D; d_start += CHUNK)
        {
            uint32_t d_end = std::min(d_start + CHUNK, D);
            uint32_t chunk_rows = d_end - d_start;
            uint64_t chunk_bytes = static_cast<uint64_t>(chunk_rows) * D;

            kreaGated_readFromFileAtOffset(fd_o, off_o + static_cast<uint64_t>(d_start) * D, w_chunk.data(),
                                           chunk_bytes);

            const uint32_t D4 = D & ~3u;

            for (uint32_t s_local = 0; s_local < cur_S; ++s_local)
            {
                uint32_t s = s_start + s_local;
                const float *gated_ctx_row = gated_ctx.data() + static_cast<uint64_t>(s_local) * D;
                float *out_row = Out + static_cast<uint64_t>(s) * D;

                for (uint32_t d_off = 0; d_off < chunk_rows; ++d_off)
                {
                    uint32_t d_idx = d_start + d_off;
                    const uint8_t *wo = w_chunk.data() + static_cast<uint64_t>(d_off) * D;

#if defined(TG_HAS_NEON)
                    float32x4_t acc = vdupq_n_f32(0.0f);
                    for (uint32_t k = 0; k < D4; k += 4)
                    {
                        float32x4_t gv = vld1q_f32(gated_ctx_row + k);
                        float32x4_t wov = {lut[wo[k]], lut[wo[k + 1]], lut[wo[k + 2]], lut[wo[k + 3]]};
                        acc = vfmaq_f32(acc, gv, wov);
                    }
                    float sum = vaddvq_f32(acc);
#else
                    float sum = 0.0f;
                    for (uint32_t k = 0; k < D4; k += 4)
                    {
                        sum += gated_ctx_row[k] * lut[wo[k]] + gated_ctx_row[k + 1] * lut[wo[k + 1]] +
                               gated_ctx_row[k + 2] * lut[wo[k + 2]] + gated_ctx_row[k + 3] * lut[wo[k + 3]];
                    }
#endif
                    for (uint32_t k = D4; k < D; ++k)
                    {
                        sum += gated_ctx_row[k] * lut[wo[k]];
                    }

                    out_row[d_idx] = sum;
                }
            }
        }
    });
}

inline LogicalId refFactoryKreaGatedAttnProj_StreamingStorage_F8_E4M3(const std::vector<LogicalId> &inputs,
                                                                      Graph &graph)
{
    LogicalId h = inputs[0];
    LogicalId ctx_flat = inputs[1];
    auto sH = graph.getNode(h).getShape();
    uint32_t S = sH[1];
    uint32_t D = sH[2];

    int32_t perm[] = {1, 0};
    LogicalId perm_node = graph.constant({2}, perm, DType::INT32);
    int32_t s3[] = {1, (int32_t)D, (int32_t)D};
    LogicalId s3_node = graph.constant({3}, s3, DType::INT32);

    LogicalId w_gate_copy = graph._copyto(inputs[2]);
    LogicalId w_gate_cast = graph.cast(w_gate_copy, DType::FLOAT32);
    LogicalId w_gate_t = graph.contiguous(graph.permute(w_gate_cast, perm_node));
    LogicalId gate = graph.dot(h, graph.reshape(w_gate_t, s3_node));

    LogicalId neg_one = graph.fill(-1.0f, {1, S, D});
    LogicalId neg_gate = graph.mul(gate, neg_one);
    LogicalId exp_neg_gate = graph.pow(graph.fill(TGConstants::E, {1, S, D}), neg_gate);
    LogicalId one = graph.fill(1.0f, {1, S, D});
    LogicalId sig = graph.div(one, graph.add(one, exp_neg_gate));

    LogicalId gated_attn = graph.mul(ctx_flat, sig);

    LogicalId w_o_copy = graph._copyto(inputs[3]);
    LogicalId w_o_cast = graph.cast(w_o_copy, DType::FLOAT32);
    LogicalId w_o_t = graph.contiguous(graph.permute(w_o_cast, perm_node));
    LogicalId attn_proj = graph.dot(gated_attn, graph.reshape(w_o_t, s3_node));

    return attn_proj;
}

REGISTER_KERNEL("Fused_Krea_Gated_Attn_Proj_StreamingStorage_F8_E4M3", 4, 4,
                matchKreaGatedAttnProj_StreamingStorage_F8_E4M3, runKreaGatedAttnProj_StreamingStorage_F8_E4M3,
                refFactoryKreaGatedAttnProj_StreamingStorage_F8_E4M3, {}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32, DType::F8_E4M3, DType::F8_E4M3},
                {{1, 8, 6144}, {1, 8, 6144}, {6144, 6144}, {6144, 6144}}, {true, true, true, true},
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(0, HandleType::STORAGE)},
                 {MemSpace(0, HandleType::STORAGE)}});