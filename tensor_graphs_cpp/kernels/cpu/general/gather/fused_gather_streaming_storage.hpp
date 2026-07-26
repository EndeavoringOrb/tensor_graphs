// File:
// tensor_graphs_cpp/kernels/cpu/general/gather/fused_gather_streaming_storage.hpp
//
// Fused Streaming Storage Gather (BF16 -> FP32 Lookup)
// ----------------------------------------------------
//
// Problem solved:
// The token embedding lookup (embed_tokens.weight) currently performs a massive
// COPY_TO of the entire 1 GB weight matrix [248320, 2048] from STORAGE to CPU,
// followed by a cast to FP32, simply to gather a few sequence tokens (e.g., 8).
//
// What this kernel does:
// Bypasses the full matrix load entirely. It reads only the requested rows
// directly from disk via positional read, casts the loaded BF16 row data to
// FP32 using NEON SIMD bit manipulation, and writes directly to the
// destination.

#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

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
inline bool matchGatherStreamingStorage(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // data: [vocab_size, hidden_size]
    // indices: any shape (e.g. [1, seq_len])
    if (inputs[0].getShape().empty())
        return false;
    if (output.getShape().empty())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

// ---------------------------------------------------------------------------
// Positional disk read helper
// ---------------------------------------------------------------------------
static inline bool gather_readFromFileAtOffset(int fd, uint64_t offset, void *buf, uint64_t bytes)
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
        suint64_t n = pread(fd, p, remaining, cur);
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
// Run function
// ---------------------------------------------------------------------------
inline void runGatherStreamingStorage(const KernelContext &ctx)
{
    // inputs[0] is MemSpace(0, HandleType::STORAGE) (nullptr). We use ctx.fd[0]
    // to read.
    const int32_t *indices = static_cast<const int32_t *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &dataShape = ctx.inViews[0].getShape(); // [vocab_size, hidden_size]
    const auto &idxShape = ctx.inViews[1].getShape();

    uint32_t vocabSize = dataShape[0];
    uint64_t rowSize = 1;
    for (uint64_t i = 1; i < dataShape.size(); ++i)
        rowSize *= dataShape[i];

    uint64_t numIndices = countElements(idxShape);
    int fd = ctx.fd[0];
    if (fd < 0)
    {
        Error::throw_err("Gather_StreamingStorage_NEON: expected STORAGE input for "
                         "W (fd[0] >= 0).");
    }

    uint64_t fileOffset = ctx.inViews[0].offset;
    uint64_t rowSizeBytes = rowSize * sizeof(uint16_t);

    // Thread-local scratchpad row buffer
    std::vector<uint16_t> row_buf(rowSize);

    for (uint64_t i = 0; i < numIndices; ++i)
    {
        int32_t idx = indices[i];
        float *dst_row = out + (i * rowSize);

        if (idx < 0 || (uint32_t)idx >= vocabSize)
        {
            std::memset(dst_row, 0, rowSize * sizeof(float));
            continue;
        }

        uint64_t offset = fileOffset + static_cast<uint64_t>(idx) * rowSizeBytes;

        if (!gather_readFromFileAtOffset(fd, offset, row_buf.data(), rowSizeBytes))
        {
            std::memset(dst_row, 0, rowSize * sizeof(float));
            continue;
        }

        uint64_t j = 0;
#if defined(TG_HAS_NEON)
        // Convert BF16 -> F32 using NEON
        for (; j + 8 <= rowSize; j += 8)
        {
            uint16x8_t bf16_val = vld1q_u16(row_buf.data() + j);
            float32x4_t f32_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf16_val), 16));
            float32x4_t f32_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf16_val), 16));
            vst1q_f32(dst_row + j, f32_lo);
            vst1q_f32(dst_row + j + 4, f32_hi);
        }
#endif
        // Scalar fallback / cleanup
        for (; j < rowSize; ++j)
        {
            uint32_t bits = static_cast<uint32_t>(row_buf[j]) << 16;
            std::memcpy(&dst_row[j], &bits, sizeof(float));
        }
    }
}

// ---------------------------------------------------------------------------
// Reference Factory
// ---------------------------------------------------------------------------
inline LogicalId refFactoryGatherStreamingStorage(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // inputs[0]: raw_weight_storage (STORAGE, BF16)
    // inputs[1]: indices (CPU, INT32)

    // 1. COPY_TO: STORAGE BF16 -> CPU BF16
    LogicalId w_cpu = graph._copyto(inputs[0]);

    // 2. CAST: CPU BF16 -> CPU FLOAT32
    LogicalId w_cast = graph.cast(w_cpu, DType::FLOAT32);

    // 3. GATHER: CPU FLOAT32
    return graph.gather(w_cast, inputs[1]);
}

REGISTER_KERNEL("Gather_StreamingStorage_NEON", 2, 2, matchGatherStreamingStorage, runGatherStreamingStorage,
                refFactoryGatherStreamingStorage, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, // output backend
                {DType::BF16, DType::INT32},  // input types: raw weight (BF16), indices (INT32)
                {{248320, 2048}, {1, 8}},     // dummy shapes
                {true, true},                 // requires contiguous inputs
                {{MemSpace(0, HandleType::STORAGE)}, {MemSpace(1, HandleType::CPP)}}); // input placement