// File: tensor_graphs_cpp/kernels/cpu/general/gather/fused_gather_streaming_storage_sorted.hpp
//
// Fused Streaming Storage Gather with Sorted & Deduplicated Loading
// -----------------------------------------------------------------
//
// Problem solved:
// Token embedding lookup (embed_tokens.weight) usually loads the entire weight 
// matrix into memory. The basic streaming gather avoids the full load but executes 
// arbitrary random disk reads and may redundantly load identical tokens.
//
// What this kernel does:
// 1. Pairs each target index with its original sequence offset.
// 2. Sorts the indices to create a monotonically increasing disk read pattern.
// 3. Deduplicates identical adjacent tokens, loading the row exactly once.
// 4. Utilizes NEON vector instructions to cast BF16 data to FP32.
// 5. Direct-stores the computed FP32 vectors back to their correct sequence positions.

#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>

#ifdef TG_OS_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// Match function
// ---------------------------------------------------------------------------
inline bool matchGatherStreamingStorageSorted(
    const std::vector<TensorNode> &inputs,
    const TensorNode &output)
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
static inline bool gather_sorted_readFromFileAtOffset(
    int fd, uint64_t offset, void *buf, uint64_t bytes)
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
        DWORD toRead = static_cast<DWORD>(
            std::min<uint64_t>(remaining, 0x40000000ull));
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
inline void runGatherStreamingStorageSorted(const KernelContext &ctx)
{
    // inputs[0] is MemSpace(1, HandleType::STORAGE) (nullptr). We use ctx.fd[0] to read.
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
        Error::throw_err("Gather_StreamingStorage_Sorted_NEON: expected STORAGE input for W (fd[0] >= 0).");
    }

    uint64_t fileOffset = ctx.inViews[0].offset;
    uint64_t rowSizeBytes = rowSize * sizeof(uint16_t);

    // Structure to pair indices with their original sequence offsets
    struct IndexTracker {
        int32_t val;
        uint32_t original_pos;

        // Sort primarily by token ID to group identical tokens and linearize disk offsets
        bool operator<(const IndexTracker& other) const {
            return val < other.val;
        }
    };

    std::vector<IndexTracker> tracked_indices(numIndices);
    for (uint32_t i = 0; i < numIndices; ++i) {
        // Resolve potentially strided input layouts
        uint64_t strided_idx = getStridedIndex(i, idxShape, ctx.inViews[1].strides);
        tracked_indices[i] = {indices[strided_idx], static_cast<uint32_t>(i)};
    }

    // Sort to group duplicate IDs and linearize positions for I/O efficiency
    std::sort(tracked_indices.begin(), tracked_indices.end());

    // Thread-local scratchpad row buffer
    std::vector<uint16_t> row_buf(rowSize);
    int32_t last_read_idx = -1;
    bool last_read_success = false;

    for (uint64_t i = 0; i < numIndices; ++i) {
        int32_t idx = tracked_indices[i].val;
        uint32_t orig_pos = tracked_indices[i].original_pos;
        float *dst_row = out + (orig_pos * rowSize);

        if (idx < 0 || (uint32_t)idx >= vocabSize) {
            std::memset(dst_row, 0, rowSize * sizeof(float));
            continue;
        }

        // Deduplication: only read from disk if the token changed
        if (idx != last_read_idx) {
            uint64_t offset = fileOffset + static_cast<uint64_t>(idx) * rowSizeBytes;
            last_read_success = gather_sorted_readFromFileAtOffset(fd, offset, row_buf.data(), rowSizeBytes);
            last_read_idx = idx;
        }

        if (!last_read_success) {
            std::memset(dst_row, 0, rowSize * sizeof(float));
            continue;
        }

        uint64_t j = 0;
#if defined(TG_HAS_NEON)
        // Convert BF16 -> F32 using NEON
        for (; j + 8 <= rowSize; j += 8) {
            uint16x8_t bf16_val = vld1q_u16(row_buf.data() + j);
            float32x4_t f32_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf16_val), 16));
            float32x4_t f32_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf16_val), 16));
            vst1q_f32(dst_row + j, f32_lo);
            vst1q_f32(dst_row + j + 4, f32_hi);
        }
#endif
        // Scalar fallback / cleanup
        for (; j < rowSize; ++j) {
            uint32_t bits = static_cast<uint32_t>(row_buf[j]) << 16;
            std::memcpy(&dst_row[j], &bits, sizeof(float));
        }
    }
}

// ---------------------------------------------------------------------------
// Reference Factory
// ---------------------------------------------------------------------------
inline LogicalId refFactoryGatherStreamingStorageSorted(const std::vector<LogicalId> &inputs,
    Graph &graph)
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

REGISTER_KERNEL("Gather_StreamingStorage_Sorted_NEON", 2, 2, matchGatherStreamingStorageSorted, runGatherStreamingStorageSorted, refFactoryGatherStreamingStorageSorted, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},                        // output backend
    {DType::BF16, DType::INT32},           // input types: raw weight (BF16), indices (INT32)
    {{248320, 2048}, {1, 8}},              // dummy shapes
    {true, true},                          // requires contiguous inputs
    {{MemSpace(1, HandleType::STORAGE)}, {MemSpace(1, HandleType::CPP)}}); // input placement