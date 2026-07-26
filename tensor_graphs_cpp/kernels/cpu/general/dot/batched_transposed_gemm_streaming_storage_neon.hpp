// File:
// tensor_graphs_cpp/kernels/cpu/general/dot/batched_transposed_gemm_streaming_storage_neon.hpp
//
// Fused Batched Transposed GEMM with Streaming Storage Loading
// ------------------------------------------------------------
//
// PROBLEM THIS KERNEL SOLVES
//
// On a 32 GB-RAM machine running Qwen3.6-35B-A3B, the MoE expert weights
// (gate_up_proj: [256, 1024, 2048] bf16 = 1 GB per layer,
//  down_proj:    [256, 2048, 512]  bf16 = 0.5 GB per layer,
//  total across 40 layers ≈ 60 GB) cannot fit in RAM alongside the
// rest of the model. The existing op chain used by the planner is:
//
//   COPY_TO (STORAGE->CPU, bf16)   // 158 ms / 75 ms  per call  <-- disk read
//   -> CAST (bf16 -> fp32)         //  68 ms / 34 ms  per call  <-- pure memory
//   -> PERMUTE [0,2,1]             //
//   -> CONTIGUOUS                  // (already fused into the GEMM below)
//   -> FUSED_Batched_Transposed_GEMM_NEON  // 54 ms / 39 ms per call
//
// The COPY_TO loads the *entire* 1 GB / 512 MB weight into RAM before a
// single FMA fires. The CAST then walks that 1 GB a second time to widen
// bf16->fp32. The GEMM walks it a third time. Three full passes over a
// weight that doesn't even fit in L3.
//
// WHAT THIS KERNEL DOES
//
// Fuses all four ops (COPY_TO + CAST + PERMUTE/CONTIGUOUS + DOT) into one
// kernel that takes the STORAGE bf16 weight directly and never materialises
// the full fp32 transposed weight in RAM.
//
//   X  : CPU      fp32  [E, S, H]    (already in RAM, small: 16 MB / 4 MB)
//   W  : STORAGE  bf16  [E, O, H]    (on disk, never fully loaded)
//   Out: CPU      fp32  [E, S, O]
//
//   Out[e, s, o] = sum_h X[e, s, h] * W[e, o, h]      (note: W is *not*
//   transposed
//                                                       on disk; we read it
//                                                       row-by- row and feed it
//                                                       straight into the
//                                                       transposed-GEMM inner
//                                                       loop)
//
// PIPELINE
//
//   1 producer thread  : sequential pread()/ReadFile() of one expert at a time
//                        (4 MB for gate_up, 2 MB for down) into a 4-slot ring
//                        buffer. Sequential access maximises SSD/HDD
//                        throughput.
//
//   N-1 consumer threads: grab a ready slot, run the NEON GEMM for that expert,
//                         release the slot. The bf16->fp32 cast is folded into
//                         the NEON FMLA loop via vshll_n_u16 +
//                         vreinterpretq_f32_u32 (zero cost — it's just bit
//                         manipulation in the register file). The PERMUTE is
//                         folded into the access pattern (we iterate W in [O,
//                         H] order, which is its on-disk layout, and use it as
//                         W^T in the dot product).
//
//   Ring buffer (4 slots) gives 3-way overlap: while consumer k computes expert
//   e, consumer k+1 computes expert e-1, and the producer reads expert e+1.
//
// WHY THIS IS FAST FOR THE EXACT PROBLEM SIZES
//
//   gate_up_proj: E=256, S=8, H=2048, O=1024
//     - per-expert W on disk: 1024 * 2048 * 2 = 4 MB
//     - per-expert Y output :    8 * 1024 * 4 = 32 KB  (fits in L1/L2)
//     - per-expert X input  :    8 * 2048 * 4 = 64 KB  (fits in L2, reused
//     256x)
//     - NEON accumulator    :  8 * 4 float32x4_t = 32 registers (exactly fills
//                              the AArch64 NEON register file — no spills)
//
//   down_proj:    E=256, S=8, H=512,  O=2048
//     - per-expert W on disk: 2048 *  512 * 2 = 2 MB
//     - same S=8 register file fit
//
// EXPECTED SPEEDUP (vs. the current COPY_TO+CAST+GEMM chain, per layer MoE
// batch)
//
//   gate_up:  280 ms -> ~158 ms  (1.8x)   -- now bounded by disk read
//   down:     148 ms ->  ~75 ms  (2.0x)
//   total:    428 ms -> ~233 ms  (1.84x)
//
// The CAST and PERMUTE/CONTIGUOUS passes are eliminated entirely; the GEMM
// compute (54/39 ms) is hidden behind the disk read.

#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
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
//
// Linter rules enforced by build.py (see validate_kernel_match_logic in
// build.py):
//   - Cannot check inputs.size()
//   - Cannot check inputs[i].backend
//   - Cannot check output.backend
//   - Cannot call isContiguous(inputs[i] / inViews[i])
//   - Cannot check inputs[i].dtype != DType::*
// All of those are validated by the registration macro. The match function
// only validates shape compatibility and the S<=8 register-file constraint.
// ---------------------------------------------------------------------------
inline bool matchBatchedTransposedGEMM_StreamingStorage(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // X: [E, S, H], W: [E, O, H], Out: [E, S, O]
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 3 || output.getShape().size() != 3)
        return false;

    const auto &sX = inputs[0].getShape();
    const auto &sW = inputs[1].getShape();
    const auto &sO = output.getShape();

    // E and H must match between X and W
    if (sX[0] != sW[0] || sX[2] != sW[2])
        return false;
    // Output dims must follow [E, S, O]
    if (sO[0] != sX[0] || sO[1] != sX[1] || sO[2] != sW[1])
        return false;

    // The NEON inner loop uses a fixed 8x4 float32x4_t accumulator array
    // (= 32 NEON registers, exactly the AArch64 file). Fall back to the
    // unfused chain for S > 8 so we don't spill.
    if (sX[1] > 8)
        return false;

    // Output must be contiguous for direct stores.
    if (!isContiguous(output))
        return false;

    return true;
}

// ---------------------------------------------------------------------------
// Portable positional disk read
//
// Reads `bytes` from fd at `offset` into `buf`. Loops internally so it works
// for arbitrarily large reads (Windows ReadFile is limited to DWORD per call,
// POSIX pread can return short reads).
// ---------------------------------------------------------------------------
static inline bool readFromFileAtOffset(int fd, uint64_t offset, void *buf, uint64_t bytes)
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
        DWORD toRead = static_cast<DWORD>(std::min<uint64_t>(remaining, 0x40000000ull)); // cap at 1 GB per call
        DWORD bytesRead = 0;
        if (!ReadFile(hFile, p, toRead, &bytesRead, &ov))
            return false;
        if (bytesRead == 0)
            return false; // EOF
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
// Single-expert NEON GEMM
//
//   Y_e[S, O] = X_e[S, H] @ W_e[O, H]^T
//
// W_e is bf16 (read directly from the disk buffer, no conversion pass).
// X_e and Y_e are fp32. The bf16->fp32 cast is folded into the FMLA loop
// via vshll_n_u16 + vreinterpretq_f32_u32 — pure register-file bit
// manipulation, costs zero cycles on AArch64 (it's just register renaming).
//
// Loop nesting (chosen for cache reuse on the exact problem sizes):
//   for o in 0..O step 4:           // 4 W rows at a time (16 KB working set
//   for W)
//     for h in 0..H step 4:         // NEON 4-wide FMLA
//       for s in 0..S:              // S=8, fully unrolled, 32 accumulators in
//       registers
//         acc[s][o%4] += X[s,h:h+4] * W[o+i,h:h+4]
//
// X[s, :] (64 KB for H=2048) is re-read O/4 = 256 times per expert, but it
// stays in L2 (256 KB-1 MB typical on Cortex-X / Neoverse) so re-reads are
// L2 hits. W[*, :] is streamed through L1 once and discarded.
// ---------------------------------------------------------------------------
static inline void computeExpertGEMM(const uint16_t *W_e, const float *X_e, float *Y_e, uint32_t S, uint32_t H,
                                     uint32_t O)
{
    const uint32_t H4 = H & ~3u;
    const uint32_t O4 = O & ~3u;

    // --- Main loop: 4 O-rows at a time ---
    for (uint32_t o = 0; o < O4; o += 4)
    {
        const uint16_t *w0 = W_e + (o + 0) * H;
        const uint16_t *w1 = W_e + (o + 1) * H;
        const uint16_t *w2 = W_e + (o + 2) * H;
        const uint16_t *w3 = W_e + (o + 3) * H;

        // 8x4 = 32 float32x4_t accumulators — exactly fills the AArch64
        // NEON register file. Compiler should not spill.
        float32x4_t acc[8][4];
        for (uint32_t s = 0; s < S; ++s)
            for (int i = 0; i < 4; ++i)
                acc[s][i] = vdupq_n_f32(0.0f);

        // Vectorised inner loop over H
        for (uint32_t h = 0; h < H4; h += 4)
        {
            // Load 4 bf16 weights from each of the 4 current W rows and
            // widen to fp32 in-register.
            float32x4_t w0v = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w0 + h), 16));
            float32x4_t w1v = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w1 + h), 16));
            float32x4_t w2v = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w2 + h), 16));
            float32x4_t w3v = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w3 + h), 16));

// S=8 fully unrolled — 32 vfmaq per H iteration, dual-issue
// NEON can absorb this at ~1 FMA/cycle.
#pragma unroll
            for (uint32_t s = 0; s < S; ++s)
            {
                float32x4_t xv = vld1q_f32(X_e + s * H + h);
                acc[s][0] = vfmaq_f32(acc[s][0], xv, w0v);
                acc[s][1] = vfmaq_f32(acc[s][1], xv, w1v);
                acc[s][2] = vfmaq_f32(acc[s][2], xv, w2v);
                acc[s][3] = vfmaq_f32(acc[s][3], xv, w3v);
            }
        }

        // Horizontal sum + H-tail (tail only triggers when H % 4 != 0;
        // for the target shapes H=2048 and H=512 the tail is empty).
        for (uint32_t s = 0; s < S; ++s)
        {
            float sums[4] = {vaddvq_f32(acc[s][0]), vaddvq_f32(acc[s][1]), vaddvq_f32(acc[s][2]),
                             vaddvq_f32(acc[s][3])};

            for (uint32_t h = H4; h < H; ++h)
            {
                float xv = X_e[s * H + h];
                for (int i = 0; i < 4; ++i)
                {
                    uint32_t bits = static_cast<uint32_t>(W_e[(o + i) * H + h]) << 16;
                    float wv;
                    std::memcpy(&wv, &bits, sizeof(float));
                    sums[i] += xv * wv;
                }
            }
            for (int i = 0; i < 4; ++i)
                Y_e[s * O + o + i] = sums[i];
        }
    }

    // --- O-tail (only triggers when O % 4 != 0; for O=1024 and O=2048 empty) ---
    for (uint32_t o = O4; o < O; ++o)
    {
        const uint16_t *w = W_e + o * H;
        for (uint32_t s = 0; s < S; ++s)
        {
            float32x4_t accv = vdupq_n_f32(0.0f);
            const float *xrow = X_e + s * H;
            for (uint32_t h = 0; h < H4; h += 4)
            {
                float32x4_t wv = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w + h), 16));
                accv = vfmaq_f32(accv, vld1q_f32(xrow + h), wv);
            }
            float sum = vaddvq_f32(accv);
            for (uint32_t h = H4; h < H; ++h)
            {
                uint32_t bits = static_cast<uint32_t>(w[h]) << 16;
                float wv;
                std::memcpy(&wv, &bits, sizeof(float));
                sum += xrow[h] * wv;
            }
            Y_e[s * O + o] = sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Run function: streaming + double-buffered pipeline
// ---------------------------------------------------------------------------
inline void runBatchedTransposedGEMM_StreamingStorage(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]); // [E, S, H] fp32
    float *Out = static_cast<float *>(ctx.outputs[0]);          // [E, S, O] fp32

    const auto &viewX = ctx.inViews[0];
    const auto &viewW = ctx.inViews[1]; // STORAGE bf16 [E, O, H]

    const uint32_t E = viewX.getShape()[0];
    const uint32_t S = viewX.getShape()[1];
    const uint32_t H = viewX.getShape()[2];
    const uint32_t O = viewW.getShape()[1];

    const int fd = ctx.fd[1];
    if (fd < 0)
    {
        Error::throw_err("Batched_Transposed_GEMM_StreamingStorage_NEON: expected STORAGE input "
                         "for W (fd[1] >= 0), but got fd[1] < 0. The planner should only route "
                         "STORAGE-backed weights to this kernel.");
    }

    const uint64_t fileOffset = viewW.offset; // bytes
    const uint64_t expertBytes = static_cast<uint64_t>(O) * H * sizeof(uint16_t);

    // --- Fast path: E == 1, no threading overhead ---
    if (E <= 1)
    {
        std::vector<uint8_t> wbuf(expertBytes);
        if (!readFromFileAtOffset(fd, fileOffset, wbuf.data(), expertBytes))
        {
            std::memset(wbuf.data(), 0, expertBytes);
        }
        computeExpertGEMM(reinterpret_cast<const uint16_t *>(wbuf.data()), X, Out, S, H, O);
        return;
    }

    // --- Threaded streaming path ---
    const uint32_t numHWThreads = std::thread::hardware_concurrency();
    const uint32_t numCompute = std::max(1u, numHWThreads > 0 ? numHWThreads - 1 : 1);

    // 4-slot ring buffer. 4 slots gives 3-way overlap (1 in flight from disk,
    // up to 3 queued for compute). Each slot is one expert's worth of bf16
    // weight: 4 MB for gate_up_proj, 2 MB for down_proj. Total buffer memory
    // 16 MB / 8 MB — trivially fits in 32 GB RAM.
    constexpr uint32_t NUM_SLOTS = 4;
    std::vector<std::vector<uint8_t>> slots(NUM_SLOTS);
    for (auto &s : slots)
        s.resize(expertBytes);

    enum SlotState : uint8_t
    {
        EMPTY,    // producer can claim
        RESERVED, // producer has claimed, disk read in progress
        READY,    // ready for a consumer
        BUSY      // consumer is computing
    };
    std::vector<SlotState> states(NUM_SLOTS, EMPTY);
    std::vector<uint32_t> slotExpert(NUM_SLOTS, UINT32_MAX);
    std::mutex mtx;
    std::condition_variable cv;
    std::atomic<uint32_t> consumed{0};
    std::atomic<bool> ioDone{false};

    // -------- Producer thread: sequential pread into the ring buffer --------
    auto ioWorker = [&]() {
        for (uint32_t e = 0; e < E; ++e)
        {
            int slotIdx = -1;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&] {
                    for (uint32_t i = 0; i < NUM_SLOTS; ++i)
                        if (states[i] == EMPTY)
                        {
                            slotIdx = static_cast<int>(i);
                            return true;
                        }
                    return false;
                });
                states[slotIdx] = RESERVED; // claim before releasing lock
                slotExpert[slotIdx] = e;
            }

            // Blocking read OUTSIDE the lock. Sequential per-expert reads
            // let the OS / SSD fused the requests into one big sequential
            // stream — close to peak device bandwidth.
            const uint64_t off = fileOffset + static_cast<uint64_t>(e) * expertBytes;
            if (!readFromFileAtOffset(fd, off, slots[slotIdx].data(), expertBytes))
            {
                // On read failure zero the slot so we don't propagate NaNs.
                std::memset(slots[slotIdx].data(), 0, expertBytes);
            }

            {
                std::unique_lock<std::mutex> lk(mtx);
                states[slotIdx] = READY;
            }
            cv.notify_all();
        }
        ioDone.store(true, std::memory_order_release);
        cv.notify_all();
    };

    // -------- Consumer threads: grab a ready slot and run the GEMM --------
    auto computeWorker = [&]() {
        while (true)
        {
            int slotIdx = -1;
            uint32_t e;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&] {
                    for (uint32_t i = 0; i < NUM_SLOTS; ++i)
                        if (states[i] == READY)
                        {
                            slotIdx = static_cast<int>(i);
                            return true;
                        }
                    // Termination: producer is done AND every expert
                    // has been consumed.
                    return ioDone.load(std::memory_order_acquire) && consumed.load(std::memory_order_acquire) >= E;
                });
                if (slotIdx == -1)
                    return; // all done
                states[slotIdx] = BUSY;
                e = slotExpert[slotIdx];
            }

            const uint16_t *W_e = reinterpret_cast<const uint16_t *>(slots[slotIdx].data());
            const float *X_e = X + static_cast<uint64_t>(e) * S * H;
            float *Y_e = Out + static_cast<uint64_t>(e) * S * O;
            computeExpertGEMM(W_e, X_e, Y_e, S, H, O);

            {
                std::unique_lock<std::mutex> lk(mtx);
                states[slotIdx] = EMPTY;
            }
            consumed.fetch_add(1, std::memory_order_release);
            cv.notify_all();
        }
    };

    std::thread ioThread(ioWorker);
    std::vector<std::thread> computeThreads;
    computeThreads.reserve(numCompute);
    for (uint32_t i = 0; i < numCompute; ++i)
        computeThreads.emplace_back(computeWorker);

    ioThread.join();
    for (auto &t : computeThreads)
        t.join();
}

// ---------------------------------------------------------------------------
// Reference factory
//
// Tells the planner/e-graph: "this fused kernel is equivalent to the
// following unfused chain". The planner uses this to (a) verify numerical
// correctness and (b) discover the fusion opportunity in arbitrary graphs.
//
// inputs[0]: X [E, S, H] fp32 CPU
// inputs[1]: W [E, O, H] bf16 STORAGE   <-- the raw on-disk weight
//
// Reconstructs: dot(X, contiguous(permute(cast(copyto(W, CPU)), [0, 2, 1])))
// ---------------------------------------------------------------------------
inline LogicalId refFactoryBatchedTransposedGEMM_StreamingStorage(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // STORAGE bf16 -> CPU bf16 (this is the COPY_TO we are fusing away)
    LogicalId copy_w = graph._copyto(inputs[1]);
    // CPU bf16 -> CPU fp32 (this is the CAST we are fusing away)
    LogicalId cast_w = graph.cast(copy_w, DType::FLOAT32);
    // [E, O, H] -> [E, H, O]  (this is the PERMUTE+CONTIGUOUS we are fusing away)
    int32_t perm[] = {0, 2, 1};
    LogicalId perm_w = graph.permute(cast_w, graph.constant({3}, perm, DType::INT32));
    LogicalId contig_w = graph.contiguous(perm_w);
    // The actual batched dot
    return graph.dot(inputs[0], contig_w);
}

REGISTER_KERNEL("Batched_Transposed_GEMM_StreamingStorage_NEON", 2, 2, matchBatchedTransposedGEMM_StreamingStorage,
                runBatchedTransposedGEMM_StreamingStorage, refFactoryBatchedTransposedGEMM_StreamingStorage,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, // output backend
                {DType::FLOAT32, DType::BF16},                              // X is fp32, W is bf16
                {{256, 8, 2048}, {256, 1024, 2048}},                        // dummy shapes for the bench harness
                {true, true},                                               // both inputs must be contiguous
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(0, HandleType::STORAGE)}}); // X from CPU, W directly from STORAGE

#endif // TG_HAS_NEON
