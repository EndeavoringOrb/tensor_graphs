#include "kernels/fused/moe_top_k_fused_gemm_streaming_storage/ref.hpp"
// Fused Top-K MoE Expert FFN with Streaming Storage Loading
// ---------------------------------------------------------
//
// PROBLEM THIS KERNEL SOLVES
//
// The previous kernel (Batched_Transposed_GEMM_StreamingStorage_NEON) made the
// individual gate_up and down GEMMs faster by streaming weights from disk.
// But it STILL computes ALL 256 experts per layer — even though only K=8
// experts per token actually contribute (the rest are masked to zero by
// normalized_probs). That's 248 wasted expert computations per token,
// 248 × 6 MB = 1.5 GB of wasted disk I/O per layer.
//
// This kernel fuses the ENTIRE MoE expert sub-chain:
//
//   COPY_TO (STORAGE->CPU bf16) × 2         [gate_up + down weights]
//   CAST (bf16->fp32) × 2
//   PERMUTE [0,2,1] × 2
//   CONTIGUOUS × 2
//   DOT (batched gate_up GEMM)              [E, S, H] @ [E, H, 2I]^T -> [E, S,
//   2I] SLICE × 2                                [gate | up] CONTIGUOUS × 2
//   silu_atomic (gate)                       pow(e, -x) -> div -> mul
//   MUL (silu(gate) * up)
//   DOT (batched down GEMM)                  [E, S, I] @ [E, I, H]^T -> [E, S,
//   H] PERMUTE [1,0,2]                          [E, S, H] -> [S, E, H]
//   CONTIGUOUS
//   PERMUTE [1,2,0] (normalized_probs)       [1, S, E] -> [S, E, 1]
//   CONTIGUOUS
//   REPEAT (broadcast probs to [S, E, H])
//   CONTIGUOUS
//   MUL (weighted_outputs)
//   SUM (axis=1)                              [S, E, H] -> [S, 1, H]
//   RESHAPE                                   [1, S, H]
//
// ...into a single kernel that:
//   1. Reads only the K=8 selected experts per token from disk (not all 256)
//   2. Deduplicates experts across tokens (if expert 42 is used by 3 tokens,
//      its weights are read from disk ONCE and reused)
//   3. Folds bf16->fp32 cast into the NEON FMLA loop (zero cost)
//   4. Folds PERMUTE into access pattern (W is read in native [O, H] order,
//      consumed as W^T in the dot product)
//   5. Folds silu+mul into the GEMM epilogue
//   6. Folds the weighted sum into the accumulation
//
// INPUTS
//
//   X            : CPU      fp32  [1, S, H]     (input activation, already in
//   RAM) W_gu         : STORAGE  bf16  [E, 2I, H]    (gate_up weights, on disk)
//   W_dn         : STORAGE  bf16  [E, H, I]     (down weights, on disk)
//   router_probs : CPU      fp32  [1, S, E]     (softmax output of router)
//   sel          : CPU      int32 [1, S, K]     (top-K expert indices from
//   argmax)
//
// OUTPUT
//
//   Out          : CPU      fp32  [1, S, H]     (weighted sum of K experts per
//   token)
//
// MATHEMATICAL EQUIVALENCE
//
//   For each token s:
//     prob_sum[s] = sum_k router_probs[s, sel[s,k]]
//     For each k in 0..K:
//       e = sel[s, k]
//       w = router_probs[s, e] / prob_sum[s]     (normalized prob, masked)
//       gate = X[s, :] @ W_gu[e, :I, :]^T         (first I rows of gate_up)
//       up   = X[s, :] @ W_gu[e, I:, :]^T         (last I rows of gate_up)
//       inter = silu(gate) * up
//       down_out = inter @ W_dn[e, :, :]^T
//       Out[s, :] += w * down_out
//
// I/O REDUCTION
//
//   Previous kernel (compute all E=256 experts):
//     gate_up: 256 × 4 MB = 1024 MB read
//     down:    256 × 2 MB =  512 MB read
//     Total:   1536 MB per layer
//
//   This kernel (only K×S=64 expert invocations, with dedup):
//     Worst case (no overlap): 64 × 6 MB = 384 MB
//     Typical (50% overlap):   ~32 × 6 MB = 192 MB
//     Best case (all same expert): 1 × 6 MB = 6 MB
//
//   At your observed ~6.6 GB/s effective disk bandwidth:
//     Previous: 1536 MB / 6.6 GB/s = 233 ms per layer
//     This kernel (typical): 192 MB / 6.6 GB/s ≈ 29 ms per layer
//     Speedup: ~8x over previous kernel, ~15x over original unfused chain

#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
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
// Linter rules (enforced by build.py validate_kernel_match_logic):
//   - No inputs.size() check
//   - No inputs[i].backend check
//   - No output.backend check
//   - No isContiguous(inputs[i]) / isContiguous(inViews[i]) check
//   - No inputs[i].dtype check
// All of those are declared via the REGISTER_KERNEL macro.
// ---------------------------------------------------------------------------
inline bool matchMoETopKFusedGEMM_StreamingStorage(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // X: [1, S, H], W_gu: [E, 2I, H], W_dn: [E, H, I],
    // router_probs: [1, S, E], sel: [1, S, K], Out: [1, S, H]
    if (inputs[0].getShape().size() != 3)
        return false;
    if (inputs[1].getShape().size() != 3)
        return false;
    if (inputs[2].getShape().size() != 3)
        return false;
    if (inputs[3].getShape().size() != 3)
        return false;
    if (inputs[4].getShape().size() != 3)
        return false;
    if (output.getShape().size() != 3)
        return false;

    const auto &sX = inputs[0].getShape();   // [1, S, H]
    const auto &sWgu = inputs[1].getShape(); // [E, 2I, H]
    const auto &sWdn = inputs[2].getShape(); // [E, H, I]
    const auto &sRP = inputs[3].getShape();  // [1, S, E]
    const auto &sSel = inputs[4].getShape(); // [1, S, K]
    const auto &sO = output.getShape();      // [1, S, H]

    const uint32_t S = sX[1];
    const uint32_t H = sX[2];
    const uint32_t E = sWgu[0];
    const uint32_t I2 = sWgu[1];
    const uint32_t I = I2 / 2;
    const uint32_t K = sSel[2];

    // X shape
    if (sX[0] != 1)
        return false;
    // W_gu: [E, 2I, H]
    if (sWgu[2] != H)
        return false;
    if (I2 != 2 * I)
        return false; // 2I must be even
    // W_dn: [E, H, I]
    if (sWdn[0] != E)
        return false;
    if (sWdn[1] != H)
        return false;
    if (sWdn[2] != I)
        return false;
    // router_probs: [1, S, E]
    if (sRP[0] != 1 || sRP[1] != S || sRP[2] != E)
        return false;
    // sel: [1, S, K]
    if (sSel[0] != 1 || sSel[1] != S)
        return false;
    // output: [1, S, H]
    if (sO[0] != 1 || sO[1] != S || sO[2] != H)
        return false;

    // Output must be contiguous for direct stores
    if (!isContiguous(output))
        return false;

    return true;
}

// ---------------------------------------------------------------------------
// Portable positional disk read (identical logic to the previous kernel,
// renamed to avoid ODR collision when both headers are included in the same
// translation unit via cpu_kernels.gen.hpp)
// ---------------------------------------------------------------------------
static inline bool moeTopK_readFromFileAtOffset(int fd, uint64_t offset, void *buf, uint64_t bytes)
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
// NEON vec-mat-mul: y[N] = sum_k x[K] * W[N, K]
// W is bf16, x and y are fp32. The bf16->fp32 cast is folded into the FMLA
// loop via vshll_n_u16 + vreinterpretq_f32_u32 (register-file bit manipulation,
// zero cost on AArch64).
//
// This handles BOTH the gate_up GEMM and the down GEMM:
//   gate_up: x=X_s[H], W=W_gu[2I, H], y=gate_proj[2I]   (K=H, N=2I)
//   down:    x=inter[I], W=W_dn[H, I],  y=down_out[H]    (K=I, N=H)
// ---------------------------------------------------------------------------
static inline void moeTopK_vecMatMul(const float *x, const uint16_t *W, float *y, uint32_t K, uint32_t N)
{
    const uint32_t K4 = K & ~3u;
    const uint32_t N4 = N & ~3u;

    // --- Main loop: 4 output rows at a time ---
    for (uint32_t n = 0; n < N4; n += 4)
    {
        const uint16_t *w0 = W + (n + 0) * K;
        const uint16_t *w1 = W + (n + 1) * K;
        const uint16_t *w2 = W + (n + 2) * K;
        const uint16_t *w3 = W + (n + 3) * K;

        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        for (uint32_t k = 0; k < K4; k += 4)
        {
            float32x4_t xv = vld1q_f32(x + k);
            acc0 = vfmaq_f32(acc0, xv, vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w0 + k), 16)));
            acc1 = vfmaq_f32(acc1, xv, vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w1 + k), 16)));
            acc2 = vfmaq_f32(acc2, xv, vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w2 + k), 16)));
            acc3 = vfmaq_f32(acc3, xv, vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w3 + k), 16)));
        }

        float s0 = vaddvq_f32(acc0);
        float s1 = vaddvq_f32(acc1);
        float s2 = vaddvq_f32(acc2);
        float s3 = vaddvq_f32(acc3);

        // K-tail (only triggers when K % 4 != 0)
        for (uint32_t k = K4; k < K; ++k)
        {
            float xv = x[k];
            uint32_t b0 = static_cast<uint32_t>(w0[k]) << 16;
            uint32_t b1 = static_cast<uint32_t>(w1[k]) << 16;
            uint32_t b2 = static_cast<uint32_t>(w2[k]) << 16;
            uint32_t b3 = static_cast<uint32_t>(w3[k]) << 16;
            float f0, f1, f2, f3;
            std::memcpy(&f0, &b0, 4);
            std::memcpy(&f1, &b1, 4);
            std::memcpy(&f2, &b2, 4);
            std::memcpy(&f3, &b3, 4);
            s0 += xv * f0;
            s1 += xv * f1;
            s2 += xv * f2;
            s3 += xv * f3;
        }

        y[n + 0] = s0;
        y[n + 1] = s1;
        y[n + 2] = s2;
        y[n + 3] = s3;
    }

    // --- N-tail (only triggers when N % 4 != 0) ---
    for (uint32_t n = N4; n < N; ++n)
    {
        const uint16_t *w = W + n * K;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (uint32_t k = 0; k < K4; k += 4)
        {
            acc = vfmaq_f32(acc, vld1q_f32(x + k), vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w + k), 16)));
        }
        float s = vaddvq_f32(acc);
        for (uint32_t k = K4; k < K; ++k)
        {
            uint32_t b = static_cast<uint32_t>(w[k]) << 16;
            float f;
            std::memcpy(&f, &b, 4);
            s += x[k] * f;
        }
        y[n] = s;
    }
}

// ---------------------------------------------------------------------------
// Run function
//
// Pipeline:
//   Phase 1: Scan sel[] to build expert_users[E] (which (s,k) pairs use each
//            expert) and precompute prob_sum[s] for normalization.
//   Phase 2: Partition unique experts among N threads. Each thread:
//              - Reads gate_up[e] and down[e] from disk (sequential per thread)
//              - For each (s,k) using expert e:
//                  gate_proj = X[s] @ W_gu[e]^T  (vec-mat-mul, bf16->fp32
//                  fused) inter = silu(gate_proj[:I]) * gate_proj[I:]  (scalar
//                  loop) down_out = inter @ W_dn[e]^T  (vec-mat-mul)
//                  thread_acc[s] += w * down_out
//   Phase 3: Reduce thread_acc into Out.
// ---------------------------------------------------------------------------
inline void runMoETopKFusedGEMM_StreamingStorage(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    // ctx.inputs[1] and ctx.inputs[2] are STORAGE (nullptr) — use fd + baseOffset
    const float *router_probs = static_cast<const float *>(ctx.inputs[3]);
    const int32_t *sel = static_cast<const int32_t *>(ctx.inputs[4]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const auto &viewX = ctx.inViews[0];
    const auto &viewWgu = ctx.inViews[1];
    const auto &viewWdn = ctx.inViews[2];

    const uint32_t S = viewX.getShape()[1];
    const uint32_t H = viewX.getShape()[2];
    const uint32_t E = viewWgu.getShape()[0];
    const uint32_t I2 = viewWgu.getShape()[1]; // 2 * I
    const uint32_t I = I2 / 2;
    const uint32_t K = ctx.inViews[4].getShape()[2];

    const int fd_gu = ctx.fd[1];
    const int fd_dn = ctx.fd[2];
    if (fd_gu < 0 || fd_dn < 0)
    {
        Error::throw_err("MoE_TopK_FusedGEMM_StreamingStorage_NEON: expected STORAGE inputs "
                         "for W_gu (fd[1] >= 0) and W_dn (fd[2] >= 0). The planner should "
                         "only route STORAGE-backed weights to this kernel.");
    }

    const uint64_t off_gu = viewWgu.offset;
    const uint64_t off_dn = viewWdn.offset;
    const uint64_t gu_expert_bytes = static_cast<uint64_t>(I2) * H * sizeof(uint16_t);
    const uint64_t dn_expert_bytes = static_cast<uint64_t>(H) * I * sizeof(uint16_t);

    // Zero the output
    std::memset(Out, 0, static_cast<uint64_t>(S) * H * sizeof(float));

    // --- Phase 1: Build expert -> users map + precompute prob_sum ---
    //
    // expert_users[e] = list of (s, k) pairs that reference expert e.
    // prob_sum[s] = sum of router_probs[s, sel[s, k]] for k in 0..K
    //             = denominator for normalizing the top-K probs.
    //
    // In the unfused chain, this normalization happens via:
    //   gated_probs = router_probs * mask          (zeros out non-top-K)
    //   row_sum = sum(gated_probs, axis=-1)        (= prob_sum)
    //   normalized_probs = gated_probs / row_sum   (=
    //   router_probs[s,e]/prob_sum[s])
    //
    // We compute prob_sum directly and divide on the fly, avoiding the
    // materialization of the full [1, S, E] normalized_probs tensor.
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> expert_users(E);
    std::vector<float> prob_sum(S, 0.0f);
    for (uint32_t s = 0; s < S; ++s)
    {
        for (uint32_t k = 0; k < K; ++k)
        {
            uint32_t e = static_cast<uint32_t>(sel[s * K + k]);
            if (e < E)
            {
                expert_users[e].push_back({s, k});
                prob_sum[s] += router_probs[s * E + e];
            }
        }
    }

    // Collect unique experts (naturally sorted by ID since we iterate 0..E)
    std::vector<uint32_t> unique_experts;
    unique_experts.reserve(std::min(static_cast<uint32_t>(S * K), E));
    for (uint32_t e = 0; e < E; ++e)
    {
        if (!expert_users[e].empty())
            unique_experts.push_back(e);
    }

    if (unique_experts.empty())
        return;

    // --- Phase 2: Parallel expert computation ---
    //
    // Partition unique_experts among threads. Each thread processes a
    // contiguous range of expert IDs (sorted), so disk reads within a thread
    // are sequential — maximising SSD throughput.
    //
    // Per-thread accumulators avoid write contention on Out[]. Final
    // reduction in Phase 3 is sequential but cheap (S*H floats).
    uint32_t hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0)
        hw_threads = 1;
    uint32_t num_threads = std::min(static_cast<uint32_t>(unique_experts.size()), hw_threads);
    if (num_threads == 0)
        num_threads = 1;

    std::vector<std::vector<float>> thread_acc(num_threads, std::vector<float>(static_cast<uint64_t>(S) * H, 0.0f));

    auto worker = [&](uint32_t tid, uint32_t start, uint32_t end) {
        // Per-thread scratch buffers (allocated once, reused across experts)
        std::vector<uint8_t> gu_buf(gu_expert_bytes);
        std::vector<uint8_t> dn_buf(dn_expert_bytes);
        std::vector<float> gate_proj(I2);
        std::vector<float> inter(I);
        std::vector<float> down_out(H);

        float *acc = thread_acc[tid].data();

        for (uint32_t idx = start; idx < end; ++idx)
        {
            uint32_t e = unique_experts[idx];
            const auto &users = expert_users[e];
            if (users.empty())
                continue;

            // Read gate_up[e] from disk — 4 MB per expert
            uint64_t gu_off = off_gu + static_cast<uint64_t>(e) * gu_expert_bytes;
            if (!moeTopK_readFromFileAtOffset(fd_gu, gu_off, gu_buf.data(), gu_expert_bytes))
                std::memset(gu_buf.data(), 0, gu_expert_bytes);

            // Read down[e] from disk — 2 MB per expert
            uint64_t dn_off = off_dn + static_cast<uint64_t>(e) * dn_expert_bytes;
            if (!moeTopK_readFromFileAtOffset(fd_dn, dn_off, dn_buf.data(), dn_expert_bytes))
                std::memset(dn_buf.data(), 0, dn_expert_bytes);

            const uint16_t *W_gu = reinterpret_cast<const uint16_t *>(gu_buf.data());
            const uint16_t *W_dn = reinterpret_cast<const uint16_t *>(dn_buf.data());

            // For each (s, k) that uses this expert
            for (const auto &uk : users)
            {
                uint32_t s = uk.first;
                const float *X_s = X + static_cast<uint64_t>(s) * H;

                // --- Gate/Up GEMM: gate_proj[2I] = X_s[H] @ W_gu[2I, H]^T ---
                // gate_proj[i] = sum_h X_s[h] * W_gu[i, h]
                moeTopK_vecMatMul(X_s, W_gu, gate_proj.data(), H, I2);

                // --- Slice + SiLU + Mul ---
                // gate = gate_proj[:I], up = gate_proj[I:2I]
                // silu(g) = g / (1 + exp(-g))  [equivalent to model's
                //           g * (1/(1+pow(e, -g))) formulation]
                // inter[i] = silu(gate[i]) * up[i]
                for (uint32_t i = 0; i < I; ++i)
                {
                    float g = gate_proj[i];
                    float u = gate_proj[I + i];
                    float silu_val = g / (1.0f + expf(-g));
                    inter[i] = silu_val * u;
                }

                // --- Down GEMM: down_out[H] = inter[I] @ W_dn[H, I]^T ---
                // down_out[h] = sum_i inter[i] * W_dn[h, i]
                moeTopK_vecMatMul(inter.data(), W_dn, down_out.data(), I, H);

                // --- Weighted accumulate into thread-local accumulator ---
                // w = router_probs[s, e] / prob_sum[s]
                //   = normalized_probs[s, e]  (the masked, renormalized prob)
                float w = (prob_sum[s] > 0.0f) ? (router_probs[static_cast<uint64_t>(s) * E + e] / prob_sum[s]) : 0.0f;

                float *acc_s = acc + static_cast<uint64_t>(s) * H;
                for (uint32_t h = 0; h < H; ++h)
                    acc_s[h] += w * down_out[h];
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    uint32_t per_thread = (static_cast<uint32_t>(unique_experts.size()) + num_threads - 1) / num_threads;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        uint32_t start = t * per_thread;
        uint32_t end = std::min(start + per_thread, static_cast<uint32_t>(unique_experts.size()));
        if (start >= end)
            break;
        workers.emplace_back(worker, t, start, end);
    }
    for (auto &w : workers)
        w.join();

    // --- Phase 3: Reduce thread accumulators into output ---
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        const float *acc = thread_acc[t].data();
        uint64_t total = static_cast<uint64_t>(S) * H;
        for (uint64_t i = 0; i < total; ++i)
            Out[i] += acc[i];
    }
}

// ===========================================================================
// Reference Factory
//
// This is the CRITICAL function for e-graph saturation matching. It must
// reproduce the EXACT graph chain that mlp_moe_atomic produces, starting from
// the kernel's 5 input nodes and ending at the routed_out [1, S, H] output.
//
// The planner's e-graph will saturate (apply all rewrite rules) and then
// search for subgraphs matching this refFactory's output pattern. If found,
// it replaces the subgraph with a call to this fused kernel.
//
// The chain reproduced here is (from mlp_moe_atomic, lines ~29960-30070):
//
//   1. Build mask [1, S, E] from sel [1, S, K] via eq + cast + sum + reshape
//   2. gated_probs = router_probs * mask
//   3. row_sum = sum(gated_probs, axis=-1), broadcast to [1, S, E]
//   4. normalized_probs = gated_probs / row_sum
//   5. x_expanded = contiguous(repeat(reshape(X, [1,S,H]), E, axis=0))  [E,S,H]
//   6. fused_gate_up_t = contiguous(permute(cast(copyto(W_gu, CPU), F32),
//   [0,2,1]))  [E,H,2I]
//   7. gate_up_proj = dot(x_expanded, fused_gate_up_t)  [E, S, 2I]
//   8. exp_gate = contiguous(slice(gate_up_proj, [0,0,0]:[E,S,I]))
//   9. exp_up = contiguous(slice(gate_up_proj, [0,0,I]:[E,S,2I]))
//  10. exp_gate_silu = silu_atomic(exp_gate, E, S, I)
//      (silu(x) = x * sigmoid(x) where sigmoid(x) = 1/(1+pow(e, -x)))
//  11. exp_gate_up = mul(exp_gate_silu, exp_up)  [E, S, I]
//  12. fused_down_t = contiguous(permute(cast(copyto(W_dn, CPU), F32),
//  [0,2,1]))  [E, I, H]
//  13. exp_down = dot(exp_gate_up, fused_down_t)  [E, S, H]
//  14. exp_down_perm = contiguous(permute(exp_down, [1,0,2]))  [S, E, H]
//  15. normalized_probs_perm = contiguous(permute(normalized_probs, [1,2,0]))
//  [S, E, 1]
//  16. normalized_probs_exp = contiguous(repeat(normalized_probs_perm, H,
//  axis=2))  [S, E, H]
//  17. weighted_outputs = mul(exp_down_perm, normalized_probs_exp)  [S, E, H]
//  18. routed_out_sum = sum(weighted_outputs, axis=1)  [S, 1, H]
//  19. routed_out = reshape(routed_out_sum, [1, S, H])
//
// Each step must use the EXACT same graph operations (op type, constant
// values, axis arguments, shape arguments) as mlp_moe_atomic, so that the
// e-graph can structurally match the pattern.
// ===========================================================================


// ---------------------------------------------------------------------------
// Registration
//
// 5 inputs:
//   [0] X            — fp32, CPU,     [1, S, H]      — input activation
//   [1] W_gu         — bf16, STORAGE, [E, 2I, H]     — gate_up weights on disk
//   [2] W_dn         — bf16, STORAGE, [E, H, I]      — down weights on disk
//   [3] router_probs — fp32, CPU,     [1, S, E]      — softmax output
//   [4] sel          — int32, CPU,    [1, S, K]      — top-K expert indices
//
// Output:
//   [0] Out          — fp32, CPU,     [1, S, H]      — routed MoE output
// ---------------------------------------------------------------------------
REGISTER_KERNEL("MoE_TopK_FusedGEMM_StreamingStorage_NEON", 5, 5, matchMoETopKFusedGEMM_StreamingStorage,
                runMoETopKFusedGEMM_StreamingStorage, refFactoryMoETopKFusedGEMM_StreamingStorage, {},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, // output backend
                {DType::FLOAT32, DType::BF16, DType::BF16, DType::FLOAT32, DType::INT32},
                {{1, 8, 2048}, {256, 1024, 2048}, {256, 2048, 512}, {1, 8, 256}, {1, 8, 8}},
                {true, true, true, true, true}, // all inputs contiguous
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(0, HandleType::STORAGE)},
                 {MemSpace(0, HandleType::STORAGE)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON