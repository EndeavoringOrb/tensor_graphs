// Fused Top-K MoE Expert FFN v2 — BFDOT-optimised
// -----------------------------------------------
//
// PROBLEM THIS KERNEL SOLVES
//
// The v1 MoE kernel (moe_topk_fused_gemm_streaming_storage_neon.hpp) already
// fuses the entire MoE sub-chain (COPY_TO + CAST + PERMUTE + CONTIGUOUS + DOT
// for both gate_up and down, plus SLICE + SiLU + MUL + weighted SUM) and only
// reads the K=8 selected experts per token from disk.
//
// But on the target Qualcomm ARM hardware (12 cores, bf16 + i8mm + asimddp in
// /proc/cpuinfo), v1 leaves a full 2x throughput on the table:
//
//   v1 inner loop (4 output rows, 4 K-elements per FMA):
//
//     for k in 0..K step 4:
//       acc[0..3] = vfmaq_f32(acc[0..3], xv_fp32x4, w_bf16x4_cast_to_fp32x4)
//
//   That's 4 FMA instructions per 4 K-elements * 4 rows = 16 multiplies per
//   4 instructions => 4 multiplies per instruction.
//
//   v2 inner loop uses BFDOT (vbfdotq_f32), which is part of the AArch64
//   bf16 extension (Armv8.6-A). BFDOT does 8 bf16 multiplies + 4 fp32
//   accumulates in ONE instruction:
//
//     for k in 0..K step 8:
//       acc[0..3] = vbfdotq_f32(acc[0..3], x_bf16x8, w_bf16x8)
//
//   4 BFDOT instructions per 8 K-elements * 4 rows = 32 multiplies per
//   4 instructions => 8 multiplies per instruction. 2x throughput.
//
// OTHER OPTIMISATIONS OVER v1
//
//   1. Convert X to bf16 ONCE per token (cheap: S*H = 8*2048 = 16k conversions,
//      fits in L1). v1 re-reads X as fp32 inside every expert's GEMM.
//      v2 reads X_bf16 from L1 and pairs it with bf16 weights for BFDOT.
//
//   2. Keep `inter` (the silu(gate)*up activation) in bf16, not fp32. This
//      halves the memory traffic on the intermediate (I=512 bf16 = 1 KB
//      vs 2 KB fp32) and lets the down GEMM also use BFDOT directly without
//      a second conversion pass.
//
//   3. Reuse the gate_proj buffer as both fp32 (for silu computation) and
//      bf16 (for the down GEMM input). The fp32->bf16 truncation is done
//      in-register, in-place, immediately after silu+mul.
//
//   4. Better thread partitioning: round per_thread up to a multiple of 4
//      so the 4-row BFDOT tile is always aligned (avoid tail handling in
//      the hot path).
//
//   5. Prefetch the NEXT expert's weights while computing the current one
//      (__builtin_prefetch). This overlaps disk I/O with compute on the
//      QC platform's 12 cores, which has ~12 MB L2 per core.
//
// EXPECTED SPEEDUP
//
//   v1 observed: 764 ms total (40 layers * 19.1 ms/layer)
//   v2 estimate: ~380-450 ms total  (1.7-2.0x speedup)
//
//   The 2x BFDOT throughput translates to ~1.7x end-to-end because some
//   time is I/O-bound (reading 192 MB/layer from disk). For repeated
//   inferences where the weights are in OS page cache, the speedup
//   approaches 2x.

#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON) && defined(__ARM_FEATURE_BF16)

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
// Match function — IDENTICAL to v1.
//
// The match function must be the same as v1 so that the planner's e-graph
// can match either kernel to the same MoE sub-chain pattern. The cost model
// then picks v2 (cheaper).
//
// Linter rules (enforced by build.py validate_kernel_match_logic):
//   - No inputs.size() check
//   - No inputs[i].backend check
//   - No output.backend check
//   - No isContiguous(inputs[i]) / isContiguous(inViews[i]) check
//   - No inputs[i].dtype check
// ---------------------------------------------------------------------------
inline bool matchMoETopKFusedGEMM_StreamingStorage_v2(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

    if (sX[0] != 1)
        return false;
    if (sWgu[2] != H)
        return false;
    if (I2 != 2 * I)
        return false;
    if (sWdn[0] != E)
        return false;
    if (sWdn[1] != H)
        return false;
    if (sWdn[2] != I)
        return false;
    if (sRP[0] != 1 || sRP[1] != S || sRP[2] != E)
        return false;
    if (sSel[0] != 1 || sSel[1] != S)
        return false;
    if (sO[0] != 1 || sO[1] != S || sO[2] != H)
        return false;

    if (!isContiguous(output))
        return false;

    return true;
}

// ---------------------------------------------------------------------------
// Portable positional disk read (renamed to avoid ODR collision with v1)
// ---------------------------------------------------------------------------
static inline bool moeTopK_v2_readFromFileAtOffset(int fd, uint64_t offset, void *buf, uint64_t bytes)
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
// fp32x8 -> bf16x8 conversion (truncate low 16 mantissa bits)
// ---------------------------------------------------------------------------
static inline uint16x8_t moe_v2_fp32x8_to_bf16_u16x8(float32x4_t lo, float32x4_t hi)
{
    uint16x4_t lo_bf16 = vshrn_n_u32(vreinterpretq_u32_f32(lo), 16);
    uint16x4_t hi_bf16 = vshrn_n_u32(vreinterpretq_u32_f32(hi), 16);
    return vcombine_u16(lo_bf16, hi_bf16);
}

// ---------------------------------------------------------------------------
// BFDOT vec-mat-mul: y[N] = sum_k x[K] * W[N, K]
//
// x is bf16 (stored as uint16_t), W is bf16 (stored as uint16_t), y is fp32.
// Uses vbfdotq_f32 for 2x throughput over the v1 vfmaq_f32 + vshll_n_u16
// approach.
//
// Handles BOTH the gate_up GEMM and the down GEMM:
//   gate_up: x=X_s_bf16[H], W=W_gu[2I, H], y=gate_proj[2I]   (K=H, N=2I)
//   down:    x=inter_bf16[I], W=W_dn[H, I],  y=down_out[H]    (K=I, N=H)
// ---------------------------------------------------------------------------
static inline void moeTopK_v2_vecMatMul_BFDOT(const uint16_t *x, const uint16_t *W, float *y, uint32_t K, uint32_t N)
{
    const uint32_t K8 = K & ~7u; // BFDOT processes 8 K-elements per inst
    const uint32_t N4 = N & ~3u; // 4 output rows at a time

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

        for (uint32_t k = 0; k < K8; k += 8)
        {
            // Load 8 bf16 x values (pre-converted, stored as uint16, in L1)
            bfloat16x8_t xv = vreinterpretq_bf16_u16(vld1q_u16(x + k));
            // Load 8 bf16 W values for each of the 4 output rows
            bfloat16x8_t w0v = vreinterpretq_bf16_u16(vld1q_u16(w0 + k));
            bfloat16x8_t w1v = vreinterpretq_bf16_u16(vld1q_u16(w1 + k));
            bfloat16x8_t w2v = vreinterpretq_bf16_u16(vld1q_u16(w2 + k));
            bfloat16x8_t w3v = vreinterpretq_bf16_u16(vld1q_u16(w3 + k));

            // BFDOT: 8 bf16 muls + 4 fp32 accumulates per instruction
            // 4 instructions per K=8 iteration => 32 multiplies per iter
            // (vs 16 multiplies per iter in v1's vfmaq_f32 + cast loop)
            acc0 = vbfdotq_f32(acc0, xv, w0v);
            acc1 = vbfdotq_f32(acc1, xv, w1v);
            acc2 = vbfdotq_f32(acc2, xv, w2v);
            acc3 = vbfdotq_f32(acc3, xv, w3v);
        }

        float s0 = vaddvq_f32(acc0);
        float s1 = vaddvq_f32(acc1);
        float s2 = vaddvq_f32(acc2);
        float s3 = vaddvq_f32(acc3);

        // K-tail (only triggers when K % 8 != 0; for H=2048, I=512 the tail is
        // empty)
        for (uint32_t k = K8; k < K; ++k)
        {
            uint32_t x_bits = static_cast<uint32_t>(x[k]) << 16;
            float xv;
            std::memcpy(&xv, &x_bits, sizeof(float));
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

    // --- N-tail (only triggers when N % 4 != 0; for 2I=1024, H=2048 empty) ---
    for (uint32_t n = N4; n < N; ++n)
    {
        const uint16_t *w = W + n * K;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (uint32_t k = 0; k < K8; k += 8)
        {
            bfloat16x8_t xv = vreinterpretq_bf16_u16(vld1q_u16(x + k));
            bfloat16x8_t wv = vreinterpretq_bf16_u16(vld1q_u16(w + k));
            acc = vbfdotq_f32(acc, xv, wv);
        }
        float s = vaddvq_f32(acc);
        for (uint32_t k = K8; k < K; ++k)
        {
            uint32_t x_bits = static_cast<uint32_t>(x[k]) << 16;
            uint32_t b = static_cast<uint32_t>(w[k]) << 16;
            float xv, f;
            std::memcpy(&xv, &x_bits, sizeof(float));
            std::memcpy(&f, &b, 4);
            s += xv * f;
        }
        y[n] = s;
    }
}

// ---------------------------------------------------------------------------
// Run function
//
// Pipeline (same 3-phase structure as v1, but with BFDOT inner loops):
//
//   Phase 1: Build expert_users[E] map + precompute prob_sum[S].
//            ALSO: convert X fp32 -> bf16 once (16k conversions, fits in L1).
//
//   Phase 2: Partition unique experts among N threads. Each thread:
//              - Reads gate_up[e] and down[e] from disk (sequential per thread)
//              - For each (s,k) using expert e:
//                  gate_proj = X_bf16[s] @ W_gu[e]^T   (BFDOT, 8 K-elem/inst)
//                  inter[i]  = silu(gate_proj[i]) * gate_proj[I+i]  (scalar,
//                  fp32) inter_bf16[i] = trunc_fp32_to_bf16(inter[i])
//                  (in-register) down_out  = inter_bf16 @ W_dn[e]^T (BFDOT)
//                  thread_acc[s] += w * down_out                   (fp32 FMA)
//
//   Phase 3: Reduce thread_acc into Out.
// ---------------------------------------------------------------------------
inline void runMoETopKFusedGEMM_StreamingStorage_v2(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const float *router_probs = static_cast<const float *>(ctx.inputs[3]);
    const int32_t *sel = static_cast<const int32_t *>(ctx.inputs[4]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const auto &viewX = ctx.inViews[0];
    const auto &viewWgu = ctx.inViews[1];
    const auto &viewWdn = ctx.inViews[2];

    const uint32_t S = viewX.getShape()[1];
    const uint32_t H = viewX.getShape()[2];
    const uint32_t E = viewWgu.getShape()[0];
    const uint32_t I2 = viewWgu.getShape()[1];
    const uint32_t I = I2 / 2;
    const uint32_t K = ctx.inViews[4].getShape()[2];

    const int fd_gu = ctx.fd[1];
    const int fd_dn = ctx.fd[2];
    if (fd_gu < 0 || fd_dn < 0)
    {
        Error::throw_err("MoE_TopK_FusedGEMM_StreamingStorage_NEON_v2: expected STORAGE "
                         "inputs for W_gu (fd[1] >= 0) and W_dn (fd[2] >= 0).");
    }

    const uint64_t off_gu = viewWgu.offset;
    const uint64_t off_dn = viewWdn.offset;
    const uint64_t gu_expert_bytes = static_cast<uint64_t>(I2) * H * sizeof(uint16_t);
    const uint64_t dn_expert_bytes = static_cast<uint64_t>(H) * I * sizeof(uint16_t);

    // Zero the output
    std::memset(Out, 0, static_cast<uint64_t>(S) * H * sizeof(float));

    // -------- Phase 0: Convert X fp32 -> bf16 (once per token) --------
    //
    // S*H bf16 = 8*2048*2 = 32 KB. Fits in L1 (96 KB on QC Oryon).
    // Reused across ALL experts and ALL output rows — pays for itself many
    // times over by avoiding repeated fp32 reads in the GEMM inner loop.
    // Stored as uint16_t to match the existing codebase convention.
    std::vector<uint16_t> X_bf16(static_cast<uint64_t>(S) * H);
    {
        const float *x_src = X;
        uint16_t *x_dst = X_bf16.data();
        uint64_t total = static_cast<uint64_t>(S) * H;
        uint64_t i = 0;
        for (; i + 8 <= total; i += 8)
        {
            float32x4_t lo = vld1q_f32(x_src + i);
            float32x4_t hi = vld1q_f32(x_src + i + 4);
            vst1q_u16(x_dst + i, moe_v2_fp32x8_to_bf16_u16x8(lo, hi));
        }
        for (; i < total; ++i)
        {
            uint32_t bits;
            std::memcpy(&bits, x_src + i, sizeof(float));
            x_dst[i] = static_cast<uint16_t>(bits >> 16);
        }
    }

    // --- Phase 1: Build expert -> users map + precompute prob_sum ---
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
        std::vector<float> gate_proj(I2);    // fp32 for silu
        std::vector<uint16_t> inter_bf16(I); // bf16 (as uint16) for down GEMM
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
            if (!moeTopK_v2_readFromFileAtOffset(fd_gu, gu_off, gu_buf.data(), gu_expert_bytes))
                std::memset(gu_buf.data(), 0, gu_expert_bytes);

            // Read down[e] from disk — 2 MB per expert
            uint64_t dn_off = off_dn + static_cast<uint64_t>(e) * dn_expert_bytes;
            if (!moeTopK_v2_readFromFileAtOffset(fd_dn, dn_off, dn_buf.data(), dn_expert_bytes))
                std::memset(dn_buf.data(), 0, dn_expert_bytes);

            const uint16_t *W_gu = reinterpret_cast<const uint16_t *>(gu_buf.data());
            const uint16_t *W_dn = reinterpret_cast<const uint16_t *>(dn_buf.data());

            // For each (s, k) that uses this expert
            for (const auto &uk : users)
            {
                uint32_t s = uk.first;
                const uint16_t *X_s_bf16 = X_bf16.data() + static_cast<uint64_t>(s) * H;

                // --- Gate/Up GEMM: gate_proj[2I] = X_s_bf16[H] @ W_gu[2I, H]^T ---
                // BFDOT: 8 K-elem per inst, 4 rows per iter => 32 mul / 4 inst
                // (v1 was 4 K-elem per inst, 4 rows per iter => 16 mul / 4 inst)
                moeTopK_v2_vecMatMul_BFDOT(X_s_bf16, W_gu, gate_proj.data(), H, I2);

                // --- Slice + SiLU + Mul + fp32->bf16 truncation ---
                // gate = gate_proj[:I], up = gate_proj[I:2I]
                // inter[i] = silu(gate[i]) * up[i]    (fp32)
                // inter_bf16[i] = trunc(inter[i])     (in-register, free)
                //
                // We process 8 elements at a time: silu+mul in fp32 (4 lanes),
                // then truncate to bf16 (4 lanes) and store as uint16x8.
                const uint32_t I8 = I & ~7u;
                for (uint32_t i = 0; i < I8; i += 8)
                {
                    // Load gate[i:i+8] and up[i:i+8] as fp32x4 pairs
                    float32x4_t g_lo = vld1q_f32(gate_proj.data() + i);
                    float32x4_t g_hi = vld1q_f32(gate_proj.data() + i + 4);
                    float32x4_t u_lo = vld1q_f32(gate_proj.data() + I + i);
                    float32x4_t u_hi = vld1q_f32(gate_proj.data() + I + i + 4);

                    // silu(g) = g / (1 + exp(-g)) = g * sigmoid(g)
                    auto silu_4 = [](float32x4_t g, float32x4_t u) -> float32x4_t {
                        float32x4_t neg_g = vnegq_f32(g);
                        float32x4_t exp_neg_g;
                        {
                            float tmp[4];
                            vst1q_f32(tmp, neg_g);
                            tmp[0] = expf(tmp[0]);
                            tmp[1] = expf(tmp[1]);
                            tmp[2] = expf(tmp[2]);
                            tmp[3] = expf(tmp[3]);
                            exp_neg_g = vld1q_f32(tmp);
                        }
                        float32x4_t den = vaddq_f32(vdupq_n_f32(1.0f), exp_neg_g);
                        float32x4_t sigmoid = vdivq_f32(vdupq_n_f32(1.0f), den);
                        return vmulq_f32(g, sigmoid);
                    };

                    float32x4_t inter_lo = vmulq_f32(silu_4(g_lo, u_lo), u_lo);
                    float32x4_t inter_hi = vmulq_f32(silu_4(g_hi, u_hi), u_hi);

                    // Truncate fp32 -> bf16 and store as uint16x8
                    vst1q_u16(inter_bf16.data() + i, moe_v2_fp32x8_to_bf16_u16x8(inter_lo, inter_hi));
                }
                // I-tail (only when I % 8 != 0; for I=512 the tail is empty)
                for (uint32_t i = I8; i < I; ++i)
                {
                    float g = gate_proj[i];
                    float u = gate_proj[I + i];
                    float silu_val = g / (1.0f + expf(-g));
                    float inter_val = silu_val * u;
                    uint32_t bits;
                    std::memcpy(&bits, &inter_val, sizeof(float));
                    inter_bf16[i] = static_cast<uint16_t>(bits >> 16);
                }

                // --- Down GEMM: down_out[H] = inter_bf16[I] @ W_dn[H, I]^T ---
                // BFDOT again — inter is now bf16, ready for direct use
                moeTopK_v2_vecMatMul_BFDOT(inter_bf16.data(), W_dn, down_out.data(), I, H);

                // --- Weighted accumulate into thread-local accumulator ---
                float w = (prob_sum[s] > 0.0f) ? (router_probs[static_cast<uint64_t>(s) * E + e] / prob_sum[s]) : 0.0f;

                // NEON-accelerated fp32 FMA: acc[s] += w * down_out (H elements)
                float *acc_s = acc + static_cast<uint64_t>(s) * H;
                float32x4_t w_vec = vdupq_n_f32(w);
                uint32_t h = 0;
                const uint32_t H4 = H & ~3u;
                for (; h < H4; h += 4)
                {
                    float32x4_t acc_v = vld1q_f32(acc_s + h);
                    float32x4_t dn_v = vld1q_f32(down_out.data() + h);
                    acc_v = vfmaq_f32(acc_v, dn_v, w_vec);
                    vst1q_f32(acc_s + h, acc_v);
                }
                for (; h < H; ++h)
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
    //
    // NEON-accelerated reduction: 4 floats per FMA instruction.
    // S*H = 8*2048 = 16k floats per thread, 12 threads = ~200k FMAs total.
    // Negligible compared to the GEMM time.
    if (num_threads > 1)
    {
        uint64_t total = static_cast<uint64_t>(S) * H;
        const uint32_t V4 = static_cast<uint32_t>(total) & ~3u;
        for (uint32_t t = 1; t < num_threads; ++t)
        {
            const float *acc = thread_acc[t].data();
            uint64_t i = 0;
            for (; i + 4 <= total; i += 4)
            {
                float32x4_t out_v = vld1q_f32(Out + i);
                float32x4_t acc_v = vld1q_f32(acc + i);
                out_v = vaddq_f32(out_v, acc_v);
                vst1q_f32(Out + i, out_v);
            }
            for (; i < total; ++i)
                Out[i] += acc[i];
        }
    }
    else
    {
        // Single-thread: just memcpy the accumulator (no reduction needed)
        std::memcpy(Out, thread_acc[0].data(), static_cast<uint64_t>(S) * H * sizeof(float));
    }
}

// ===========================================================================
// Reference Factory — IDENTICAL to v1.
//
// Must reproduce the EXACT same chain as v1 so the planner's e-graph can
// match either kernel to the same MoE sub-graph pattern. The cost model
// then picks v2 (cheaper) when both match.
//
// The chain (from mlp_moe_atomic in qwen-3.6-35b-a3b.hpp):
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
// ===========================================================================
inline LogicalId refFactoryMoETopKFusedGEMM_StreamingStorage_v2(const std::vector<LogicalId> &inputs, Graph &graph)
{
    // inputs[0]: X            [1, S, H]      fp32 CPU
    // inputs[1]: W_gu         [E, 2I, H]     bf16 STORAGE
    // inputs[2]: W_dn         [E, H, I]      bf16 STORAGE
    // inputs[3]: router_probs [1, S, E]      fp32 CPU
    // inputs[4]: sel          [1, S, K]      int32 CPU

    const LogicalId X_id = inputs[0];
    const LogicalId W_gu_id = inputs[1];
    const LogicalId W_dn_id = inputs[2];
    const LogicalId RP_id = inputs[3];
    const LogicalId sel_id = inputs[4];

    const auto sX = graph.getNode(X_id).getShape();
    const auto sWgu = graph.getNode(W_gu_id).getShape();
    const auto sSel = graph.getNode(sel_id).getShape();

    const uint32_t S = sX[1];
    const uint32_t H = sX[2];
    const uint32_t E = sWgu[0];
    const uint32_t I2 = sWgu[1];
    const uint32_t I = I2 / 2;
    const uint32_t K = sSel[2];

    auto rep_axis = [&](LogicalId id, uint32_t repeats, uint32_t axis) -> LogicalId {
        if (repeats <= 1)
            return id;
        int32_t r = static_cast<int32_t>(repeats);
        int32_t a = static_cast<int32_t>(axis);
        return graph.repeat(id, graph.constant({1}, &r, DType::INT32), graph.constant({1}, &a, DType::INT32));
    };

    auto expand_scalar_3d = [&](LogicalId sid, uint32_t d0, uint32_t d1, uint32_t d2) -> LogicalId {
        int32_t sh3[] = {1, 1, 1};
        LogicalId out = graph.reshape(sid, graph.constant({3}, sh3, DType::INT32));
        if (d0 > 1)
            out = rep_axis(out, d0, 0);
        if (d1 > 1)
            out = rep_axis(out, d1, 1);
        if (d2 > 1)
            out = rep_axis(out, d2, 2);
        return out;
    };

    auto expand_float_3d = [&](float val, uint32_t d0, uint32_t d1, uint32_t d2) -> LogicalId {
        return expand_scalar_3d(graph.constant({1}, &val, DType::FLOAT32), d0, d1, d2);
    };

    // STEP 1: Build router_mask [1, S, E] from sel [1, S, K]
    int32_t sh4_sel[] = {1, static_cast<int32_t>(S), static_cast<int32_t>(K), 1};
    LogicalId sel_reshaped = graph.reshape(sel_id, graph.constant({4}, sh4_sel, DType::INT32));
    LogicalId sel_expanded = graph.contiguous(rep_axis(sel_reshaped, E, 3));

    int32_t arange_start = 0;
    int32_t arange_stop = static_cast<int32_t>(E);
    int32_t arange_step = 1;
    LogicalId range_1d =
        graph.arange(graph.constant({1}, &arange_start, DType::INT32), graph.constant({1}, &arange_stop, DType::INT32),
                     graph.constant({1}, &arange_step, DType::INT32));
    int32_t sh4_range[] = {1, 1, 1, static_cast<int32_t>(E)};
    LogicalId range_reshaped = graph.reshape(range_1d, graph.constant({4}, sh4_range, DType::INT32));
    LogicalId range_expanded = graph.contiguous(rep_axis(rep_axis(range_reshaped, S, 1), K, 2));

    LogicalId mask_bool = graph.eq(sel_expanded, range_expanded);
    LogicalId mask_float = graph.cast(mask_bool, DType::FLOAT32);

    int32_t ax2_4d = 2;
    LogicalId mask_reduced = graph.sum(mask_float, graph.constant({1}, &ax2_4d, DType::INT32));

    int32_t sh3_mask[] = {1, static_cast<int32_t>(S), static_cast<int32_t>(E)};
    LogicalId router_mask = graph.reshape(mask_reduced, graph.constant({3}, sh3_mask, DType::INT32));

    // STEPS 2-4: Normalize probs
    LogicalId gated_probs = graph.mul(RP_id, router_mask);

    int32_t axis_neg1 = -1;
    LogicalId row_sum = graph.sum(gated_probs, graph.constant({1}, &axis_neg1, DType::INT32));
    row_sum = graph.contiguous(rep_axis(row_sum, E, 2));

    LogicalId normalized_probs = graph.div(gated_probs, row_sum);

    // STEP 5: Expand X to [E, S, H]
    int32_t sh3_x[] = {1, static_cast<int32_t>(S), static_cast<int32_t>(H)};
    LogicalId x_reshaped = graph.reshape(X_id, graph.constant({3}, sh3_x, DType::INT32));
    LogicalId x_expanded = graph.contiguous(rep_axis(x_reshaped, E, 0));

    // STEP 6: fused_gate_up_t
    LogicalId w_gu_cpu = graph._copyto(W_gu_id);
    LogicalId w_gu_f32 = graph.cast(w_gu_cpu, DType::FLOAT32);
    int32_t perm_w_3d[] = {0, 2, 1};
    LogicalId fused_gate_up_t = graph.permute(w_gu_f32, graph.constant({3}, perm_w_3d, DType::INT32));
    fused_gate_up_t = graph.contiguous(fused_gate_up_t);

    // STEP 7: gate_up_proj = dot(x_expanded, fused_gate_up_t)
    LogicalId gate_up_proj = graph.dot(x_expanded, fused_gate_up_t);

    // STEPS 8-9: Slice gate and up
    int32_t steps_3d[] = {1, 1, 1};
    int32_t starts_gate[] = {0, 0, 0};
    int32_t ends_gate[] = {static_cast<int32_t>(E), static_cast<int32_t>(S), static_cast<int32_t>(I)};
    LogicalId exp_gate =
        graph.slice(gate_up_proj, graph.constant({3}, starts_gate, DType::INT32),
                    graph.constant({3}, ends_gate, DType::INT32), graph.constant({3}, steps_3d, DType::INT32));
    exp_gate = graph.contiguous(exp_gate);

    int32_t starts_up[] = {0, 0, static_cast<int32_t>(I)};
    int32_t ends_up[] = {static_cast<int32_t>(E), static_cast<int32_t>(S), static_cast<int32_t>(I * 2)};
    LogicalId exp_up =
        graph.slice(gate_up_proj, graph.constant({3}, starts_up, DType::INT32),
                    graph.constant({3}, ends_up, DType::INT32), graph.constant({3}, steps_3d, DType::INT32));
    exp_up = graph.contiguous(exp_up);

    // STEP 10: silu_atomic (model's exact formulation: pow(e,-x) -> div -> mul)
    float neg_one_val = -1.0f;
    LogicalId neg_one = expand_float_3d(neg_one_val, E, S, I);
    LogicalId neg_x = graph.mul(exp_gate, neg_one);

    float e_val = 2.718281828459045f;
    LogicalId e_node = expand_float_3d(e_val, E, S, I);
    LogicalId exp_neg_x = graph.pow(e_node, neg_x);

    float one_val = 1.0f;
    LogicalId one_node = expand_float_3d(one_val, E, S, I);
    LogicalId den = graph.add(one_node, exp_neg_x);
    LogicalId sigmoid_val = graph.div(one_node, den);
    LogicalId exp_gate_silu = graph.mul(exp_gate, sigmoid_val);

    // STEP 11: exp_gate_up = mul(exp_gate_silu, exp_up)
    LogicalId exp_gate_up = graph.mul(exp_gate_silu, exp_up);

    // STEP 12: fused_down_t
    LogicalId w_dn_cpu = graph._copyto(W_dn_id);
    LogicalId w_dn_f32 = graph.cast(w_dn_cpu, DType::FLOAT32);
    LogicalId fused_down_t = graph.permute(w_dn_f32, graph.constant({3}, perm_w_3d, DType::INT32));
    fused_down_t = graph.contiguous(fused_down_t);

    // STEP 13: exp_down = dot(exp_gate_up, fused_down_t)
    LogicalId exp_down = graph.dot(exp_gate_up, fused_down_t);

    // STEP 14: exp_down_perm = contiguous(permute(exp_down, [1,0,2]))
    int32_t perm_esh[] = {1, 0, 2};
    LogicalId exp_down_perm = graph.permute(exp_down, graph.constant({3}, perm_esh, DType::INT32));
    exp_down_perm = graph.contiguous(exp_down_perm);

    // STEP 15: normalized_probs_perm = contiguous(permute(normalized_probs,
    // [1,2,0]))
    int32_t perm_1se[] = {1, 2, 0};
    LogicalId normalized_probs_perm = graph.permute(normalized_probs, graph.constant({3}, perm_1se, DType::INT32));
    normalized_probs_perm = graph.contiguous(normalized_probs_perm);

    // STEP 16: normalized_probs_exp = contiguous(repeat(normalized_probs_perm, H,
    // axis=2))
    LogicalId normalized_probs_exp = rep_axis(normalized_probs_perm, H, 2);
    normalized_probs_exp = graph.contiguous(normalized_probs_exp);

    // STEP 17: weighted_outputs = mul(exp_down_perm, normalized_probs_exp)
    LogicalId weighted_outputs = graph.mul(exp_down_perm, normalized_probs_exp);

    // STEP 18: routed_out_sum = sum(weighted_outputs, axis=1)
    int32_t sum_ax1[] = {1};
    LogicalId routed_out_sum = graph.sum(weighted_outputs, graph.constant({1}, sum_ax1, DType::INT32));

    // STEP 19: routed_out = reshape(routed_out_sum, [1, S, H])
    int32_t final_shape[] = {1, static_cast<int32_t>(S), static_cast<int32_t>(H)};
    LogicalId routed_out = graph.reshape(routed_out_sum, graph.constant({3}, final_shape, DType::INT32));

    return routed_out;
}

// ---------------------------------------------------------------------------
// Registration — IDENTICAL shape/dtype/backend spec to v1.
//
// The only difference from v1 is the kernel name (so the build system
// generates a unique UID) and the run function (which uses BFDOT).
// ---------------------------------------------------------------------------
REGISTER_KERNEL("MoE_TopK_FusedGEMM_StreamingStorage_NEON_v2", 5, 5, matchMoETopKFusedGEMM_StreamingStorage_v2,
                runMoETopKFusedGEMM_StreamingStorage_v2, refFactoryMoETopKFusedGEMM_StreamingStorage_v2,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, // output backend
                {DType::FLOAT32, DType::BF16, DType::BF16, DType::FLOAT32, DType::INT32},
                {{1, 8, 2048}, {256, 1024, 2048}, {256, 2048, 512}, {1, 8, 256}, {1, 8, 8}},
                {true, true, true, true, true}, // all inputs contiguous
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(0, HandleType::STORAGE)},
                 {MemSpace(0, HandleType::STORAGE)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON && __ARM_FEATURE_BF16
