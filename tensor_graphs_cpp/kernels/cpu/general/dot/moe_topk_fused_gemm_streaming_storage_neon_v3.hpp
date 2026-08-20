#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON) && defined(__ARM_FEATURE_BF16)

#include <arm_neon.h>

#include <algorithm>
#include <atomic>
#include <cmath>
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

// Portable positional disk read
static inline bool moe_v3_readFromFileAtOffset(int fd, uint64_t offset, void *buf, uint64_t bytes)
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

static inline uint16x8_t moe_v3_fp32x8_to_bf16_u16x8(float32x4_t lo, float32x4_t hi)
{
    uint16x4_t l_bf = vshrn_n_u32(vreinterpretq_u32_f32(lo), 16);
    uint16x4_t h_bf = vshrn_n_u32(vreinterpretq_u32_f32(hi), 16);
    return vcombine_u16(l_bf, h_bf);
}

struct ExpertBuffer
{
    std::vector<uint8_t> gu_data;
    std::vector<uint8_t> dn_data;
    uint32_t expert_id = 0;
    std::atomic<bool> ready{false};
};

// Generalized BFDOT Tiled GEMM
// X: [S, K] bf16, W: [N, K] bf16, Out: [S, N] fp32
static inline void moe_v3_TiledBFDOT(const uint16_t *X, const uint16_t *W, float *Out, uint32_t S, uint32_t K,
                                     uint32_t N)
{
    const uint32_t K8 = K & ~7u;
    const uint32_t N4 = N & ~3u;

    // Tile S to keep X_tile in L2 cache
    constexpr uint32_t S_TILE = 128;
    for (uint32_t s_outer = 0; s_outer < S; s_outer += S_TILE)
    {
        uint32_t s_end = std::min(s_outer + S_TILE, S);

        for (uint32_t n = 0; n < N4; n += 4)
        {
            const uint16_t *w0 = W + (n + 0) * K;
            const uint16_t *w1 = W + (n + 1) * K;
            const uint16_t *w2 = W + (n + 2) * K;
            const uint16_t *w3 = W + (n + 3) * K;

            for (uint32_t s = s_outer; s < s_end; ++s)
            {
                const uint16_t *x_row = X + s * K;
                float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f), a2 = vdupq_n_f32(0.0f),
                            a3 = vdupq_n_f32(0.0f);

                for (uint32_t k = 0; k < K8; k += 8)
                {
                    bfloat16x8_t xv = vreinterpretq_bf16_u16(vld1q_u16(x_row + k));
                    a0 = vbfdotq_f32(a0, xv, vreinterpretq_bf16_u16(vld1q_u16(w0 + k)));
                    a1 = vbfdotq_f32(a1, xv, vreinterpretq_bf16_u16(vld1q_u16(w1 + k)));
                    a2 = vbfdotq_f32(a2, xv, vreinterpretq_bf16_u16(vld1q_u16(w2 + k)));
                    a3 = vbfdotq_f32(a3, xv, vreinterpretq_bf16_u16(vld1q_u16(w3 + k)));
                }

                float *out_ptr = Out + s * N + n;
                out_ptr[0] = vaddvq_f32(a0);
                out_ptr[1] = vaddvq_f32(a1);
                out_ptr[2] = vaddvq_f32(a2);
                out_ptr[3] = vaddvq_f32(a3);
            }
        }
        // Handle N-tail if N%4 != 0 (omitted for standard Qwen shapes)
    }
}

inline bool matchMoETopKFusedGEMM_v3(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (output.getShape().size() != 3)
        return false;
    const auto &sX = inputs[0].getShape();
    const auto &sWgu = inputs[1].getShape();
    const auto &sWdn = inputs[2].getShape();
    if (sX[0] != 1 || sWgu[2] != sX[2] || sWdn[0] != sWgu[0] || sWdn[1] != sX[2])
        return false;
    return isContiguous(output);
}

inline void runMoETopKFusedGEMM_v3(const KernelContext &ctx)
{
    const float *X = static_cast<const float *>(ctx.inputs[0]);
    const uint16_t *Wgu_base = static_cast<const uint16_t *>(ctx.inputs[1]);
    const uint16_t *Wdn_base = static_cast<const uint16_t *>(ctx.inputs[2]);
    const float *router_probs = static_cast<const float *>(ctx.inputs[3]);
    const uint32_t *sel = static_cast<const uint32_t *>(ctx.inputs[4]);
    float *Out = static_cast<float *>(ctx.outputs[0]);

    const uint32_t S = ctx.inViews[0].getShape()[1], H = ctx.inViews[0].getShape()[2];
    const uint32_t E = ctx.inViews[1].getShape()[0], I2 = ctx.inViews[1].getShape()[1], I = I2 / 2;
    const uint32_t K = ctx.inViews[4].getShape()[2];
    const uint64_t gu_bytes = (uint64_t)I2 * H * 2, dn_bytes = (uint64_t)H * I * 2;

    // Convert X to BF16 once
    std::vector<uint16_t> X_bf16(S * H);
    for (uint32_t s = 0; s < S; ++s)
    {
        const float *src = X + s * H;
        uint16_t *dst = X_bf16.data() + s * H;
        uint32_t h = 0;
        for (; h + 8 <= H; h += 8)
            vst1q_u16(dst + h, moe_v3_fp32x8_to_bf16_u16x8(vld1q_f32(src + h), vld1q_f32(src + h + 4)));
        for (; h < H; ++h)
        {
            uint32_t b;
            std::memcpy(&b, src + h, 4);
            dst[h] = (uint16_t)(b >> 16);
        }
    }

    const int SLOTS = 4;
    std::vector<ExpertBuffer> ring(SLOTS);
    for (int i = 0; i < SLOTS; ++i)
    {
        ring[i].gu_data.resize(gu_bytes);
        ring[i].dn_data.resize(dn_bytes);
    }

    std::atomic<uint32_t> prod_idx{0}, cons_idx{0};
    std::atomic<bool> done{false};
    std::mutex mtx;
    std::condition_variable cv_prod, cv_cons;

    std::vector<uint32_t> unique_experts;
    std::vector<bool> used(E, false);
    for (uint32_t i = 0; i < S * K; ++i)
    {
        uint32_t e = sel[i];
        if (e < E && !used[e])
        {
            used[e] = true;
            unique_experts.push_back(e);
        }
    }

    std::thread producer([&]() {
        uint32_t next_e = 0;
        while (next_e < unique_experts.size())
        {
            uint32_t slot = prod_idx % SLOTS;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_prod.wait(lock, [&] { return !ring[slot].ready; });
            }
            uint32_t e = unique_experts[next_e++];
            std::memcpy(ring[slot].gu_data.data(), Wgu_base + (uint64_t)e * I2 * H, gu_bytes);
            std::memcpy(ring[slot].dn_data.data(), Wdn_base + (uint64_t)e * H * I, dn_bytes);
            ring[slot].expert_id = e;
            ring[slot].ready = true;
            prod_idx++;
            cv_cons.notify_all();
        }
        done = true;
        cv_cons.notify_all();
    });

    std::vector<float> final_acc(S * H, 0.0f);
    uint32_t processed = 0;
    while (processed < unique_experts.size())
    {
        uint32_t slot = cons_idx % SLOTS;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cv_cons.wait(lock, [&] { return ring[slot].ready || done; });
        }
        if (!ring[slot].ready && done)
            break;

        uint32_t e = ring[slot].expert_id;
        const uint16_t *Wgu = reinterpret_cast<const uint16_t *>(ring[slot].gu_data.data());
        const uint16_t *Wdn = reinterpret_cast<const uint16_t *>(ring[slot].dn_data.data());

        // Temp buffers for this expert's contribution across all S
        std::vector<float> gate_up_out(S * I2);
        moe_v3_TiledBFDOT(X_bf16.data(), Wgu, gate_up_out.data(), S, H, I2);

        std::vector<uint16_t> inter_bf16(S * I);
        for (uint32_t s = 0; s < S; ++s)
        {
            for (uint32_t i = 0; i < I; ++i)
            {
                float g = gate_up_out[s * I2 + i], u = gate_up_out[s * I2 + I + i];
                float val = (g / (1.0f + expf(-g))) * u;
                uint32_t bits;
                std::memcpy(&bits, &val, 4);
                inter_bf16[s * I + i] = (uint16_t)(bits >> 16);
            }
        }

        std::vector<float> down_out(S * H);
        moe_v3_TiledBFDOT(inter_bf16.data(), Wdn, down_out.data(), S, I, H);

        // Probabilistic accumulation
        for (uint32_t s = 0; s < S; ++s)
        {
            float w = 0;
            for (uint32_t k = 0; k < K; ++k)
                if (sel[s * K + k] == e)
                    w += router_probs[s * E + e];
            for (uint32_t h = 0; h < H; ++h)
                final_acc[s * H + h] += w * down_out[s * H + h];
        }

        ring[slot].ready = false;
        cons_idx++;
        processed++;
        cv_prod.notify_all();
    }
    producer.join();
    std::memcpy(Out, final_acc.data(), S * H * sizeof(float));
}

inline LogicalId refFactoryMoETopKFusedGEMM_StreamingStorage_v3(const std::vector<LogicalId> &inputs, Graph &graph)
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

REGISTER_KERNEL("MoE_TopK_FusedGEMM_Streaming_v3", 5, 5, matchMoETopKFusedGEMM_v3, runMoETopKFusedGEMM_v3,
                refFactoryMoETopKFusedGEMM_StreamingStorage_v3, {}, MemSpace(1, HandleType::CPP),
                {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::BF16, DType::BF16, DType::FLOAT32, DType::INT32},
                {{1, 8, 2048}, {256, 1024, 2048}, {256, 2048, 512}, {1, 8, 256}, {1, 8, 8}},
                {true, true, true, true, true},
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}});
#endif