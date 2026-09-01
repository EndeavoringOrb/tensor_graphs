#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

inline bool matchKreaPerHeadRMSNormRoPE(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &sX = inputs[0].getShape();   // [1, H, S, D]
    const auto &sW = inputs[1].getShape();   // [D]
    const auto &sCos = inputs[2].getShape(); // [1, 1, S, D/2]
    const auto &sSin = inputs[3].getShape(); // [1, 1, S, D/2]
    const auto &sO = output.getShape();      // [1, H, S, D]

    if (sX.size() != 4 || sW.size() != 1 || sCos.size() != 4 || sSin.size() != 4 || sO.size() != 4)
        return false;

    if (sX[3] != sW[0] || sX[3] != 2 * sCos[3] || sCos != sSin || sO != sX)
        return false;

    return isContiguous(output);
}

inline void runKreaPerHeadRMSNormRoPE(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    const float *w = static_cast<const float *>(ctx.inputs[1]);
    const float *cos_table = static_cast<const float *>(ctx.inputs[2]);
    const float *sin_table = static_cast<const float *>(ctx.inputs[3]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &sX = ctx.inViews[0].getShape();
    const uint32_t H = sX[1];
    const uint32_t S = sX[2];
    const uint32_t D = sX[3];
    const uint32_t half_dim = D / 2;

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    uint32_t total_rows = H * S;
    num_threads = std::min(num_threads, total_rows);

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint32_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;
        uint32_t start_row = t * rows_per_thread;
        uint32_t end_row = std::min(start_row + rows_per_thread, total_rows);

        for (uint32_t r = start_row; r < end_row; ++r)
        {
            uint32_t s_idx = r % S;
            const float *row_x = x + static_cast<uint64_t>(r) * D;
            float *row_out = out + static_cast<uint64_t>(r) * D;
            const float *cos_s = cos_table + static_cast<uint64_t>(s_idx) * half_dim;
            const float *sin_s = sin_table + static_cast<uint64_t>(s_idx) * half_dim;

            // 1. RMS of head row
            float sum_sq = 0.0f;
#if defined(TG_HAS_NEON)
            float32x4_t v_acc = vdupq_n_f32(0.0f);
            for (uint32_t d = 0; d < D; d += 4)
            {
                float32x4_t vx = vld1q_f32(row_x + d);
                v_acc = vfmaq_f32(v_acc, vx, vx);
            }
            sum_sq = vaddvq_f32(v_acc);
#else
            for (uint32_t d = 0; d < D; ++d)
                sum_sq += row_x[d] * row_x[d];
#endif
            float inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(D)) + 1e-6f);

            // 2. Interleaved pair normalization & rotation
            for (uint32_t i = 0; i < half_dim; ++i)
            {
                float even = row_x[2 * i] * inv_rms * (w[2 * i] + 1.0f);
                float odd = row_x[2 * i + 1] * inv_rms * (w[2 * i + 1] + 1.0f);
                float c = cos_s[i];
                float s_val = sin_s[i];

                row_out[2 * i] = even * c - odd * s_val;
                row_out[2 * i + 1] = even * s_val + odd * c;
            }
        }
    });
}

inline LogicalId refFactoryKreaPerHeadRMSNormRoPE(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];        // [1, H, S, D]
    LogicalId w = inputs[1];        // [D]
    LogicalId cos_node = inputs[2]; // [1, 1, S, D/2]
    LogicalId sin_node = inputs[3]; // [1, 1, S, D/2]

    auto sX = g.getNode(x).getShape();
    uint32_t H = sX[1];
    uint32_t S = sX[2];
    uint32_t D = sX[3];
    uint32_t half_dim = D / 2;

    LogicalId x_sq = g.mul(x, x);
    int32_t axis_val = -1;
    LogicalId sum_sq = g.sum(x_sq, g.constant({1}, &axis_val, DType::INT32));
    LogicalId mean_sq = g.div(sum_sq, g.fill(static_cast<float>(D), {1, H, S, 1}));
    LogicalId std = g.pow(g.add(mean_sq, g.fill(1e-6f, {1, H, S, 1})), g.fill(0.5f, {1, H, S, 1}));
    LogicalId inv_std = g.repeat(g.div(g.fill(1.0f, {1, H, S, 1}), std), D, 3);
    LogicalId x_norm = g.mul(x, inv_std);

    LogicalId w_4d = g.reshape(w, {1, 1, 1, static_cast<int32_t>(D)});
    LogicalId w_exp = g.repeat(g.repeat(w_4d, H, 1), S, 2);
    LogicalId one_full = g.fill(1.0f, {1, H, S, D});
    LogicalId scale = g.add(w_exp, one_full);
    LogicalId x_scaled = g.mul(x_norm, scale);

    LogicalId x_5d = g.reshape(
        x_scaled, {1, static_cast<int32_t>(H), static_cast<int32_t>(S), static_cast<int32_t>(half_dim), 2});
    LogicalId x_even = g.contiguous(
        g.slice(x_5d, {0, 0, 0, 0, 0},
                {1, static_cast<int32_t>(H), static_cast<int32_t>(S), static_cast<int32_t>(half_dim), 1}));
    x_even = g.reshape(
        x_even, {1, static_cast<int32_t>(H), static_cast<int32_t>(S), static_cast<int32_t>(half_dim)});
    LogicalId x_odd = g.contiguous(
        g.slice(x_5d, {0, 0, 0, 0, 1},
                {1, static_cast<int32_t>(H), static_cast<int32_t>(S), static_cast<int32_t>(half_dim), 2}));
    x_odd = g.reshape(
        x_odd, {1, static_cast<int32_t>(H), static_cast<int32_t>(S), static_cast<int32_t>(half_dim)});

    LogicalId cos_exp = g.repeat(cos_node, H, 1);
    LogicalId sin_exp = g.repeat(sin_node, H, 1);

    LogicalId x_rot_even = g.add(g.mul(x_even, cos_exp), g.neg(g.mul(x_odd, sin_exp)));
    LogicalId x_rot_odd = g.add(g.mul(x_even, sin_exp), g.mul(x_odd, cos_exp));

    LogicalId e_5d = g.reshape(x_rot_even, {1, static_cast<int32_t>(H), static_cast<int32_t>(S),
                                            static_cast<int32_t>(half_dim), 1});
    LogicalId o_5d = g.reshape(x_rot_odd, {1, static_cast<int32_t>(H), static_cast<int32_t>(S),
                                           static_cast<int32_t>(half_dim), 1});
    LogicalId pair_5d = g.concat({e_5d, o_5d}, 4);
    return g.reshape(pair_5d, {1, static_cast<int32_t>(H), static_cast<int32_t>(S), static_cast<int32_t>(D)});
}

REGISTER_KERNEL("Fused_Krea_PerHead_RMSNorm_RoPE_NEON", 4, 4, matchKreaPerHeadRMSNormRoPE, runKreaPerHeadRMSNormRoPE,
                refFactoryKreaPerHeadRMSNormRoPE, {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
                {{1, 48, 4224, 128}, {128}, {1, 1, 4224, 64}, {1, 1, 4224, 64}}, {true, true, true, true},
                {{MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)},
                 {MemSpace(1, HandleType::CPP)}});