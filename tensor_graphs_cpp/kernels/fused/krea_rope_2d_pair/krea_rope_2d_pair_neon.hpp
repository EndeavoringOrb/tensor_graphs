#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/common/thread_pool.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/krea_rope_2d_pair/ref.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

inline bool matchKreaRoPE2DPair(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &sX = inputs[0].getShape();   // [1, H, S, D]
    const auto &sCos = inputs[1].getShape(); // [1, 1, S, D/2]
    const auto &sSin = inputs[2].getShape(); // [1, 1, S, D/2]
    const auto &sO = output.getShape();      // [1, H, S, D]

    if (sX.size() != 4 || sCos.size() != 4 || sSin.size() != 4 || sO.size() != 4)
        return false;

    if (sX[3] != 2 * sCos[3] || sCos != sSin || sO != sX)
        return false;

    return isContiguous(output);
}

inline void runKreaRoPE2DPair(const KernelContext &ctx)
{
    const float *x = static_cast<const float *>(ctx.inputs[0]);
    const float *cos_table = static_cast<const float *>(ctx.inputs[1]);
    const float *sin_table = static_cast<const float *>(ctx.inputs[2]);
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

            for (uint32_t i = 0; i < half_dim; ++i)
            {
                float even = row_x[2 * i];
                float odd = row_x[2 * i + 1];
                float c = cos_s[i];
                float s_val = sin_s[i];

                row_out[2 * i] = even * c - odd * s_val;
                row_out[2 * i + 1] = even * s_val + odd * c;
            }
        }
    });
}



REGISTER_KERNEL("Krea_RoPE_2D_Pair_NEON", 3, 3, matchKreaRoPE2DPair, runKreaRoPE2DPair, refFactoryKreaRoPE2DPair, {0},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
                {{1, 48, 4224, 128}, {1, 1, 4224, 64}, {1, 1, 4224, 64}}, {true, true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});