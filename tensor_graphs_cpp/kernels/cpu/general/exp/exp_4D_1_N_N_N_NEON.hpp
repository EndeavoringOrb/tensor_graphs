#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"
#include "core/common/thread_pool.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#endif

/**
 * FUSED KERNEL: Exp 4D (Multi-threaded)
 *
 * This kernel fuses the pattern:
 * pow(expand_scalar_to_4d(2.7182818f, ...), x) -> exp(x)
 *
 * While ARM NEON does not have a single-instruction hardware exp(),
 * this kernel provides speedup by:
 * 1. Eliminating the overhead of the repeated 'e' tensor.
 * 2. Avoiding the expensive pow(a, b) implementation in favor of std::exp(x).
 * 3. Parallelizing across all CPU cores.
 */

inline bool matchExpF32_4D_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Signature: [x] (The 'e' expansion is handled by the refFactory/E-Graph
    // mapping)
    if (inputs[0].getShape().size() != 4 || output.getShape().size() != 4)
        return false;

    if (inputs[0].getShape() != output.getShape())
        return false;

    return isContiguous(output);
}

inline void runExpF32_4D_NEON(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    uint32_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 1;

    ThreadPool::get().parallel_for(num_threads, [=](uint32_t t) {
        uint64_t chunk = (n + num_threads - 1) / num_threads;
        uint64_t start = t * chunk;
        uint64_t end = std::min(start + chunk, n);

        // Note: Standard math library exp() is typically well-optimized by
        // the compiler (cl.exe or g++) for the target architecture.
        for (uint64_t i = start; i < end; ++i)
        {
            out[i] = std::exp(in[i]);
        }
    });
}

/**
 * Reference Factory
 * This precisely mirrors the graph structure created by:
 * LogicalId exps = g.pow(expand_scalar_to_4d(2.7182818f, 1, cfg.num_heads, L_q,
 * total_seq_len), shifted);
 */
inline LogicalId refFactoryExp4D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x = inputs[0];
    auto shape = g.getNode(x).getShape();

    // 1. Create constant e
    float e_val = 2.7182818f;
    LogicalId e_node = g.constant({1}, &e_val, DType::FLOAT32);

    // 2. Reshape to 4D [1, 1, 1, 1]
    int32_t sh4[] = {1, 1, 1, 1};
    LogicalId e_4d = g.reshape(e_node, g.constant({4}, sh4, DType::INT32));

    // 3. Mirror expand_scalar_to_4d repeat logic
    LogicalId current_e = e_4d;
    for (int ax = 0; ax < 4; ++ax)
    {
        if (shape[ax] > 1)
        {
            int32_t r = (int32_t)shape[ax];
            int32_t a = ax;
            current_e = g.repeat(current_e, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
    }

    // 4. Return pow(e_expanded, x)
    return g.pow(current_e, x);
}

// Register for typical FLUX Attention score shapes
REGISTER_KERNEL("Exp_4D_NEON", 1, 1, matchExpF32_4D_NEON, runExpF32_4D_NEON, refFactoryExp4D,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32}, {{1, 24, 512, 1024}},
                {true}, {{MemSpace(1, HandleType::CPP)}});