#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <thread>

inline bool matchDivF32_ND_Fast(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape() == inputs[1].getShape() && isContiguous(inputs[0]) && isContiguous(inputs[1]) && isContiguous(output);
}

inline void runDivF32_ND_Fast(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *a = static_cast<const float *>(inputs[0]);
    const float *b = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);
    uint64_t n = countElements(inViews[0].getShape());

    uint32_t num_threads = std::thread::hardware_concurrency();
    uint64_t chunk = (n + num_threads - 1) / num_threads;
    std::vector<std::thread> workers;

    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint64_t start = t * chunk;
            uint64_t end = std::min(start + chunk, n);
            uint64_t i = start;
            for (; i + 4 <= end; i += 4) {
                // Note: vdivq_f32 is standard NEON
                vst1q_f32(out + i, vdivq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
            }
            for (; i < end; ++i) out[i] = a[i] / b[i]; });
    }
    for (auto &w : workers)
        w.join();
}

REGISTER_KERNEL("Div_ND_NEON_Threaded", 2, matchDivF32_ND_Fast, runDivF32_ND_Fast, nullptr, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 24, 1536, 1536}, {1, 24, 1536, 1536}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});
#endif