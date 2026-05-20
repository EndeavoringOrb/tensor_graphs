#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

#if defined(TG_HAS_NEON)
#include <cstring>
#include <thread>

inline bool matchConcatF32_Fast(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runConcatF32_Fast(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    float *out_ptr = static_cast<float *>(outputs[0]);
    int32_t axis = *static_cast<const int32_t *>(inputs.back());
    const auto &out_shape = outViews[0].getShape();
    if (axis < 0)
        axis += out_shape.size();

    uint64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i)
        outer *= out_shape[i];
    for (int i = axis + 1; i < (int)out_shape.size(); ++i)
        inner *= out_shape[i];

    uint32_t num_threads = std::thread::hardware_concurrency();
    uint32_t chunk = (outer + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t)
    {
        workers.emplace_back([=]()
                             {
            uint32_t o_start = t * chunk;
            uint32_t o_end = std::min(o_start + chunk, (uint32_t)outer);
            for (uint32_t o = o_start; o < o_end; ++o) {
                uint64_t out_axis_offset = 0;
                for (size_t n = 0; n < inputs.size() - 1; ++n) {
                    uint32_t axis_dim = inViews[n].getShape()[axis];
                    const float *src = static_cast<const float *>(inputs[n]) + (o * axis_dim * inner);
                    float *dst = out_ptr + (o * out_shape[axis] * inner) + (out_axis_offset * inner);
                    std::memcpy(dst, src, axis_dim * inner * sizeof(float));
                    out_axis_offset += axis_dim;
                }
            } });
    }
    for (auto &w : workers)
        w.join();
}

inline uint32_t refFactoryConcatF32_Fast(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() < 2)
        Error::throw_err("Concat Fast requires at least 2 inputs");

    std::vector<uint32_t> tensors(inputs.begin(), inputs.end() - 1);
    uint32_t axis = inputs.back();
    return graph.concat(tensors, axis);
}

REGISTER_KERNEL("Concat_F32_Fast", 2, matchConcatF32_Fast, runConcatF32_Fast, refFactoryConcatF32_Fast, {Backend::CPU}, {DType::FLOAT32, DType::INT32}, {{1, 24, 1536, 128}, {1}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});
#endif