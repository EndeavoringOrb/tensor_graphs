#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>
#include <thread>
#include <algorithm>

inline bool matchSmartConcat(const std::vector<TensorNode> &inputs, const TensorNode &output) {
    return isContiguous(output);
}

inline void runSmartConcat(const KernelContext &ctx) {
    float *out_ptr = static_cast<float *>(ctx.outputs[0]);
    int32_t axis = *static_cast<const int32_t *>(ctx.inputs.back());
    const auto &out_shape = ctx.outViews[0].getShape();
    if (axis < 0) axis += out_shape.size();

    uint64_t outer = 1, inner = 1;
    for (int i = 0; i < axis; ++i) outer *= out_shape[i];
    for (int i = axis + 1; i < (int)out_shape.size(); ++i) inner *= out_shape[i];

    uint64_t total_elements = outer * inner * out_shape[axis];
    
    auto compute = [&](uint64_t o_start, uint64_t o_end) {
        for (uint64_t o = o_start; o < o_end; ++o) {
            uint64_t out_axis_offset = 0;
            for (uint64_t n = 0; n < ctx.inputs.size() - 1; ++n) {
                uint32_t axis_dim = ctx.inViews[n].getShape()[axis];
                const float *src = static_cast<const float *>(ctx.inputs[n]) + (o * axis_dim * inner);
                float *dst = out_ptr + (o * out_shape[axis] * inner) + (out_axis_offset * inner);
                std::memcpy(dst, src, axis_dim * inner * sizeof(float));
                out_axis_offset += axis_dim;
            }
        }
    };

    if (total_elements < 262144) { 
        compute(0, outer); 
        return; 
    }

    uint32_t num_threads = std::thread::hardware_concurrency();
    uint32_t chunk = (outer + num_threads - 1) / num_threads;

    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < num_threads; ++t) {
        workers.emplace_back([=]() { compute(t * chunk, std::min((uint64_t)(t * chunk + chunk), outer)); });
    }
    for (auto &w : workers) w.join();
}

inline uint32_t refSmartConcat(const std::vector<uint32_t> &inputs, Graph &graph) {
    std::vector<uint32_t> tensors(inputs.begin(), inputs.end() - 1);
    uint32_t axis = inputs.back();
    return graph.concat(tensors, axis);
}

REGISTER_KERNEL("Smart_Concat_F32", 2, matchSmartConcat, runSmartConcat, refSmartConcat, {Backend::CPU}, {DType::FLOAT32, DType::INT32}, {{1, 32, 1, 128}, {1}}, {true, false}, {{Backend::CPU}, {Backend::CPU}});