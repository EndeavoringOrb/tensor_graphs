#pragma once

#include "core/kernels.hpp"
#include "core/shape_propagator.hpp"

inline LogicalId refFactoryFusedMXFP4_CUDA(const std::vector<LogicalId> &inputs, Graph &graph) {
    // inputs[0]: raw packed weight [out_d, in_d / 2]
    // inputs[1]: raw scale [out_d, in_d / 32]
    LogicalId packed = graph.cast(inputs[0], DType::E2M1_PACKED_INT8);
    LogicalId unpacked = graph.unpack(packed, DType::E2M1);
    LogicalId unpacked_f32 = graph.cast(unpacked, DType::FLOAT32);

    LogicalId scale_f32 = graph.cast(inputs[1], DType::FLOAT32);

    // Infer shapes for intermediate nodes before accessing getShape()
    ShapePropagator propagator;
    propagator.inferShapeRecursive(unpacked_f32, graph);

    auto out_shape = graph.getNode(unpacked_f32).getShape();
    uint32_t out_d = out_shape[0];
    uint32_t in_d  = out_shape[1];
    uint32_t scale_w = in_d / 32;

    int32_t sh3_scale[] = {(int32_t)out_d, (int32_t)scale_w, 1};
    LogicalId scale_reshaped = graph.reshape(scale_f32, graph.constant({3}, sh3_scale, DType::INT32));
    
    int32_t rep32[] = {32};
    int32_t ax2[] = {2};
    LogicalId scale_repeated = graph.repeat(scale_reshaped, graph.constant({1}, rep32, DType::INT32), graph.constant({1}, ax2, DType::INT32));
    
    int32_t sh2_final[] = {(int32_t)out_d, (int32_t)in_d};
    LogicalId scale_final = graph.reshape(scale_repeated, graph.constant({2}, sh2_final, DType::INT32));

    return graph.mul(unpacked_f32, scale_final);
}
