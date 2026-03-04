#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>

/**
 * FUSED KERNEL: ADD FP32 3D + 0D (Broadcasting)
 * Pattern: Output[b, s, d] = Input3D[b, s, d] + Input0D[]
 */

inline bool matchAddFP32_3D_0D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;

    const auto &in3D = inputs[0];
    const auto &in0D = inputs[1];

    if (in3D.dtype != DType::FLOAT32 || in0D.dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Rank Check: 3D + Scalar
    if (in3D.shape.size() != 3 || in0D.shape.size() != 0)
        return false;

    if (in3D.shape != output.shape)
        return false;

    if (!in3D.view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

inline void runAddFP32_3D_0D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                             const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *data3D = static_cast<const float *>(inputs[0]);
    float scalar = *static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint64_t totalElements = countElements(inViews[0].shape);

    for (uint64_t i = 0; i < totalElements; ++i)
    {
        out[i] = data3D[i] + scalar;
    }
}

inline uint32_t refFactoryAdd3D_0D(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        throw std::runtime_error("Fused Add 3D+0D requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t id0D = inputs[1];

    uint32_t reshaped = graph.reshape(id0D, graph.constant());

    uint32_t b_rep = graph.repeat(reshaped, graph.constant(), graph.constant());
    uint32_t s_rep = graph.repeat(b_rep, graph.constant(), graph.constant());
    uint32_t d_rep = graph.repeat(s_rep, graph.constant(), graph.constant());

    return graph.add(id3D, d_rep);
}

REGISTER_FUSED_KERNEL("Add_3D_0D", 2, Backend::CPU, matchAddFP32_3D_0D, runAddFP32_3D_0D, refFactoryAdd3D_0D);