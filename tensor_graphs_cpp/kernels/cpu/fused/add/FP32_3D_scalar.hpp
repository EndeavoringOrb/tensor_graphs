#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>

/**
 * FUSED KERNEL: ADD FP32 3D + Scalar (Broadcasting)
 * Pattern: Output[b, s, d] = Input3D[b, s, d] + InputScalar[]
 */

inline bool matchAddFP32_3D_Scalar(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;

    const auto &in3D = inputs[0];
    const auto &inScalar = inputs[1];

    if (in3D.dtype != DType::FLOAT32 || inScalar.dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Rank Check: 3D + Scalar
    if (in3D.shape.size() != 3 || inScalar.shape.size() != 1 || inScalar.shape[0] != 1)
        return false;

    if (in3D.shape != output.shape)
        return false;

    if (!in3D.view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

inline void runAddFP32_3D_Scalar(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                 const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *data3D = static_cast<const float *>(inputs[0]);
    float scalarValue = *static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint64_t totalElements = countElements(inViews[0].shape);

    for (uint64_t i = 0; i < totalElements; ++i)
    {
        out[i] = data3D[i] + scalarValue;
    }
}

inline uint32_t refFactoryAdd3D_Scalar(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        throw std::runtime_error("Fused Add 3D+Scalar requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t idScalar = inputs[1];

    uint32_t reshaped = graph.reshape(idScalar, graph.constant());

    uint32_t b_rep = graph.repeat(reshaped, graph.constant(), graph.constant());
    uint32_t s_rep = graph.repeat(b_rep, graph.constant(), graph.constant());
    uint32_t d_rep = graph.repeat(s_rep, graph.constant(), graph.constant());

    return graph.add(id3D, d_rep);
}

REGISTER_FUSED_KERNEL("Add_3D_Scalar", 2, Backend::CPU, matchAddFP32_3D_Scalar, runAddFP32_3D_Scalar, refFactoryAdd3D_Scalar);