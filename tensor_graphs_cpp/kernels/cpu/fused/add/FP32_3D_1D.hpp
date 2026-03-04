#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>

/**
 * FUSED KERNEL: ADD FP32 3D + 1D (Broadcasting)
 * Pattern: Output[b, s, d] = Input3D[b, s, d] + Input1D[d]
 *
 * This kernel replaces the sequence: RESHAPE(1D) -> REPEAT(axis 0) -> REPEAT(axis 1) -> ADD
 */

inline bool matchAddFP32_3D_1D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;

    const auto &in3D = inputs[0];
    const auto &in1D = inputs[1];

    // Check Dtypes
    if (in3D.dtype != DType::FLOAT32 || in1D.dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Check Ranks
    if (in3D.shape.size() != 3 || in1D.shape.size() != 1 || output.shape.size() != 3)
        return false;

    // The last dimension (D) must match
    if (in3D.shape[2] != in1D.shape[0] || output.shape[2] != in1D.shape[0])
        return false;

    // Output must match 3D input shape
    if (in3D.shape != output.shape)
        return false;

    // Reference implementation assumes contiguity for the large tensors
    if (!in3D.view.isContiguous() || !in1D.view.isContiguous() || !output.view.isContiguous())
        return false;

    return true;
}

inline void runAddFP32_3D_1D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                             const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *data3D = static_cast<const float *>(inputs[0]);
    const float *data1D = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint32_t B = inViews[0].shape[0];
    uint32_t S = inViews[0].shape[1];
    uint32_t D = inViews[0].shape[2];

    uint64_t totalElements = (uint64_t)B * S * D;

    // Optimized loop: The 1D data is accessed cyclically
    for (uint64_t i = 0; i < totalElements; ++i)
    {
        out[i] = data3D[i] + data1D[i % D];
    }
}

/**
 * Reference Factory: Defines the pattern the planner looks for to apply this fusion.
 * Pattern: add(x_3d, repeat(repeat(reshape(x_1d, [1,1,D]), B, 0), S, 1))
 */
inline uint32_t refFactoryAdd3D_1D(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        throw std::runtime_error("Fused Add 3D+1D requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t id1D = inputs[1];

    // 1. Reshape 1D -> [1, 1, D]
    uint32_t shape_node = graph.constant();
    uint32_t reshaped = graph.reshape(id1D, shape_node);

    // 2. Repeat axis 0 (Batch)
    uint32_t repeated_b = graph.repeat(reshaped,
                                       graph.constant(),
                                       graph.constant());

    // 3. Repeat axis 1 (Sequence)
    uint32_t expanded = graph.repeat(repeated_b,
                                     graph.constant(),
                                     graph.constant());

    // 4. Final Add
    return graph.add(id3D, expanded);
}

REGISTER_FUSED_KERNEL("Add_3D_1D", 2, Backend::CPU, matchAddFP32_3D_1D, runAddFP32_3D_1D, refFactoryAdd3D_1D);