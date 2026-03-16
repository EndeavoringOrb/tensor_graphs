#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>

/**
 * FUSED KERNEL: ADD FP32 3D + 1D (In-place Broadcasting)
 * Pattern: Input3D[b, s, d] += Input1D[d]
 *
 * This kernel performs the addition directly on the 3D input buffer.
 */

inline bool matchAddFP32_3D_1D_Inplace(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
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

    // In-place requires shape identity between input 3D and output
    if (in3D.shape != output.shape)
        return false;

    // Reference implementation assumes contiguity for the large tensors
    if (!in3D.view.isContiguous() || !in1D.view.isContiguous() || !output.view.isContiguous())
        return false;

    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;
    auto it = refCounts.find(inputs[0].id);
    if (it == refCounts.end())
        return false;
    if (it->second != 1)
        return false;

    return true;
}

inline void runAddFP32_3D_1D_Inplace(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                     const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    float *data3D = static_cast<float *>(outputs[0]); // Modifying output directly
    const float *data1D = static_cast<const float *>(inputs[1]);

    uint32_t B = outViews[0].shape[0];
    uint32_t S = outViews[0].shape[1];
    uint32_t D = outViews[0].shape[2];

    uint64_t totalElements = (uint64_t)B * S * D;

    // Optimized in-place loop
    for (uint64_t i = 0; i < totalElements; ++i)
    {
        data3D[i] += data1D[i % D];
    }
}

/**
 * Reference Factory: Defines the pattern the planner looks for to apply this fusion.
 * Pattern: add(x_3d, repeat(repeat(reshape(x_1d, [1,1,D]), B, 0), S, 1))
 */
inline uint32_t refFactoryAdd3D_1D_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_error("Fused Add 3D+1D requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t id1D = inputs[1];

    auto shape3D = graph.nodes[id3D].shape;
    auto shape1D = graph.nodes[id1D].shape;

    // 1. Reshape 1D -> [1, 1, D]
    int32_t reshape_dims[] = {1, 1, (int32_t)shape1D[0]};
    uint32_t shape_node = graph.constant({3}, reshape_dims, DType::INT32);
    uint32_t reshaped = graph.reshape(id1D, shape_node);

    // 2. Repeat axis 0 (Batch)
        int32_t b_repeats[] = {(int32_t)shape3D[0]};
        int32_t b_axis[] = {0};
        uint32_t rep_b = graph.constant({1}, b_repeats, DType::INT32);
        uint32_t ax_b = graph.constant({1}, b_axis, DType::INT32);
    uint32_t repeated_b = graph.repeat(reshaped, rep_b, ax_b);

    // 3. Repeat axis 1 (Sequence)
        int32_t s_repeats[] = {(int32_t)shape3D[1]};
        int32_t s_axis[] = {1};
        uint32_t rep_s = graph.constant({1}, s_repeats, DType::INT32);
        uint32_t ax_s = graph.constant({1}, s_axis, DType::INT32);
    uint32_t expanded = graph.repeat(repeated_b, rep_s, ax_s);

    // 4. Final Add
    return graph.add(id3D, expanded);
}

REGISTER_KERNEL_INPLACE("Add_3D_1D_inplace", 2, Backend::CPU, matchAddFP32_3D_1D_Inplace, runAddFP32_3D_1D_Inplace, refFactoryAdd3D_1D_Inplace, {DType::FLOAT32, DType::FLOAT32}, {{1, 128, 640}, {640}}, {true, true});