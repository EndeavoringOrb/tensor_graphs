#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>

/**
 * FUSED KERNEL: ADD FP32 3D + SCALAR (In-place Broadcasting)
 * Pattern: Input3D[b, s, d] += InputScalar[]
 */

inline bool matchAddFP32_3D_Scalar_Inplace(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2)
        return false;

    const auto &in3D = inputs[0];
    const auto &inScalar = inputs[1];

    if (in3D.dtype != DType::FLOAT32 || inScalar.dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    // Checks for a 1D tensor of size 1 (commonly treated as a scalar in this context)
    if (in3D.shape.size() != 3 || inScalar.shape.size() != 1 || inScalar.shape[0] != 1)
        return false;

    if (in3D.shape != output.shape)
        return false;

    if (in3D.view.baseOffset != output.view.baseOffset)
        return false;

    if (!in3D.view.isContiguous() || !output.view.isContiguous())
        return false;

    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;
    auto it = refCounts.find(inputs[0].id);
    if (it == refCounts.end() || it->second != 1)
        return false;

    return true;
}

inline void runAddFP32_3D_Scalar_Inplace(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    float *data3D = static_cast<float *>(outputs[0]);
    float scalarVal = *static_cast<const float *>(inputs[1]);

    uint64_t totalElements = countElements(outViews[0].shape);

    for (uint64_t i = 0; i < totalElements; ++i)
    {
        data3D[i] += scalarVal;
    }
}

/**
 * Reference Factory: Defines the pattern the planner looks for to apply this fusion.
 * Pattern: add(x_3d, repeat(repeat(repeat(reshape(x_scalar, [1,1,1]), B, 0), S, 1), D, 2))
 */
inline uint32_t refFactoryAdd3D_Scalar_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        throw std::runtime_error("Fused Add 3D+Scalar requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t idScalar = inputs[1];

    auto shape3D = graph.nodes[id3D].shape;

    // 1. Reshape Scalar -> [1, 1, 1]
    int32_t reshape_dims[] = {1, 1, 1};
    uint32_t shape_node = graph.constant({3}, reshape_dims, DType::INT32);
    uint32_t reshaped = graph.reshape(idScalar, shape_node);

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
    uint32_t repeated_s = graph.repeat(repeated_b, rep_s, ax_s);

    // 4. Repeat axis 2 (Dimension)
        int32_t d_repeats[] = {(int32_t)shape3D[2]};
        int32_t d_axis[] = {2};
        uint32_t rep_d = graph.constant({1}, d_repeats, DType::INT32);
        uint32_t ax_d = graph.constant({1}, d_axis, DType::INT32);
    uint32_t expanded = graph.repeat(repeated_s, rep_d, ax_d);

    // 5. Final Add
    return graph.add(id3D, expanded);
}

REGISTER_KERNEL_INPLACE("Add_3D_Scalar_inplace", 2, Backend::CPU, matchAddFP32_3D_Scalar_Inplace, runAddFP32_3D_Scalar_Inplace, refFactoryAdd3D_Scalar_Inplace, {DType::FLOAT32, DType::FLOAT32}, {{1, 128, 640}, {1}}, {true, true});