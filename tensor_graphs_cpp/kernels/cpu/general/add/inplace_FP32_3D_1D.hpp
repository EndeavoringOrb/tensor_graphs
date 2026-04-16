// File: tensor_graphs_cpp/kernels/cpu/general/add/inplace_FP32_3D_1D.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>

inline bool matchAddFP32_3D_1D_Inplace(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1 || output.getShape().size() != 3)
        return false;
    if (inputs[0].getShape()[2] != inputs[1].getShape()[0] || output.getShape()[2] != inputs[1].getShape()[0])
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;
    return true;
}

inline void runAddFP32_3D_1D_Inplace(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                     const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    float *data3D = static_cast<float *>(outputs[0]);
    const float *data1D = static_cast<const float *>(inputs[1]);

    uint32_t B = outViews[0].getShape()[0];
    uint32_t S = outViews[0].getShape()[1];
    uint32_t D = outViews[0].getShape()[2];
    uint64_t totalElements = (uint64_t)B * S * D;

    for (uint64_t i = 0; i < totalElements; ++i)
        data3D[i] += data1D[i % D];
}

inline uint32_t refFactoryAdd3D_1D_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Add 3D+1D requires 2 inputs");

    uint32_t id3D = inputs[0];
    uint32_t id1D = inputs[1];

    auto shape3D = graph.getNode(id3D).getShape();
    auto shape1D = graph.getNode(id1D).getShape();

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

REGISTER_KERNEL_INPLACE("Add_3D_1D_inplace", 2, matchAddFP32_3D_1D_Inplace, runAddFP32_3D_1D_Inplace, refFactoryAdd3D_1D_Inplace, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 1, 640}, {640}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});