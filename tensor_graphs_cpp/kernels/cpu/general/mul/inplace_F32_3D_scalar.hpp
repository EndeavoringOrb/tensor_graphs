// File: tensor_graphs_cpp/kernels/cpu/general/mul/inplace_FP32_3D_scalar.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>

inline bool matchMulFP32_3D_Scalar_Inplace(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != 1)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(inputs[0]) || !isContiguous(output))
        return false;
    if (inputs[0].storageType == StorageType::PERSISTENT)
        return false;
    auto it = refCounts.find(inputs[0].id);
    if (it == refCounts.end() || it->second != 1)
        return false;
    return true;
}

inline void runMulFP32_3D_Scalar_Inplace(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                                         const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    float *data3D = static_cast<float *>(outputs[0]);
    float scalarVal = *static_cast<const float *>(inputs[1]);
    uint64_t totalElements = countElements(outViews[0].getShape());
    for (uint64_t i = 0; i < totalElements; ++i)
        data3D[i] *= scalarVal;
}

inline uint32_t refFactoryMul3D_Scalar_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Mul 3D+Scalar requires 2 inputs");

    const auto &shape3D = graph.getNode(inputs[0]).getShape();

    int32_t reshape_dims[] = {1, 1, 1};
    uint32_t out = graph.reshape(inputs[1], graph.constant({3}, reshape_dims, DType::INT32));

    for (int i = 0; i < 3; ++i)
    {
        int32_t rep = (int32_t)shape3D[i];
        int32_t axis = i;
        out = graph.repeat(out, graph.constant({1}, &rep, DType::INT32), graph.constant({1}, &axis, DType::INT32));
    }

    return graph.mul(inputs[0], out);
}

REGISTER_KERNEL_INPLACE("Mul_3D_Scalar_inplace", 2, matchMulFP32_3D_Scalar_Inplace, runMulFP32_3D_Scalar_Inplace, refFactoryMul3D_Scalar_Inplace, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 1, 1}, {1}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});