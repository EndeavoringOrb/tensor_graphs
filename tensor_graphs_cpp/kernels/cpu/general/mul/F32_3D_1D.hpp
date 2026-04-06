// File: tensor_graphs_cpp/kernels/cpu/general/mul/FP32_3D_1D.hpp
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>

inline bool matchMulFP32_3D_1D(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1 || output.getShape().size() != 3)
        return false;
    if (inputs[0].getShape()[2] != inputs[1].getShape()[0] || output.getShape()[2] != inputs[1].getShape()[0])
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(inputs[0]) || !isContiguous(inputs[1]) || !isContiguous(output))
        return false;
    return true;
}

inline void runMulFP32_3D_1D(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                             const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *data3D = static_cast<const float *>(inputs[0]);
    const float *data1D = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);

    uint32_t B = inViews[0].getShape()[0];
    uint32_t S = inViews[0].getShape()[1];
    uint32_t D = inViews[0].getShape()[2];
    uint64_t totalElements = (uint64_t)B * S * D;

    for (uint64_t i = 0; i < totalElements; ++i)
        out[i] = data3D[i] * data1D[i % D];
}

inline uint32_t refFactoryMul3D_1D(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Mul 3D+1D requires 2 inputs");

    int32_t reshape_dims[] = {1, 1, 1}; // Dummy initialization, match doesn't care
    uint32_t out = graph.reshape(inputs[1], graph.constant({3}, reshape_dims, DType::INT32));

    int32_t rep = 1;
    int32_t b_axis = 0;
    int32_t s_axis = 1;
    uint32_t rN = graph.constant({1}, &rep, DType::INT32);
    out = graph.repeat(out, rN, graph.constant({1}, &b_axis, DType::INT32));
    out = graph.repeat(out, rN, graph.constant({1}, &s_axis, DType::INT32));

    return graph.mul(inputs[0], out);
}

REGISTER_KERNEL("Mul_3D_1D", 2, matchMulFP32_3D_1D, runMulFP32_3D_1D, refFactoryMul3D_1D, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 1, 640}, {640}}, {true, true});