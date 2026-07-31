#pragma once
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchMulFP32_3D_Scalar(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != 1)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runMulFP32_3D_Scalar(const KernelContext &ctx)
{
    const float *data3D = static_cast<const float *>(ctx.inputs[0]);
    float scalarValue = *static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    uint64_t totalElements = countElements(ctx.inViews[0].getShape());
    for (uint64_t i = 0; i < totalElements; ++i)
        out[i] = data3D[i] * scalarValue;
}

inline LogicalId refFactoryMul3D_Scalar(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Mul 3D+Scalar requires 2 inputs");

    const auto &shape3D = graph.getNode(inputs[0]).getShape();

    // 1. Reshape [1] -> [1, 1, 1]
    int32_t reshape_dims[] = {1, 1, 1};
    LogicalId out = graph.reshape(inputs[1], graph.constant({3}, reshape_dims, DType::INT32));

    // 2. Repeat for B, S, and D
    for (int i = 0; i < 3; ++i)
    {
        int32_t rep = (int32_t)shape3D[i];
        int32_t axis = i;
        out = graph.repeat(out, graph.constant({1}, &rep, DType::INT32), graph.constant({1}, &axis, DType::INT32));
    }

    return graph.mul(inputs[0], out);
}

REGISTER_KERNEL("Mul_3D_Scalar", 2, 2, matchMulFP32_3D_Scalar, runMulFP32_3D_Scalar, refFactoryMul3D_Scalar,
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 1, 1}, {1}}, {true, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
