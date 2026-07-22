#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <vector>

inline bool matchAddFP32_3D_Scalar(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != 1)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runAddFP32_3D_Scalar(const KernelContext &ctx)
{
    const float *data3D = static_cast<const float *>(ctx.inputs[0]);
    float scalarValue = *static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    uint64_t totalElements = countElements(ctx.inViews[0].getShape());
    for (uint64_t i = 0; i < totalElements; ++i)
        out[i] = data3D[i] + scalarValue;
}

inline LogicalId refFactoryAdd3D_Scalar(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Fused Add 3D+Scalar requires 2 inputs");

    LogicalId id3D = inputs[0];
    LogicalId idScalar = inputs[1];

    auto shape3D = graph.getNode(id3D).getShape();

    // 1. Reshape Scalar -> [1, 1, 1]
    int32_t reshape_dims[] = {1, 1, 1};
    LogicalId shape_node = graph.constant({3}, reshape_dims, DType::INT32);
    LogicalId reshaped = graph.reshape(idScalar, shape_node);

    // 2. Repeat axis 0 (Batch)
    int32_t b_repeats[] = {(int32_t)shape3D[0]};
    int32_t b_axis[] = {0};
    LogicalId rep_b = graph.constant({1}, b_repeats, DType::INT32);
    LogicalId ax_b = graph.constant({1}, b_axis, DType::INT32);
    LogicalId repeated_b = graph.repeat(reshaped, rep_b, ax_b);

    // 3. Repeat axis 1 (Sequence)
    int32_t s_repeats[] = {(int32_t)shape3D[1]};
    int32_t s_axis[] = {1};
    LogicalId rep_s = graph.constant({1}, s_repeats, DType::INT32);
    LogicalId ax_s = graph.constant({1}, s_axis, DType::INT32);
    LogicalId repeated_s = graph.repeat(repeated_b, rep_s, ax_s);

    // 4. Repeat axis 2 (Hidden)
    int32_t d_repeats[] = {(int32_t)shape3D[2]};
    int32_t d_axis[] = {2};
    LogicalId rep_d = graph.constant({1}, d_repeats, DType::INT32);
    LogicalId ax_d = graph.constant({1}, d_axis, DType::INT32);
    LogicalId expanded = graph.repeat(repeated_s, rep_d, ax_d);

    return graph.add(id3D, expanded);
}

REGISTER_KERNEL("Add_3D_Scalar", 2, 2, matchAddFP32_3D_Scalar, runAddFP32_3D_Scalar, refFactoryAdd3D_Scalar, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32}, {{1, 1, 1}, {1}}, {true, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
