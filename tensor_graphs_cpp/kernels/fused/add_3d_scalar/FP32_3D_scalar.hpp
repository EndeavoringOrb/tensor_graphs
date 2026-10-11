#pragma once
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/add_3d_scalar/ref.hpp"

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

REGISTER_KERNEL("Add_3D_Scalar", 2, 2, matchAddFP32_3D_Scalar, runAddFP32_3D_Scalar, refFactoryAdd3DScalar, {0},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 1, 1}, {1}}, {true, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
