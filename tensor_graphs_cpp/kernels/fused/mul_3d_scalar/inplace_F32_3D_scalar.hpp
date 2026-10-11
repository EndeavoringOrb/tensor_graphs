#pragma once
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/mul_3d_scalar/ref.hpp"
inline bool matchMulFP32_3D_Scalar_Inplace(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != 1)
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runMulFP32_3D_Scalar_Inplace(const KernelContext &ctx)
{
    float *data3D = static_cast<float *>(ctx.outputs[0]);
    float scalarVal = *static_cast<const float *>(ctx.inputs[1]);
    uint64_t totalElements = countElements(ctx.outViews[0].getShape());
    for (uint64_t i = 0; i < totalElements; ++i)
        data3D[i] *= scalarVal;
}



REGISTER_KERNEL("Mul_3D_Scalar_inplace", 2, 2, matchMulFP32_3D_Scalar_Inplace, runMulFP32_3D_Scalar_Inplace,
                refFactoryMul3D_Scalar, {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32}, {{1, 1, 1}, {1}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});