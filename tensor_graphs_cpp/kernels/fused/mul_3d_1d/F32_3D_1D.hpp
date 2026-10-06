#pragma once
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"

#include "kernels/fused/mul_3d_1d/ref.hpp"
inline bool matchMulFP32_3D_1D(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 3 || inputs[1].getShape().size() != 1 || output.getShape().size() != 3)
        return false;
    if (inputs[0].getShape()[2] != inputs[1].getShape()[0] || output.getShape()[2] != inputs[1].getShape()[0])
        return false;
    if (inputs[0].getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runMulFP32_3D_1D(const KernelContext &ctx)
{
    const float *data3D = static_cast<const float *>(ctx.inputs[0]);
    const float *data1D = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    uint32_t B = ctx.inViews[0].getShape()[0];
    uint32_t S = ctx.inViews[0].getShape()[1];
    uint32_t D = ctx.inViews[0].getShape()[2];
    uint64_t totalElements = (uint64_t)B * S * D;

    for (uint64_t i = 0; i < totalElements; ++i)
        out[i] = data3D[i] * data1D[i % D];
}



REGISTER_KERNEL("Mul_3D_1D", 2, 2, matchMulFP32_3D_1D, runMulFP32_3D_1D, refFactoryMul3D_1D, {0},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 1, 640}, {640}}, {true, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
