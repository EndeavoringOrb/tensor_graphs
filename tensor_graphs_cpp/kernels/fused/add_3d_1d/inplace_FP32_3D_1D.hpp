#pragma once
#include <vector>

#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/add_3d_1d/ref.hpp"

inline bool matchAddFP32_3D_1D_Inplace(const std::vector<TensorNode> &inputs, const TensorNode &output)
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

inline void runAddFP32_3D_1D_Inplace(const KernelContext &ctx)
{
    const float *data_3d = static_cast<const float *>(ctx.inputs[0]);
    float *output = static_cast<float *>(ctx.outputs[0]);
    const float *data_1d = static_cast<const float *>(ctx.inputs[1]);

    uint32_t batch_size = ctx.outViews[0].getShape()[0];
    uint32_t sequence_length = ctx.outViews[0].getShape()[1];
    uint32_t depth = ctx.outViews[0].getShape()[2];
    uint64_t total_elements = (uint64_t)batch_size * sequence_length * depth;

    for (uint64_t i = 0; i < total_elements; ++i)
        output[i] = data_3d[i] + data_1d[i % depth];
}

REGISTER_KERNEL("Add_3D_1D_inplace", 2, 2, matchAddFP32_3D_1D_Inplace, runAddFP32_3D_1D_Inplace,
                refFactoryAdd3D1D, {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32, DType::FLOAT32}, {{1, 1, 640}, {640}}, {true, true},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
