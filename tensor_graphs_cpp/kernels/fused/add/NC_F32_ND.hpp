#pragma once
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/add/ref.hpp"
inline bool matchAddNC_F32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;
    return true;
}

inline void runAddNC_F32_ND(const KernelContext &ctx)
{
    const float *a = static_cast<const float *>(ctx.inputs[0]);
    const float *b = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t numElements = countElements(ctx.inViews[0].getShape());

    for (uint64_t i = 0; i < numElements; ++i)
    {
        out[getStridedIndex(i, ctx.outViews[0].getShape(), ctx.outViews[0].strides)] =
            a[getStridedIndex(i, ctx.inViews[0].getShape(), ctx.inViews[0].strides)] +
            b[getStridedIndex(i, ctx.inViews[1].getShape(), ctx.inViews[1].strides)];
    }
}

REGISTER_KERNEL("Add_NC_F32_ND", 2, 2, matchAddNC_F32_ND, runAddNC_F32_ND, refFactoryAdd, {0, 1},
                MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32},
                {{1, 1, 1}, {1, 1, 1}}, {false, false},
                {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
