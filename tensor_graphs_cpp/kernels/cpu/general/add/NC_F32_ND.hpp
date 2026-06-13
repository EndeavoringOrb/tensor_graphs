#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

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

inline uint32_t refFactoryAddNC_F32_ND(const std::vector<uint32_t> &inputs, Graph &graph)
{
    if (inputs.size() != 2)
        Error::throw_err("Add Non-Contiguous F32 ND requires 2 inputs");

    uint32_t id0 = inputs[0];
    uint32_t id1 = inputs[1];
    return graph.add(id0, id1);
}

REGISTER_KERNEL("Add_NC_F32_ND", 2, matchAddNC_F32_ND, runAddNC_F32_ND, refFactoryAddNC_F32_ND, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 1, 1}, {1, 1, 1}}, {false, false}, {{Backend::CPU}, {Backend::CPU}});
