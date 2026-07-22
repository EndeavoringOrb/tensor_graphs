#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchSumF32_ND(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return output.dtype == DType::FLOAT32;
}

inline void runSumF32_ND(const KernelContext &ctx)
{
    const float *in = static_cast<const float *>(ctx.inputs[0]);
    int32_t axis = *static_cast<const int32_t *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);

    const auto &inShape = ctx.inViews[0].getShape();
    const auto &outShape = ctx.outViews[0].getShape();
    int ndim = static_cast<int>(inShape.size());
    if (axis < 0)
        axis += ndim;

    // Initialize output with 0 safely
    uint64_t out_count = countElements(outShape);
    for (uint64_t i = 0; i < out_count; ++i)
    {
        out[getStridedIndex(i, outShape, ctx.outViews[0].strides)] = 0.0f;
    }

    uint64_t in_count = countElements(inShape);
    for (uint64_t i = 0; i < in_count; ++i)
    {
        uint64_t temp = i;
        uint64_t out_phys_idx = 0;

        // Map input flat index 'i' to output physical offset
        for (int d = ndim - 1; d >= 0; --d)
        {
            uint32_t coord = temp % inShape[d];
            temp /= inShape[d];
            // If d is the reduction axis, it contributes to output coord 0 (since dim is 1)
            uint32_t out_coord = (d == axis) ? 0 : coord;
            out_phys_idx += (uint64_t)out_coord * ctx.outViews[0].strides[d];
        }

        out[out_phys_idx] += in[getStridedIndex(i, inShape, ctx.inViews[0].strides)];
    }
}

REGISTER_REF_KERNEL(OpType::SUM, 2, 2, matchSumF32_ND, runSumF32_ND, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::INT32}, {{8, 32}, {1}}, {false, false}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});
