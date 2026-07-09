#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchMulF32_1D_NEON_Inplace(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != 1 || inputs[1].getShape().size() != 1 || output.getShape().size() != 1)
        return false;
    if (inputs[0].getShape() != inputs[1].getShape() || inputs[0].getShape() != output.getShape())
        return false;

    return isContiguous(output);
}

inline void runMulF32_1D_NEON_Inplace(const KernelContext &ctx)
{
    float *out = static_cast<float *>(ctx.outputs[0]);
    const float *b = static_cast<const float *>(ctx.inputs[1]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    uint64_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t va = vld1q_f32(out + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
    // Tail loop
    for (; i < n; ++i)
    {
        out[i] *= b[i];
    }
}

inline uint32_t refFactoryMul1D_NEON_Inplace(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.mul(inputs[0], inputs[1]);
}

REGISTER_KERNEL_INPLACE(
    "Mul_1D_NEON_inplace",
    2,
    matchMulF32_1D_NEON_Inplace,
    runMulF32_1D_NEON_Inplace,
    refFactoryMul1D_NEON_Inplace,
    {Backend::CPU},
    {DType::FLOAT32, DType::FLOAT32},
    {{2048}, {2048}},
    {true, true},
    {{Backend::CPU}, {Backend::CPU}});

#endif // TG_HAS_NEON