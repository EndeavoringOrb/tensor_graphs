#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchAddF32_3D_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Only match if both inputs and output are contiguous to allow simple linear NEON processing
    return inputs[0].getShape().size() == 3 &&
           isContiguous(output);
}

inline void runAddF32_3D_NEON(const KernelContext& ctx)
{
    const float *a = static_cast<const float *>(ctx.inputs[0]);
    const float *b = static_cast<const float *>(ctx.inputs[1]);
    float *out = static_cast<float *>(ctx.outputs[0]);
    uint64_t n = countElements(ctx.inViews[0].getShape());

    uint64_t i = 0;
    for (; i + 4 <= n; i += 4)
    {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
    // Tail loop
    for (; i < n; ++i)
        out[i] = a[i] + b[i];
}

inline LogicalId refFactoryAdd3D_NEON(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph.add(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Add_3D_NEON", 2, 2, matchAddF32_3D_NEON, runAddF32_3D_NEON, refFactoryAdd3D_NEON, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::FLOAT32, DType::FLOAT32}, {{1, 8, 2048}, {1, 8, 2048}}, {true, true}, {{MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

#endif // TG_HAS_NEON