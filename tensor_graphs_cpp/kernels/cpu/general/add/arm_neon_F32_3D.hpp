#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>

inline bool matchAddF32_3D_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs.size() != 2)
        return false;
    if (inputs[0].dtype != DType::FLOAT32 || inputs[1].dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    // Only match if both inputs and output are contiguous to allow simple linear NEON processing
    return inputs[0].getShape().size() == 3 &&
           isContiguous(inputs[0]) &&
           isContiguous(inputs[1]) &&
           isContiguous(output);
}

inline void runAddF32_3D_NEON(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                              const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *a = static_cast<const float *>(inputs[0]);
    const float *b = static_cast<const float *>(inputs[1]);
    float *out = static_cast<float *>(outputs[0]);
    uint64_t n = countElements(inViews[0].getShape());

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

inline uint32_t refFactoryAdd3D_NEON(const std::vector<uint32_t> &inputs, Graph &graph)
{
    return graph.add(inputs[0], inputs[1]);
}

REGISTER_KERNEL("Add_3D_NEON", 2, matchAddF32_3D_NEON, runAddF32_3D_NEON, refFactoryAdd3D_NEON, {Backend::CPU}, {DType::FLOAT32, DType::FLOAT32}, {{1, 8, 2048}, {1, 8, 2048}}, {true, true}, {{Backend::CPU}, {Backend::CPU}});

#endif // TG_HAS_NEON