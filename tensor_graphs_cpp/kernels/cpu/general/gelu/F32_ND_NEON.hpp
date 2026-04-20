#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#if defined(TG_HAS_NEON)
#include <arm_neon.h>
#include <cmath>

inline bool matchGeluF32_3D_NEON(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape().size() == 3 && isContiguous(inputs[0]) && isContiguous(output);
}

inline void runGeluF32_3D_NEON(const std::vector<const void *> &inputs, const std::vector<void *> &outputs,
                               const std::vector<TensorView> &inViews, const std::vector<TensorView> &outViews)
{
    const float *in = static_cast<const float *>(inputs[0]);
    float *out = static_cast<float *>(outputs[0]);
    uint64_t n = countElements(inViews[0].getShape());

    for (uint64_t i = 0; i < n; ++i)
    {
        float x = in[i];
        // Approximate GeLU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = 0.79788456f * (x + 0.044715f * x3);
        float t = std::tanh(inner);
        out[i] = 0.5f * x * (1.0f + t);
    }
}

REGISTER_KERNEL("Gelu_3D_NEON", 1, matchGeluF32_3D_NEON, runGeluF32_3D_NEON, refFactoryGelu, {Backend::CPU}, {DType::FLOAT32}, {{1, 8, 2048}}, {true}, {{Backend::CPU}});

#endif // TG_HAS_NEON