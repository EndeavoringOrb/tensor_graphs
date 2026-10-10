#pragma once

#include "core/types.hpp"

#if defined(TG_HAS_AVX2)
#pragma GCC push_options
#pragma GCC target("avx2")
#include <immintrin.h>

#include "core/kernels.hpp"
#include "kernels/fused/neg_f32_nd/ref.hpp"

inline bool matchNegF32_12x2340x16_AVX2(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &in = inputs[0];
    return in.dtype == DType::FLOAT32 && output.dtype == DType::FLOAT32 &&
           in.getShape() == std::vector<uint32_t>{12, 2340, 16} && output.getShape() == in.getShape() &&
           in.strides.size() == 3 && in.strides[2] == 1 && isContiguous(output);
}

__attribute__((target("avx2")))
inline void runNegF32_12x2340x16_AVX2(const KernelContext &ctx)
{
    const float *input = static_cast<const float *>(ctx.inputs[0]);
    float *output = static_cast<float *>(ctx.outputs[0]);
    const auto &input_strides = ctx.inViews[0].strides;
    const __m256 zero = _mm256_setzero_ps();

    for (uint32_t batch = 0; batch < 12; ++batch)
    {
        for (uint32_t row = 0; row < 2340; ++row)
        {
            const float *src = input + batch * input_strides[0] + row * input_strides[1];
            float *dst = output + (batch * 2340 + row) * 16;
            _mm256_storeu_ps(dst, _mm256_sub_ps(zero, _mm256_loadu_ps(src)));
            _mm256_storeu_ps(dst + 8, _mm256_sub_ps(zero, _mm256_loadu_ps(src + 8)));
        }
    }
}

REGISTER_KERNEL("Neg_F32_12x2340x16_AVX2", 1, 1, matchNegF32_12x2340x16_AVX2, runNegF32_12x2340x16_AVX2,
                refFactoryNegF32_ND_CUDA, {0}, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)},
                {DType::FLOAT32}, {{12, 2340, 16}}, {true}, {{MemSpace(1, HandleType::CPP)}});

#pragma GCC pop_options
#endif
