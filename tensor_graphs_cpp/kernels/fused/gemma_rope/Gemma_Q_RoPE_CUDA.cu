#ifdef TG_USE_CUDA

#include "kernels/fused/gemma_rope/ref.hpp"
#include "kernels/fused/gemma_rope/gemma_rope_cuda.cuh"

inline bool matchGemmaQRoPECUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &x = inputs[0];
    const auto &cos_tab = inputs[1];
    const auto &sin_tab = inputs[2];

    if (x.dtype != DType::FLOAT32 || cos_tab.dtype != DType::FLOAT32 || sin_tab.dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    const auto &x_shape = x.getShape();
    if (x_shape.size() != 3 || x_shape[0] != 4 || x_shape[2] != 256)
        return false;

    const auto &c_shape = cos_tab.getShape();
    if (c_shape.size() != 3 || c_shape[0] != 1 || c_shape[1] != x_shape[1] || c_shape[2] != 256)
        return false;

    const auto &s_shape = sin_tab.getShape();
    if (s_shape.size() != 3 || s_shape[0] != 1 || s_shape[1] != x_shape[1] || s_shape[2] != 256)
        return false;

    if (output.getShape() != x_shape)
        return false;

    return true;
}

inline void runGemmaQRoPECUDA(const KernelContext &ctx)
{
    gemma_rope_cuda::runGemmaRoPECUDABase(ctx, "Gemma_Q_RoPE_CUDA", 4, 256);
}

REGISTER_KERNEL("Gemma_Q_RoPE_CUDA", 3, 3, matchGemmaQRoPECUDA, runGemmaQRoPECUDA,
                refFactoryGemmaRoPE, {}, MemSpace(2, HandleType::CUDA),
                {Engine(0, EngineType::CUDA_GPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
                {{4, 16, 256}, {1, 16, 256}, {1, 16, 256}},
                {true, true, true},
                {MemSpace(2, HandleType::CUDA), MemSpace(2, HandleType::CUDA), MemSpace(2, HandleType::CUDA)});

#endif
