#include "kernels/fused/gemma_rms_norm/ref.hpp"
#include "kernels/fused/gemma_rms_norm/gemma_rms_norm_cuda.cuh"

#ifdef TG_USE_CUDA

inline bool matchGemmaRMSNormCUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &in = inputs[0];
    const auto &w = inputs[1];
    if (in.dtype != DType::FLOAT32 || w.dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;
    if (in.getShape().size() != 3 || w.getShape().size() != 1)
        return false;
    if (in.getShape()[2] != w.getShape()[0])
        return false;
    if (in.getShape() != output.getShape())
        return false;
    if (!isContiguous(output))
        return false;
    return true;
}

inline void runGemmaRMSNormCUDA(const KernelContext &ctx)
{
    gemma_rms_norm_cuda::runGemmaRMSNormCUDAImpl(ctx, "Gemma_RMS_Norm_CUDA");
}

REGISTER_KERNEL("Gemma_RMS_Norm_CUDA", 2, 2, matchGemmaRMSNormCUDA, runGemmaRMSNormCUDA, refFactoryGemmaRMSNorm,
                {0}, MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)},
                {DType::FLOAT32, DType::FLOAT32},
                {{1, 16, 640}, {640}}, {true, true},
                {MemSpace(2, HandleType::CUDA), MemSpace(2, HandleType::CUDA)});

#endif
