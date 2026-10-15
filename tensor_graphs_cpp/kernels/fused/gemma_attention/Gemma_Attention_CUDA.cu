#ifdef TG_USE_CUDA

#include "kernels/fused/gemma_attention/ref.hpp"
#include "kernels/fused/gemma_attention/gemma_attention_cuda.cuh"

inline bool matchGemmaAttentionCUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    const auto &q = inputs[0];
    const auto &k = inputs[1];
    const auto &v = inputs[2];
    const auto &mask = inputs[3];

    if (q.dtype != DType::FLOAT32 || k.dtype != DType::FLOAT32 || v.dtype != DType::FLOAT32 ||
        mask.dtype != DType::FLOAT32 || output.dtype != DType::FLOAT32)
        return false;

    const auto &q_sh = q.getShape();
    if (q_sh.size() != 3 || q_sh[0] != 4 || q_sh[2] != 256)
        return false;
    uint32_t seq_len = q_sh[1];

    if (k.getShape() != q_sh || v.getShape() != q_sh)
        return false;

    const auto &m_sh = mask.getShape();
    if (m_sh.size() != 3 || m_sh[0] != 1 || m_sh[1] != seq_len || m_sh[2] != seq_len)
        return false;

    const auto &out_sh = output.getShape();
    if (out_sh.size() != 3 || out_sh[0] != 1 || out_sh[1] != seq_len || out_sh[2] != 1024)
        return false;

    return true;
}

inline void runGemmaAttentionCUDA(const KernelContext &ctx)
{
    gemma_attention_cuda::runGemmaAttentionCUDAImpl(ctx);
}

REGISTER_KERNEL("Gemma_Attention_CUDA", 4, 4, matchGemmaAttentionCUDA, runGemmaAttentionCUDA,
                refFactoryGemmaAttention, {}, MemSpace(2, HandleType::CUDA),
                {Engine(0, EngineType::CUDA_GPU)},
                {DType::FLOAT32, DType::FLOAT32, DType::FLOAT32, DType::FLOAT32},
                {{4, 16, 256}, {4, 16, 256}, {4, 16, 256}, {1, 16, 16}},
                {true, true, true, true},
                {MemSpace(2, HandleType::CUDA), MemSpace(2, HandleType::CUDA), MemSpace(2, HandleType::CUDA), MemSpace(2, HandleType::CUDA)});

#endif
