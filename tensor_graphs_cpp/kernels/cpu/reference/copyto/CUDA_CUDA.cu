#ifdef TG_USE_CUDA
#include "core/kernels.hpp"
#include "core/types.hpp"


#include <cuda_runtime.h>


inline bool match_copy_cuda_p2p(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape() == output.getShape();
}

inline void run_copy_cuda_p2p(const KernelContext &ctx)
{
    const void *src = ctx.inputs[0];
    void *dst = ctx.outputs[0];
    const TensorView &inView = ctx.inViews[0];
    uint64_t bytes = countElements(inView.getShape()) * getDTypeSize(inView.dtype);
    if (bytes == 0)
        return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream);
    if (stream)
    {
        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream);
        if (err != cudaSuccess)
        {
            Error::throw_err("cudaMemcpyAsync failed in CUDA_CUDA P2P COPY_TO: " + std::string(cudaGetErrorString(err)));
        }
    }
    else
    {
        cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
        if (err != cudaSuccess)
        {
            Error::throw_err("cudaMemcpy failed in CUDA_CUDA P2P COPY_TO: " + std::string(cudaGetErrorString(err)));
        }
    }
}

REGISTER_REF_KERNEL(
    OpType::COPY_TO,
    1, 1,
    match_copy_cuda_p2p,
    run_copy_cuda_p2p,
    MemSpace{1, HandleType::CUDA},                  // Output: Local CUDA space 1
    {Engine{1, EngineType::CUDA_GPU}},             // Engine: Local CUDA engine 1
    {DType::ANY},                                  // DTypes
    {{1}},                                         // Dummy shapes
    {true},                                        // Requires contiguous
    {MemSpace{0, HandleType::CUDA}}                // Input: Local CUDA space 0
);
#endif