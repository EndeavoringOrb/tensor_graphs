#ifdef TG_USE_CUDA
#include <cuda_runtime.h>
#include <vector>

#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool match_copy_cuda_cuda(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return true;
}

inline LogicalId ref_copy_cuda_cuda(const std::vector<LogicalId> &inputs, Graph &graph)
{
    return graph._copyto(inputs[0]);
}

inline void run_copy_cuda_cuda(const KernelContext &ctx)
{
    const void *src = ctx.inputs[0];
    void *dst = ctx.outputs[0];
    const TensorView &outView = ctx.outViews[0];

    uint64_t num_elements = countElements(outView);
    uint64_t bytes = num_elements * getDTypeSize(outView.dtype);

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());

    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream);
    if (err != cudaSuccess)
    {
        Error::throw_err("cudaMemcpyAsync failed in run_copy_cuda_cuda: " + std::string(cudaGetErrorString(err)));
    }
}

REGISTER_KERNEL(
    "copy_cuda_cuda",
    1, 1,
    match_copy_cuda_cuda,
    run_copy_cuda_cuda,
    ref_copy_cuda_cuda,
    {},
    MemSpace{1, HandleType::CUDA},                                      // Destination GPU
    {Engine{1, EngineType::CUDA_DMA}, Engine{0, EngineType::CUDA_DMA}}, // Blocks destination DMA & models link occupancy
    {DType::ANY},
    {{1024}},
    {true},
    {MemSpace{0, HandleType::CUDA}} // Source GPU
);
#endif