#ifdef TG_USE_CUDA
#include "kernels/fused/copy/ref.hpp"
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include <cuda_runtime.h>

inline bool matchCopyCudaSameDevice(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs[0].getShape() == output.getShape() && inputs[0].dtype == output.dtype && isContiguous(output);
}

inline void runCopyCudaSameDevice(const KernelContext &ctx)
{
    auto bytes = countElements(ctx.outViews[0]) * getDTypeSize(ctx.outViews[0].dtype);
    auto stream = reinterpret_cast<cudaStream_t>(ctx.cuda_stream());
    auto result = cudaMemcpyAsync(ctx.outputs[0], ctx.inputs[0], bytes, cudaMemcpyDeviceToDevice, stream);
    if (result != cudaSuccess)
        Error::throw_err("Same-device CUDA copy failed: " + std::string(cudaGetErrorString(result)));
}

REGISTER_KERNEL("Copy_Cuda_SameDevice", 1, 1, matchCopyCudaSameDevice, runCopyCudaSameDevice,
                ref_copy_cuda_cuda, {}, MemSpace{0, HandleType::CUDA},
                {Engine{0, EngineType::CUDA_DMA}}, {DType::ANY}, {{1024}}, {true},
                {MemSpace{0, HandleType::CUDA}});
#endif
