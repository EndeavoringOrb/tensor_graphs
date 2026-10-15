#ifdef TG_USE_CUDA

#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/reshape_cuda/ref.hpp"

inline bool matchReshapeCUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    std::vector<uint64_t> dummy_strides;
    return computeReshapeViewStrides(inputs[0].getShape(), inputs[0].strides, output.getShape(), dummy_strides);
}

inline void inferViewReshapeCUDA(const std::vector<TensorNode> &inputs, TensorView &output, const Graph &graph)
{
    if (!computeReshapeViewStrides(inputs[0].getShape(), inputs[0].strides, output.getShape(), output.strides))
    {
        Error::throw_err("Reshape_CUDA::inferViewReshapeCUDA: incompatible strides for reshape view");
    }
}

REGISTER_KERNEL_VIEW("Reshape_CUDA", 2, 2, matchReshapeCUDA, refFactoryReshapeCUDA, inferViewReshapeCUDA,
                     MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)},
                     {DType::ANY, DType::INT32}, {{1}, {1}}, {true, true},
                     {MemSpace(2, HandleType::CUDA), MemSpace(1, HandleType::CPP)});

#endif
