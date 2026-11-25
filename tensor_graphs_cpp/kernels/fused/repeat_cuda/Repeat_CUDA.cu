#ifdef TG_USE_CUDA

#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/repeat_cuda/ref.hpp"

inline bool matchRepeatCUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[0].getShape().size() != output.getShape().size())
        return false;
    for (uint64_t d = 0; d < inputs[0].getShape().size(); ++d)
    {
        if (inputs[0].getShape()[d] != output.getShape()[d])
        {
            if (inputs[0].getShape()[d] != 1)
                return false;
        }
    }
    return true;
}

inline void inferViewRepeatCUDA(const std::vector<TensorNode> &inputs, TensorView &output, const Graph &graph)
{
    if (inputs[0].getShape().size() != output.getShape().size())
    {
        Error::throw_err("Repeat_CUDA::inferViewRepeatCUDA: input and output rank mismatch");
    }
    output.strides = inputs[0].strides;
    for (uint64_t d = 0; d < output.getShape().size(); ++d)
    {
        if (inputs[0].getShape()[d] != output.getShape()[d])
        {
            output.strides[d] = 0;
        }
    }
}

REGISTER_KERNEL_VIEW("Repeat_CUDA", 3, 3, matchRepeatCUDA, refFactoryRepeatCUDA, inferViewRepeatCUDA,
                     MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)},
                     {DType::ANY, DType::INT32, DType::INT32}, {{1}, {1}, {1}}, {false, false, false},
                     {MemSpace(2, HandleType::CUDA), MemSpace(1, HandleType::CPP), MemSpace(1, HandleType::CPP)});

#endif
