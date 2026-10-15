#ifdef TG_USE_CUDA

#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/permute_cuda/ref.hpp"

inline bool matchPermuteCUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    if (inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != inputs[0].getShape().size())
        return false;
    return true;
}

inline void inferViewPermuteCUDA(const std::vector<TensorNode> &inputs, TensorView &output, const Graph &graph)
{
    auto dims = graph.getConstantInt32(inputs[1].id);
    output.strides.resize(dims.size());
    for (uint64_t i = 0; i < dims.size(); ++i)
    {
        output.strides[i] = inputs[0].strides[dims[i]];
    }
}

REGISTER_KERNEL_VIEW("Permute_CUDA", 2, 2, matchPermuteCUDA, refFactoryPermuteCUDA, inferViewPermuteCUDA,
                     MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)},
                     {DType::ANY, DType::INT32}, {{1}, {1}}, {false, false},
                     {MemSpace(2, HandleType::CUDA), MemSpace(1, HandleType::CPP)});

#endif
