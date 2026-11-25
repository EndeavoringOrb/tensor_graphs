#ifdef TG_USE_CUDA

#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"
#include "kernels/fused/slice_cuda/ref.hpp"

inline bool matchSliceCUDA(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return true;
}

inline void inferViewSliceCUDA(const std::vector<TensorNode> &inputs, TensorView &output, const Graph &graph)
{
    auto starts = graph.getConstantInt32(inputs[1].id);
    auto steps = graph.getConstantInt32(inputs[3].id);

    output.strides.resize(inputs[0].strides.size());

    for (uint64_t i = 0; i < inputs[0].strides.size(); ++i)
    {
        int32_t start = i < starts.size() ? starts[i] : 0;
        int32_t step = i < steps.size() ? steps[i] : 1;

        if (start < 0)
            start += inputs[0].getShape()[i];

        output.strides[i] = inputs[0].strides[i] * step;
        output.offset += start * inputs[0].strides[i] * getDTypeSize(output.dtype);
    }
}

REGISTER_KERNEL_VIEW("Slice_CUDA", 4, 4, matchSliceCUDA, refFactorySliceCUDA, inferViewSliceCUDA,
                     MemSpace(2, HandleType::CUDA), {Engine(0, EngineType::CUDA_GPU)},
                     {DType::ANY, DType::INT32, DType::INT32, DType::INT32}, {{1}, {1}, {1}, {1}},
                     {false, false, false, false},
                     {MemSpace(2, HandleType::CUDA), MemSpace(1, HandleType::CPP), MemSpace(1, HandleType::CPP),
                      MemSpace(1, HandleType::CPP)});

#endif
