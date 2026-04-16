// File: tensor_graphs_cpp/kernels/cpu/reference/slice/F32_ND.hpp
// TODO: make view only
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchSliceView(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return inputs.size() == 4;
}

inline void inferViewSlice(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph)
{
    auto starts = getConstantInt32(inputs[1].id, graph);
    auto steps = getConstantInt32(inputs[3].id, graph);

    node.strides.resize(inputs[0].strides.size());
    uint64_t offset = inputs[0].viewOffset;

    for (size_t i = 0; i < inputs[0].strides.size(); ++i)
    {
        int32_t start = i < starts.size() ? starts[i] : 0;
        int32_t step = i < steps.size() ? steps[i] : 1;

        if (start < 0)
            start += inputs[0].getShape()[i];

        offset += start * inputs[0].strides[i];
        node.strides[i] = inputs[0].strides[i] * step;
    }

    node.viewOffset = offset;
}

REGISTER_REF_KERNEL_VIEW(OpType::SLICE, 4, matchSliceView, inferViewSlice, {Backend::CPU, Backend::CUDA}, {DType::FLOAT32, DType::INT32, DType::INT32, DType::INT32}, {{1}, {1}, {1}, {1}}, {false, false, false, false}, {{Backend::CPU, Backend::CUDA}, {Backend::CPU}, {Backend::CPU}, {Backend::CPU}});

