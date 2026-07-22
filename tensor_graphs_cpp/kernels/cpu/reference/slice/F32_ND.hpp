// File: tensor_graphs_cpp/kernels/cpu/reference/slice/F32_ND.hpp
// TODO: make view only
#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"

inline bool matchSliceView(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return true;
}

inline void inferViewSlice(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph)
{
    auto starts = graph.getConstantInt32(inputs[1].id);
    auto steps = graph.getConstantInt32(inputs[3].id);

    node.strides.resize(inputs[0].strides.size());
    uint64_t offset = inputs[0].viewOffset;

    for (uint64_t i = 0; i < inputs[0].strides.size(); ++i)
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

REGISTER_REF_KERNEL_VIEW(OpType::SLICE, 4, 4, matchSliceView, inferViewSlice, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::ANY, DType::INT32, DType::INT32, DType::INT32}, {{1}, {1}, {1}, {1}}, {false, false, false, false}, {{MemSpace(1, HandleType::CPP), MemSpace(1, HandleType::CUDA)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}, {MemSpace(1, HandleType::CPP)}});

