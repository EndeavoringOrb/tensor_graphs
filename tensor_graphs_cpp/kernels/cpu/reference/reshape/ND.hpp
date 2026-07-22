#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchReshapeView(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    return countElements(inputs[0].getShape()) == countElements(output.getShape());
}

inline void inferViewReshape(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph)
{
    node.strides = calcContiguousStrides(node.getShape());
    node.viewOffset = inputs[0].viewOffset;
}

REGISTER_REF_KERNEL_VIEW(OpType::RESHAPE, 2, 2, matchReshapeView, inferViewReshape, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::ANY, DType::INT32}, {{1}, {1}}, {true, true}, {{MemSpace(1, HandleType::CPP), MemSpace(1, HandleType::CUDA)}, {MemSpace(1, HandleType::CPP)}});
