#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include <cstring>

inline bool matchReshapeView(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    if (inputs.size() != 2)
        return false;
    return isContiguous(inputs[0]) && countElements(inputs[0].getShape()) == countElements(output.getShape());
}

inline void inferViewReshape(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph)
{
    node.strides = calcContiguousStrides(node.getShape());
    node.viewOffset = inputs[0].viewOffset;
}

REGISTER_REF_KERNEL_VIEW(OpType::RESHAPE, 2, matchReshapeView, inferViewReshape, {Backend::CPU, Backend::CUDA}, {DType::FLOAT32, DType::INT32}, {{1}, {1}}, {true, false});