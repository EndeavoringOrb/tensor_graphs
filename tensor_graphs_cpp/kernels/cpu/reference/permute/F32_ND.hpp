#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "core/graph.hpp"

inline bool matchPermuteView(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Inputs: Data (0), Permutation Indices (1)
    if (inputs.size() != 2)
        return false;

    if (inputs[1].dtype != DType::INT32)
        return false;

    // Check permutation tensor shape matches data rank
    if (inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != inputs[0].getShape().size())
        return false;

    return true;
}

inline void inferViewPermute(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph)
{
    auto dims = getConstantInt32(inputs[1].id, graph);

    node.strides.resize(dims.size());
    for (size_t i = 0; i < dims.size(); ++i)
    {
        node.strides[i] = inputs[0].strides[dims[i]];
    }
    node.viewOffset = inputs[0].viewOffset;
}

REGISTER_REF_KERNEL_VIEW(OpType::PERMUTE, 2, matchPermuteView, inferViewPermute, {Backend::CPU, Backend::CUDA}, {DType::FLOAT32, DType::INT32}, {{1}, {1}}, {false, false}, {{Backend::CPU, Backend::CUDA}, {Backend::CPU}});
