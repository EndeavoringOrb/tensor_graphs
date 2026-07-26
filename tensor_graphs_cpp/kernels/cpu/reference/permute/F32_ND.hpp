#pragma once
#include "core/graph.hpp"
#include "core/kernels.hpp"
#include "core/types.hpp"

inline bool matchPermuteView(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Inputs: Data (0), Permutation Indices (1)

    // Check permutation tensor shape matches data rank
    if (inputs[1].getShape().size() != 1 || inputs[1].getShape()[0] != inputs[0].getShape().size())
        return false;

    return true;
}

inline void inferViewPermute(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph)
{
    auto dims = graph.getConstantInt32(inputs[1].id);

    node.strides.resize(dims.size());
    for (uint64_t i = 0; i < dims.size(); ++i)
    {
        node.strides[i] = inputs[0].strides[dims[i]];
    }
}

REGISTER_REF_KERNEL_VIEW(OpType::PERMUTE, 2, 2, matchPermuteView, inferViewPermute, MemSpace(1, HandleType::CPP),
                         {Engine(0, EngineType::CPU)}, {DType::ANY, DType::INT32}, {{1}, {1}}, {false, false},
                         {MemSpace(1, HandleType::CPP), MemSpace(1, HandleType::CPP)});
