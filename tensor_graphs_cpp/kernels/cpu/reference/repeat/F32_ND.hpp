#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "core/graph.hpp"

inline bool matchRepeatView(const std::vector<TensorNode> &inputs, const TensorNode &output)
{
    // Inputs: Data (0), Repeats (1), Axis (2)

    // Strides can only natively represent repeating a dimension if it originally had size 1.
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

inline void inferViewRepeat(TensorNode &node, const std::vector<TensorNode> &inputs, const Graph &graph)
{
    node.strides = inputs[0].strides;

    for (uint64_t d = 0; d < node.getShape().size(); ++d)
    {
        if (inputs[0].getShape()[d] != node.getShape()[d])
        {
            node.strides[d] = 0;
        }
    }
}

REGISTER_REF_KERNEL_VIEW(OpType::REPEAT, 3, 3, matchRepeatView, inferViewRepeat, MemSpace(1, HandleType::CPP), {Engine(0, EngineType::CPU)}, {DType::ANY, DType::INT32, DType::INT32}, {{1}, {1}, {1}}, {false, false, false}, {MemSpace(1, HandleType::CPP), MemSpace(1, HandleType::CPP), MemSpace(1, HandleType::CPP)});
