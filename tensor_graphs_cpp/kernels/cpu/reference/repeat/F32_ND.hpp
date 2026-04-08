#pragma once
#include "core/types.hpp"
#include "core/kernels.hpp"
#include "core/graph.hpp"

inline bool matchRepeatView(const std::vector<TensorNode> &inputs, const TensorNode &output, const std::unordered_map<uint32_t, uint32_t> &refCounts)
{
    // Inputs: Data (0), Repeats (1), Axis (2)
    if (inputs.size() != 3)
        return false;

    if (inputs[1].dtype != DType::INT32 || inputs[2].dtype != DType::INT32)
        return false;

    // Strides can only natively represent repeating a dimension if it originally had size 1.
    for (size_t d = 0; d < inputs[0].getShape().size(); ++d)
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

    for (size_t d = 0; d < node.getShape().size(); ++d)
    {
        if (inputs[0].getShape()[d] != node.getShape()[d])
        {
            node.strides[d] = 0;
        }
    }
    node.viewOffset = inputs[0].viewOffset;
}

REGISTER_REF_KERNEL_VIEW(OpType::REPEAT, 3, matchRepeatView, inferViewRepeat, {Backend::CPU, Backend::CUDA}, {DType::FLOAT32, DType::INT32, DType::INT32}, {{1}, {1}, {1}}, {false, false, false}, {{Backend::CPU, Backend::CUDA}, {Backend::CPU}, {Backend::CPU}});