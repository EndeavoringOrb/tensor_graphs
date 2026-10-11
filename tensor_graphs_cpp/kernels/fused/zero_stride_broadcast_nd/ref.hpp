#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryZeroStrideBroadcast_ND(const std::vector<LogicalId> &inputs, Graph &graph)
{
    if (inputs.size() != 1)
        Error::throw_err("ZeroStrideBroadcast requires exactly 1 input");

    TensorNode &node = graph.getNode(inputs[0]);
    if (!node.getShape().empty())
    {
        std::vector<uint64_t> strides(node.getShape().size(), 0);
        strides[0] = 1;
        node.strides = strides; // TODO: add dummy input strides to REGISTER_KERNEL
                                // macro so we don't have to do this hack
    }

    return graph.contiguous(inputs[0]);
}
