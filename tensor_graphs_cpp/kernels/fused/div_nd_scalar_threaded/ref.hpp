#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryDivND_Scalar_Threaded(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId idND = inputs[0];
    LogicalId idScalar = inputs[1];
    auto shapeND = graph.getNode(idND).getShape();

    std::vector<int32_t> ones(shapeND.size(), 1);
    LogicalId reshaped = graph.reshape(idScalar, graph.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));

    LogicalId out = reshaped;
    for (uint64_t i = 0; i < shapeND.size(); ++i)
    {
        if (shapeND[i] > 1)
        {
            int32_t rep = shapeND[i];
            int32_t ax = i;
            out = graph.repeat(out, graph.constant({1}, &rep, DType::INT32), graph.constant({1}, &ax, DType::INT32));
        }
    }
    return graph.div(idND, out);
}
