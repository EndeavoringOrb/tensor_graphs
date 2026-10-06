#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactorySilu(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x_id = inputs[0];
    const auto &target_shape = graph.getNode(x_id).getShape();
    auto broadcast = [&graph, &target_shape](LogicalId scalar_id) {
        std::vector<int32_t> ones(target_shape.size(), 1);
        LogicalId out = graph.reshape(scalar_id,
                                      graph.constant({(uint32_t)ones.size()}, ones.data(), DType::INT32));
        for (uint64_t i = 0; i < target_shape.size(); ++i)
        {
            if (target_shape[i] > 1)
            {
                int32_t repeat = (int32_t)target_shape[i];
                int32_t axis = (int32_t)i;
                out = graph.repeat(out, graph.constant({1}, &repeat, DType::INT32),
                                   graph.constant({1}, &axis, DType::INT32));
            }
        }
        return out;
    };
    LogicalId neg_x = graph.neg(x_id);
    float e_value = 2.7182818f;
    LogicalId e_node = broadcast(graph.constant({1}, &e_value, DType::FLOAT32));
    LogicalId exp_neg = graph.pow(e_node, neg_x);
    float one_value = 1.0f;
    LogicalId one_node = broadcast(graph.constant({1}, &one_value, DType::FLOAT32));
    LogicalId denominator = graph.add(one_node, exp_neg);
    LogicalId sigmoid = graph.div(one_node, denominator);
    return graph.mul(x_id, sigmoid);
}
