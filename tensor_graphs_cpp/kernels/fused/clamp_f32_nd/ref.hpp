#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryClamp_F32_ND_CUDA(const std::vector<LogicalId> &inputs, Graph &graph)
{
    LogicalId x = inputs[0];
    LogicalId min_val = inputs[1];
    LogicalId max_val = inputs[2];

    const auto &shape = graph.getNode(x).getShape();

    // --- clamp_max ---
    LogicalId max_node = graph.fill(max_val, shape);
    LogicalId is_less = graph.lt(x, max_node);
    LogicalId is_less_f = graph.cast(is_less, DType::FLOAT32);

    float one_val = 1.0f;
    LogicalId fill_one_1 = graph.fill(one_val, shape);
    LogicalId not_less_f = graph.add(fill_one_1, graph.neg(is_less_f));

    LogicalId clamp_max_res = graph.add(graph.mul(x, is_less_f), graph.mul(max_node, not_less_f));

    // --- clamp_min ---
    LogicalId min_node = graph.fill(min_val, shape);
    LogicalId is_greater = graph.lt(min_node, clamp_max_res);
    LogicalId is_greater_f = graph.cast(is_greater, DType::FLOAT32);

    LogicalId fill_one_2 = graph.fill(one_val, shape);
    LogicalId not_greater_f = graph.add(fill_one_2, graph.neg(is_greater_f));

    return graph.add(graph.mul(clamp_max_res, is_greater_f), graph.mul(min_node, not_greater_f));
}
