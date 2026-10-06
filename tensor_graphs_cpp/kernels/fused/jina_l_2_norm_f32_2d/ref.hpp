#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryJinaL2Norm_F32_2D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x_id = inputs[0];
    const auto &shape = g.getNode(x_id).getShape();
    uint32_t D = shape[1];

    // Helper: expand_scalar_to_2d(val, 1, 1) → {1, 1}
    // Mirrors JinaV5OmniNanoRetrievalModel::expand_scalar_to_2d(val, 1, 1).
    auto expand_scalar_11 = [&](float val) -> LogicalId {
        LogicalId node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh[] = {1, 1};
        return g.reshape(node, g.constant({2}, sh, DType::INT32));
        // dim0=1, dim1=1 → no repeats needed
    };

    // x_sq = x * x
    LogicalId x_sq = g.mul(x_id, x_id);

    // sum_sq = sum(x_sq, axis=-1)
    int32_t ax_val = -1;
    LogicalId axis_node = g.constant({1}, &ax_val, DType::INT32);
    LogicalId sum_sq = g.sum(x_sq, axis_node);

    // std = sqrt(sum_sq + eps)
    LogicalId eps_node = expand_scalar_11(1e-12f);
    LogicalId sum_sq_plus_eps = g.add(sum_sq, eps_node);
    LogicalId sqrt_node = expand_scalar_11(0.5f);
    LogicalId std = g.pow(sum_sq_plus_eps, sqrt_node);

    // inv_std = 1 / std
    LogicalId one_node = expand_scalar_11(1.0f);
    LogicalId inv_std = g.div(one_node, std);

    // inv_std_expanded = repeat(inv_std, D, axis=1)
    int32_t rep = (int32_t)D;
    int32_t ax = 1;
    LogicalId inv_std_expanded =
        g.repeat(inv_std, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));

    // result = x * inv_std_expanded
    return g.mul(x_id, inv_std_expanded);
}
