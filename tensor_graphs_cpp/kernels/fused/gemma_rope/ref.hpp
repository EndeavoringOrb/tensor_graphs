#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryGemmaRoPEEx(LogicalId x_id, LogicalId cos_id, LogicalId sin_id,
                                       uint32_t n_groups, uint32_t seq_len, uint32_t head_dim, Graph &g)
{
    uint32_t half_dim = head_dim / 2;

    int32_t starts1[] = {0, 0, 0};
    int32_t ends1[] = {static_cast<int32_t>(n_groups), static_cast<int32_t>(seq_len), static_cast<int32_t>(half_dim)};
    int32_t steps1[] = {1, 1, 1};
    LogicalId x1 = g.slice(x_id, g.constant({3}, starts1, DType::INT32), g.constant({3}, ends1, DType::INT32),
                           g.constant({3}, steps1, DType::INT32));
    x1 = g.contiguous(x1);

    int32_t starts2[] = {0, 0, static_cast<int32_t>(half_dim)};
    int32_t ends2[] = {static_cast<int32_t>(n_groups), static_cast<int32_t>(seq_len), static_cast<int32_t>(head_dim)};
    LogicalId x2 = g.slice(x_id, g.constant({3}, starts2, DType::INT32), g.constant({3}, ends2, DType::INT32),
                           g.constant({3}, steps1, DType::INT32));
    LogicalId neg_x2 = g.neg(x2);
    int32_t axis = 2;
    LogicalId rotated = g.concat({neg_x2, x1}, g.constant({1}, &axis, DType::INT32));
    LogicalId cos_expanded = g.repeat(cos_id, n_groups, 0);
    LogicalId sin_expanded = g.repeat(sin_id, n_groups, 0);
    LogicalId term1 = g.mul(x_id, cos_expanded);
    LogicalId term2 = g.mul(rotated, sin_expanded);
    return g.add(term1, term2);
}

inline LogicalId refFactoryGemmaRoPE(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x_id = inputs[0];
    LogicalId cos_id = inputs[1];
    LogicalId sin_id = inputs[2];
    const auto &shape = g.getNode(x_id).getShape();
    return refFactoryGemmaRoPEEx(x_id, cos_id, sin_id, shape[0], shape[1], shape[2], g);
}
