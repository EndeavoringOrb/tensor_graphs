#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryGemmaRMSNorm(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x_id = inputs[0];
    LogicalId weight_id = inputs[1];
    auto shape = g.getNode(x_id).getShape();
    uint32_t B = shape[0], S = shape[1], D = shape[2];

    LogicalId x_sq = g.mul(x_id, x_id);
    int32_t axis_val = -1;
    LogicalId axis_node = g.constant({1}, &axis_val, DType::INT32);
    LogicalId sum_sq = g.sum(x_sq, axis_node);

    float n_val = (float)D, eps_val = 1e-6f, half_val = 0.5f, one_val = 1.0f;

    auto expand = [&](float val, uint32_t last_d) {
        int32_t sh[] = {1, 1, 1};
        LogicalId out = g.reshape(g.constant({1}, &val, DType::FLOAT32), g.constant({3}, sh, DType::INT32));
        if (B > 1)
        {
            int32_t r = B, a = 0;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        if (S > 1)
        {
            int32_t r = S, a = 1;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        if (last_d > 1)
        {
            int32_t r = last_d, a = 2;
            out = g.repeat(out, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
        }
        return out;
    };

    LogicalId inv_std =
        g.div(expand(1.0f, 1), g.pow(g.add(g.div(sum_sq, expand(n_val, 1)), expand(eps_val, 1)), expand(half_val, 1)));

    int32_t r_d = D, a_d = 2;
    LogicalId x_norm =
        g.mul(x_id, g.repeat(inv_std, g.constant({1}, &r_d, DType::INT32), g.constant({1}, &a_d, DType::INT32)));

    int32_t sh_w[] = {1, 1, (int32_t)D};
    LogicalId w_exp = g.reshape(weight_id, g.constant({3}, sh_w, DType::INT32));
    if (B > 1)
    {
        int32_t r = B, a = 0;
        w_exp = g.repeat(w_exp, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }
    if (S > 1)
    {
        int32_t r = S, a = 1;
        w_exp = g.repeat(w_exp, g.constant({1}, &r, DType::INT32), g.constant({1}, &a, DType::INT32));
    }

    return g.mul(x_norm, g.add(w_exp, expand(1.0f, D)));
}
