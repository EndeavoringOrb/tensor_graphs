#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryJinaGeluExact_F32_3D(const std::vector<LogicalId> &inputs, Graph &g)
{
    LogicalId x_id = inputs[0];
    const auto &shape = g.getNode(x_id).getShape();
    uint32_t S = shape[1];
    uint32_t D = shape[2];

    // Helper: expand_scalar_to_3d(val, 1, S, D) → {1, S, D}
    auto expand_scalar_SD = [&](float val) -> LogicalId {
        LogicalId node = g.constant({1}, &val, DType::FLOAT32);
        int32_t sh[] = {1, 1, 1};
        LogicalId out = g.reshape(node, g.constant({3}, sh, DType::INT32));
        if (S > 1)
        {
            int32_t rep = (int32_t)S;
            int32_t ax = 1;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));
        }
        if (D > 1)
        {
            int32_t rep = (int32_t)D;
            int32_t ax = 2;
            out = g.repeat(out, g.constant({1}, &rep, DType::INT32), g.constant({1}, &ax, DType::INT32));
        }
        return out;
    };

    // Create each expansion ONCE and reuse (matching the model's local variables)
    LogicalId inv_sqrt2 = expand_scalar_SD(0.7071067811865475f);
    LogicalId half = expand_scalar_SD(0.5f);
    LogicalId eps_node = expand_scalar_SD(1e-12f);
    LogicalId p_node = expand_scalar_SD(0.3275911f);
    LogicalId one_node = expand_scalar_SD(1.0f);
    LogicalId a1 = expand_scalar_SD(0.254829592f);
    LogicalId a2 = expand_scalar_SD(-0.284496736f);
    LogicalId a3 = expand_scalar_SD(1.421413741f);
    LogicalId a4 = expand_scalar_SD(-1.453152027f);
    LogicalId a5 = expand_scalar_SD(1.061405429f);
    LogicalId e_node = expand_scalar_SD(2.718281828459045f);

    // x_scaled = x / sqrt(2)  →  x * inv_sqrt2
    LogicalId x_scaled = g.mul(x_id, inv_sqrt2);

    // xs_sq = x_scaled^2
    LogicalId xs_sq = g.mul(x_scaled, x_scaled);

    // abs_xs = pow(xs_sq, 0.5)  = sqrt(xs_sq) = |x_scaled|
    LogicalId abs_xs = g.pow(xs_sq, half);

    // sign_xs = x_scaled / (|x_scaled| + eps)
    LogicalId abs_xs_eps = g.add(abs_xs, eps_node);
    LogicalId sign_xs = g.div(x_scaled, abs_xs_eps);

    // t = 1 / (1 + p * |x_scaled|)
    LogicalId p_abs = g.mul(p_node, abs_xs);
    LogicalId denom = g.add(one_node, p_abs);
    LogicalId t = g.div(one_node, denom);

    // t^2..t^5
    LogicalId t2 = g.mul(t, t);
    LogicalId t3 = g.mul(t2, t);
    LogicalId t4 = g.mul(t3, t);
    LogicalId t5 = g.mul(t4, t);

    // poly = a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    LogicalId poly = g.mul(a1, t);
    poly = g.add(poly, g.mul(a2, t2));
    poly = g.add(poly, g.mul(a3, t3));
    poly = g.add(poly, g.mul(a4, t4));
    poly = g.add(poly, g.mul(a5, t5));

    // exp(-x_scaled^2) = pow(e, neg(xs_sq))
    LogicalId neg_xs_sq = g.neg(xs_sq);
    LogicalId exp_neg_xs_sq = g.pow(e_node, neg_xs_sq);

    // erf_pos = 1 - poly * exp(-x_scaled^2)
    LogicalId product = g.mul(poly, exp_neg_xs_sq);
    LogicalId erf_pos = g.add(one_node, g.neg(product));

    // erf(x_scaled) = sign(x_scaled) * erf_pos
    LogicalId erf_val = g.mul(sign_xs, erf_pos);

    // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    LogicalId one_plus_erf = g.add(one_node, erf_val);
    LogicalId half_x = g.mul(x_id, half);
    return g.mul(half_x, one_plus_erf);
}
