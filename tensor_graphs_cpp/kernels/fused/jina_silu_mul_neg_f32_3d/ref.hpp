#pragma once

#include "core/kernels.hpp"

inline LogicalId refFactoryJinaSiluMulNeg_F32_3D(const std::vector<LogicalId> &inputs, Graph &g)
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

    // neg_one = expand_scalar_to_3d(-1.0, 1, S, D)
    LogicalId neg_one = expand_scalar_SD(-1.0f);

    // neg_x = mul(x, neg_one)   <-- mul by -1, NOT neg(x)
    LogicalId neg_x = g.mul(x_id, neg_one);

    // e_node = expand_scalar_to_3d(2.718281828459045, 1, S, D)
    LogicalId e_node = expand_scalar_SD(2.718281828459045f);

    // exp_neg = pow(e, -x)
    LogicalId exp_neg = g.pow(e_node, neg_x);

    // one_node = expand_scalar_to_3d(1.0, 1, S, D)
    LogicalId one_node = expand_scalar_SD(1.0f);

    // den = 1 + exp(-x)
    LogicalId den = g.add(one_node, exp_neg);

    // sig = 1 / den
    LogicalId sig = g.div(one_node, den);

    // result = x * sig
    return g.mul(x_id, sig);
}
